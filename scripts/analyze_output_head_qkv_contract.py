#!/usr/bin/env python3
"""Test the output-head cosine-attention Q normalization contract."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
import re


LANES = 32
MMA_RECORD_BYTES = 40
E4_RECORD_BYTES = 8
HEAD_OFFSET = 147429888
FP8_MMA = re.compile(
    r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*"
    r"\{(?P<dst>[^}]+)\}", re.MULTILINE
)
REG = re.compile(r"%r\d+")
CONVERSION = re.compile(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+%rs\d+,\s*(%r\d+)")


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def half_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def half_mul(a: int, b: int) -> int:
    return half_bits(half_value(a) * half_value(b))


def half_add(a: int, b: int) -> int:
    return half_bits(half_value(a) + half_value(b))


def encode_e4m3(bits: int) -> int:
    sign = (bits >> 8) & 0x80
    exponent = (bits >> 10) & 31
    mantissa = bits & 1023
    if exponent == 31 and mantissa:
        return 0x7F

    def round_shift(value: int, shift: int) -> int:
        base = value >> shift
        remainder = value & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        return base + int(remainder > halfway or (remainder == halfway and base & 1))

    if exponent < 9:
        code = min(round_shift(1024 + mantissa, 16 - exponent), 8)
    else:
        rounded = round_shift(mantissa, 7)
        carry = rounded >= 8
        code = ((exponent - 8 + int(carry)) << 3) | (0 if carry else rounded)
        code = min(code, 0x7E)
    return sign | code


def trace_d(trace: bytes, mma: int, row: int, column: int) -> int:
    lane = (row % 8) * 4 + column // 2
    element = (2 if row >= 8 else 0) + (column & 1)
    offset = (mma * LANES + lane) * MMA_RECORD_BYTES + 32 + (element // 2) * 4
    word = struct.unpack_from("<I", trace, offset)[0]
    return (word >> (16 * (element & 1))) & 0xFFFF


def packed_a_half(trace: bytes, first_operation: int, row: int, k: int) -> int:
    lane = (row % 8) * 4 + (k % 16) // 4
    element = (k % 4) + (4 if row >= 8 else 0) + (8 if k >= 16 else 0)
    operation = first_operation + element // 2
    pair_half = element & 1
    offset = (operation * LANES + lane) * E4_RECORD_BYTES + pair_half * 2
    return struct.unpack_from("<H", trace, offset)[0]


def projection_value(trace: bytes, group: int, row: int, channel: int) -> int:
    mma = 128 + group * 12 + (channel // 16) * 2 + (channel % 16) // 8
    return trace_d(trace, mma, row, channel % 8)


def packed_k_source_channel(k: int) -> int:
    block, within = divmod(k, 16)
    return block * 16 + 2 * (within // 4) + (within & 1) + (8 if within & 2 else 0)


def reduce_half(values: list[int], mode: str) -> int:
    products = [half_mul(value, value) for value in values]
    if mode == "sequential":
        result = half_bits(0.0)
        for value in products:
            result = half_add(result, value)
        return result
    if mode == "balanced":
        while len(products) > 1:
            products = [half_add(products[i], products[i + 1])
                        for i in range(0, len(products), 2)]
        return products[0]
    if mode == "float_then_half":
        return half_bits(sum(half_value(value) for value in products))
    raise ValueError(mode)


def _word_binary(a: int, b: int, operation: str) -> int:
    result = 0
    for half_index in range(2):
        av = half_value((a >> (16 * half_index)) & 0xFFFF)
        bv = half_value((b >> (16 * half_index)) & 0xFFFF)
        value = av * bv if operation == "mul" else av + bv if operation == "add" else max(av, bv)
        result |= half_bits(value) << (16 * half_index)
    return result


def replay_q_ptx(ptx: str, mma_trace: bytes, e4_trace: bytes,
                 rsqrt_ulp_bias: int = 0) -> dict:
    register_values: dict[str, list[int]] = {}
    for mma_index, match in enumerate(FP8_MMA.finditer(ptx)):
        if not 128 <= mma_index <= 175:
            continue
        for word, register in enumerate(REG.findall(match.group("dst"))):
            register_values[register] = [
                struct.unpack_from("<I", mma_trace,
                                   (mma_index * LANES + lane) * MMA_RECORD_BYTES + 32 + word * 4)[0]
                for lane in range(LANES)
            ]
    epsilon_word = half_bits(6.199999916134402e-05) * 0x10001
    register_values["%r2720"] = [epsilon_word] * LANES
    marker = ptx.index("ld.global.b32 %r2839")
    end = ptx.index("mov.u32 %r2937", marker)
    block = ptx[marker:end]
    scale_match = re.search(r"ld\.global\.b32 %r2839[^\n]*\n.*?cvt\.rn\.f16\.f32 low, %r2839;.*?mov\.b32 %r2842", block, re.DOTALL)
    if not scale_match:
        raise ValueError("could not locate Q scale conversion")
    scale_word = half_bits(9.938164710998535) * 0x10001
    register_values["%r2842"] = [scale_word] * LANES

    patterns = [
        ("binary", re.compile(r"\{(mul|add|max)\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+);")),
        ("shfl", re.compile(r"\{shfl\.sync\.bfly\.b32\s+(%r\d+),\s*(%r\d+),\s*(%r2654|%r2659),")),
        ("swap", re.compile(r"mov\.b32\s+\{%rs\d+,\s*%rs\d+\},\s*(%r\d+);\s*mov\.b32\s+(%r\d+),\s*\{%rs\d+,\s*%rs\d+\};")),
        ("rsqrt", re.compile(r"mov\.b32\s+\{hl, hu\},\s*(%r\d+);.*?rsqrt\.approx\.ftz\.f32.*?mov\.b32\s+(%r\d+),\s*\{hl, hu\};", re.DOTALL)),
    ]
    events = sorted((match.start(), kind, match) for kind, pattern in patterns for match in pattern.finditer(block))
    for _, kind, match in events:
        if kind == "binary":
            operation, destination, left, right = match.groups()
            if left not in register_values or right not in register_values:
                continue
            register_values[destination] = [
                _word_binary(a, b, operation)
                for a, b in zip(register_values[left], register_values[right])
            ]
        elif kind == "shfl":
            destination, source, delta_register = match.groups()
            if source not in register_values:
                continue
            delta = 2 if delta_register == "%r2654" else 1
            register_values[destination] = [register_values[source][lane ^ delta]
                                            for lane in range(LANES)]
        elif kind == "swap":
            source, destination = match.groups()
            if source not in register_values:
                continue
            register_values[destination] = [((word & 0xFFFF) << 16) | (word >> 16)
                                            for word in register_values[source]]
        else:
            source, destination = match.groups()
            if source not in register_values:
                continue
            words = []
            for word in register_values[source]:
                result = 0
                for half_index in range(2):
                    value = half_value((word >> (16 * half_index)) & 0xFFFF)
                    exact_bits = struct.unpack("<I", struct.pack("<f", 1.0 / math.sqrt(value)))[0]
                    biased = struct.unpack("<f", struct.pack("<I", exact_bits + rsqrt_ulp_bias))[0]
                    result |= half_bits(biased) << (16 * half_index)
                words.append(result)
            register_values[destination] = words

    sources = [match.group(1) for match in CONVERSION.finditer(ptx)][196:228]
    mismatches = 0
    e4_mismatches = 0
    first = None
    missing = []
    for operation, source in enumerate(sources, 196):
        if source not in register_values:
            missing.append(source)
            continue
        for lane, actual in enumerate(register_values[source]):
            expected = struct.unpack_from("<I", e4_trace,
                                          (operation * LANES + lane) * E4_RECORD_BYTES)[0]
            if actual != expected:
                mismatches += sum(
                    ((actual >> (16 * h)) & 0xFFFF) != ((expected >> (16 * h)) & 0xFFFF)
                    for h in range(2)
                )
                if first is None:
                    first = {"operation": operation, "lane": lane,
                             "actual": f"0x{actual:08X}", "expected": f"0x{expected:08X}"}
            expected_e4 = e4_trace[(operation * LANES + lane) * E4_RECORD_BYTES + 4:
                                   (operation * LANES + lane) * E4_RECORD_BYTES + 6]
            actual_e4 = bytes((encode_e4m3(actual & 0xFFFF), encode_e4m3(actual >> 16)))
            e4_mismatches += sum(a != b for a, b in zip(actual_e4, expected_e4))
    return {"rsqrt_f32_ulp_bias": rsqrt_ulp_bias,
            "half_mismatches": mismatches, "e4m3_mismatches": e4_mismatches,
            "missing_registers": sorted(set(missing)),
            "first_mismatch": first}


def analyze(mma_path: Path, e4_path: Path, model_path: Path, ptx_path: Path | None = None) -> dict:
    mma = mma_path.read_bytes()
    e4 = e4_path.read_bytes()
    with model_path.open("rb") as stream:
        stream.seek(HEAD_OFFSET + 19664)
        scale_f32 = struct.unpack("<f", stream.read(4))[0]
    scale = half_bits(scale_f32)
    epsilon = half_bits(6.199999916134402e-05)
    modes = {}
    for mode in ("sequential", "balanced", "float_then_half"):
        mismatches = 0
        e4_mismatches = 0
        first = None
        for group in range(4):
            for row in range(16):
                values = [projection_value(mma, group, row, channel)
                          for channel in range(32)]
                sum_bits = reduce_half(values, mode)
                denominator = max(half_value(sum_bits), half_value(epsilon))
                inverse = half_bits(1.0 / math.sqrt(denominator))
                for channel in range(32):
                    value = values[packed_k_source_channel(channel)]
                    normalized = half_mul(half_mul(value, inverse), scale)
                    expected = packed_a_half(e4, 196 + group * 8, row, channel)
                    lane = (row % 8) * 4 + (channel % 16) // 4
                    element = (channel % 4) + (4 if row >= 8 else 0) + (8 if channel >= 16 else 0)
                    operation = 196 + group * 8 + element // 2
                    expected_e4 = e4[(operation * LANES + lane) * E4_RECORD_BYTES + 4 + (element & 1)]
                    e4_mismatches += encode_e4m3(normalized) != expected_e4
                    if normalized != expected:
                        mismatches += 1
                        if first is None:
                            first = {
                                "group": group, "row": row, "channel": channel,
                                "actual": f"0x{normalized:04X}",
                                "expected": f"0x{expected:04X}",
                                "sum": f"0x{sum_bits:04X}",
                                "inverse": f"0x{inverse:04X}",
                            }
        modes[mode] = {"half_mismatches": mismatches,
                       "e4m3_mismatches": e4_mismatches,
                       "first_mismatch": first}
    best = min(modes, key=lambda name: modes[name]["half_mismatches"])
    ptx_replay = None
    if ptx_path:
        ptx = ptx_path.read_text(encoding="utf-8")
        ptx_replay = {str(bias): replay_q_ptx(ptx, mma, e4, bias)
                      for bias in (-1, 0, 1)}
    return {
        "schema": 1,
        "experiment": "output_head_q_normalization_contract",
        "status": "PASS" if modes[best]["half_mismatches"] == 0 else "INCOMPLETE",
        "q_shape": [64, 32],
        "scale_f32": scale_f32,
        "scale_fp16": half_value(scale),
        "epsilon_fp16": half_value(epsilon),
        "candidates": modes,
        "best_candidate": best,
        "ptx_reduction_replay_with_exact_rsqrt": ptx_replay,
        "next_gate": "Match the exact FP16 shuffle reduction and RTX rsqrt rounding when no candidate is exact.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mma_trace", type=Path)
    parser.add_argument("e4_trace", type=Path)
    parser.add_argument("model_arena", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ptx", type=Path)
    args = parser.parse_args()
    report = analyze(args.mma_trace, args.e4_trace, args.model_arena, args.ptx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
