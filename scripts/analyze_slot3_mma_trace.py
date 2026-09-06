#!/usr/bin/env python3
"""Locate the first RTX/RX internal FP8 MMA fragment divergence in slot 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

try:
    from scripts.lower_ptx_fp8_mma import a_source, b_source, e4m3_constructed
except ModuleNotFoundError:
    from lower_ptx_fp8_mma import a_source, b_source, e4m3_constructed


MMA_COUNT = 256
LANES = 32
LANE_BYTES = 40
PRE_BYTES = 32
D_BYTES = 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def half_mismatches(left: bytes, right: bytes) -> int:
    count = len(left) // 2
    a = struct.unpack(f"<{count}H", left)
    b = struct.unpack(f"<{count}H", right)
    return sum(x != y for x, y in zip(a, b))


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def half_bits(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0xFC00 if math.copysign(1.0, value) < 0 else 0x7C00


def round_mantissa(value: float, bits: int) -> float:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    if (raw & 0x7F800000) == 0x7F800000:
        return value
    shift = 23 - bits
    lsb = (raw >> shift) & 1
    raw = (raw + (1 << (shift - 1)) - 1 + lsb) & 0xFFFFFFFF
    raw &= (0xFFFFFFFF << shift) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def truncate_mantissa(value: float, bits: int) -> float:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    if (raw & 0x7F800000) == 0x7F800000:
        return value
    shift = 23 - bits
    raw &= (0xFFFFFFFF << shift) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def decode_register_trace(data: bytes) -> list[list[tuple[int, ...]]]:
    records = []
    for mma in range(MMA_COUNT):
        lanes = []
        for lane in range(LANES):
            start = (mma * LANES + lane) * LANE_BYTES
            lanes.append(struct.unpack("<10I", data[start:start + LANE_BYTES]))
        records.append(lanes)
    return records


def byte_from_words(words: tuple[int, ...], element: int) -> int:
    return (words[element // 4] >> ((element & 3) * 8)) & 0xFF


def half_from_words(words: tuple[int, ...], element: int) -> int:
    return (words[element // 2] >> ((element & 1) * 16)) & 0xFFFF


def candidate_inputs(lanes: list[tuple[int, ...]], lane: int,
                     element: int) -> tuple[float, list[float], list[bool]]:
    c_bits = half_from_words(lanes[lane][6:8], element)
    c_value = half_value(c_bits)
    products = []
    subnormal_inputs = []
    for k in range(32):
        a_lane, a_element = a_source(lane, element, k)
        b_lane, b_element = b_source(lane, element, k)
        a_code = byte_from_words(lanes[a_lane][0:4], a_element)
        b_code = byte_from_words(lanes[b_lane][4:6], b_element)
        product = e4m3_constructed(a_code) * e4m3_constructed(b_code)
        products.append(f32(product))
        subnormal_inputs.append(
            ((a_code & 0x78) == 0 and (a_code & 0x07) != 0)
            or ((b_code & 0x78) == 0 and (b_code & 0x07) != 0)
        )
    return c_value, products, subnormal_inputs


def candidate_half_from_values(c_value: float, base_products: list[float],
                               subnormal_inputs: list[bool], model: dict) -> int:
    products = list(base_products)
    if model.get("flush_inputs"):
        products = [0.0 if subnormal else value
                    for value, subnormal in zip(products, subnormal_inputs)]
    if model.get("flush_products"):
        products = [0.0 if 0.0 < abs(value) < 2.0 ** -14 else value
                    for value in products]
    if model.get("product_precision") == "f16":
        products = [half_value(half_bits(value)) for value in products]

    if model.get("dot_round") == "f16":
        dot = 0.0
        for product in products:
            dot = f32(dot + product)
        dot = half_value(half_bits(dot))
        accumulator = f32(dot + c_value)
        if model.get("final_add") == "f16":
            accumulator = half_value(half_bits(accumulator))
        return half_bits(accumulator)

    order = model.get("order", "sequential_c_first")

    def step(accumulator: float, product: float) -> float:
        value = f32(accumulator + product)
        precision = model.get("mantissa")
        if precision == "f16":
            return half_value(half_bits(value))
        if isinstance(precision, int):
            if model.get("rounding") == "rz":
                return truncate_mantissa(value, precision)
            return round_mantissa(value, precision)
        return value

    if order == "exact_fsum":
        accumulator = math.fsum([c_value, *products])
    elif order.startswith("pairwise"):
        values = products if order.endswith("c_last") else [c_value, *products]
        while len(values) > 1:
            if len(values) & 1:
                values.append(0.0)
            values = [step(values[i], values[i + 1])
                      for i in range(0, len(values), 2)]
        accumulator = values[0]
        if order.endswith("c_last"):
            accumulator = step(accumulator, c_value)
    elif order.startswith("interleaved"):
        width = int(order.removeprefix("interleaved"))
        partials = [0.0] * width
        for k, product in enumerate(products):
            partials[k % width] = step(partials[k % width], product)
        while len(partials) > 1:
            partials = [f32(partials[i] + partials[i + 1])
                        for i in range(0, len(partials), 2)]
        accumulator = f32(partials[0] + c_value)
    else:
        sequence = list(reversed(products)) if order.startswith("reverse") else products
        accumulator = 0.0 if order.endswith("c_last") else c_value
        for product in sequence:
            accumulator = step(accumulator, product)
        if order.endswith("c_last"):
            accumulator = f32(accumulator + c_value)
    return half_bits(accumulator)


def candidate_half(lanes: list[tuple[int, ...]], lane: int, element: int,
                   model: dict) -> int:
    return candidate_half_from_values(*candidate_inputs(lanes, lane, element), model)


def candidate_models() -> list[dict]:
    return [
        {"name": "f32_sequential_c_first"},
        {"name": "f32_sequential_c_last", "order": "sequential_c_last"},
        {"name": "f32_reverse_c_first", "order": "reverse_c_first"},
        {"name": "f32_reverse_c_last", "order": "reverse_c_last"},
        *({"name": f"f32_interleaved{width}", "order": f"interleaved{width}"}
          for width in (2, 4, 8)),
        {"name": "f32_pairwise_c_first", "order": "pairwise_c_first"},
        {"name": "f32_pairwise_c_last", "order": "pairwise_c_last"},
        {"name": "exact_fsum", "order": "exact_fsum"},
        {"name": "f16_each_fma", "mantissa": "f16"},
        *({"name": f"mantissa{bits}_each_fma", "mantissa": bits}
          for bits in range(10, 23)),
        *({"name": f"mantissa{bits}_rz_each_fma", "mantissa": bits, "rounding": "rz"}
          for bits in range(10, 23)),
        {"name": "f16_products_f32_acc", "product_precision": "f16"},
        {"name": "f16_products_f16_acc", "product_precision": "f16", "mantissa": "f16"},
        {"name": "f16_dot_then_f32_add_c", "dot_round": "f16"},
        {"name": "f16_dot_then_f16_add_c", "dot_round": "f16", "final_add": "f16"},
        {"name": "flush_e4m3_inputs", "flush_inputs": True},
        {"name": "flush_f16_subnormal_products", "flush_products": True},
    ]


def evaluate_models(reference_trace: bytes, observed_trace: bytes) -> tuple[list[dict], bool]:
    reference = decode_register_trace(reference_trace)
    observed = decode_register_trace(observed_trace)
    models = candidate_models()
    mismatches = [0] * len(models)
    first_mismatches = [None] * len(models)
    observed_model_gate = True
    for mma, lanes in enumerate(reference):
        current = [0] * len(models)
        for lane in range(LANES):
            for element in range(4):
                inputs = candidate_inputs(lanes, lane, element)
                rtx_bits = half_from_words(lanes[lane][8:10], element)
                amd_bits = half_from_words(observed[mma][lane][8:10], element)
                for index, model in enumerate(models):
                    predicted = candidate_half_from_values(*inputs, model)
                    current[index] += predicted != rtx_bits
                    if index == 0:
                        observed_model_gate &= predicted == amd_bits
        for index, count in enumerate(current):
            mismatches[index] += count
            if count and first_mismatches[index] is None:
                first_mismatches[index] = mma
    reports = [
        {"name": model["name"], "d_half_mismatches_vs_rtx": mismatches[index],
         "first_mismatch_mma": first_mismatches[index], "exact": mismatches[index] == 0}
        for index, model in enumerate(models)
    ]
    reports.sort(key=lambda item: item["d_half_mismatches_vs_rtx"])
    return reports, observed_model_gate


def analyze(rtx_path: Path, amd_path: Path, *, include_models: bool = True) -> dict:
    rtx = rtx_path.read_bytes()
    amd = amd_path.read_bytes()
    expected = MMA_COUNT * LANES * LANE_BYTES
    if len(rtx) != expected or len(amd) != expected:
        raise ValueError(f"trace size must be {expected} bytes")
    per_mma = []
    first_pre = None
    first_d = None
    first_pre_details = []
    first_d_details = []
    total_pre_bytes = total_d_bytes = total_d_halves = 0
    for mma in range(MMA_COUNT):
        pre_mismatch = d_byte_mismatch = d_half_mismatch = 0
        for lane in range(LANES):
            start = (mma * LANES + lane) * LANE_BYTES
            r_pre = rtx[start:start + PRE_BYTES]
            a_pre = amd[start:start + PRE_BYTES]
            r_d = rtx[start + PRE_BYTES:start + PRE_BYTES + D_BYTES]
            a_d = amd[start + PRE_BYTES:start + PRE_BYTES + D_BYTES]
            pre_mismatch += sum(x != y for x, y in zip(r_pre, a_pre))
            d_byte_mismatch += sum(x != y for x, y in zip(r_d, a_d))
            d_half_mismatch += half_mismatches(r_d, a_d)
            if first_pre is None:
                for byte, (rtx_value, amd_value) in enumerate(zip(r_pre, a_pre)):
                    if rtx_value == amd_value:
                        continue
                    word = byte // 4
                    if word < 4:
                        fragment, register = "A", word
                    elif word < 6:
                        fragment, register = "B", word - 4
                    else:
                        fragment, register = "C", word - 6
                    first_pre_details.append({
                        "mma": mma, "lane": lane, "fragment": fragment,
                        "register": register, "byte_in_register": byte & 3,
                        "rtx_hex": f"{rtx_value:02X}", "amd_hex": f"{amd_value:02X}",
                    })
            if first_d is None:
                for element in range(4):
                    offset = element * 2
                    rtx_bits = int.from_bytes(r_d[offset:offset + 2], "little")
                    amd_bits = int.from_bytes(a_d[offset:offset + 2], "little")
                    if rtx_bits != amd_bits:
                        first_d_details.append({
                            "mma": mma, "lane": lane, "element": element,
                            "rtx_hex": f"{rtx_bits:04X}", "amd_hex": f"{amd_bits:04X}",
                        })
        if pre_mismatch and first_pre is None:
            first_pre = mma
        if d_half_mismatch and first_d is None:
            first_d = mma
        total_pre_bytes += pre_mismatch
        total_d_bytes += d_byte_mismatch
        total_d_halves += d_half_mismatch
        per_mma.append({"mma": mma, "pre_fragment_byte_mismatches": pre_mismatch,
                        "d_byte_mismatches": d_byte_mismatch,
                        "d_half_mismatches": d_half_mismatch})
    pre_gate = total_pre_bytes == 0
    result_gate = total_d_halves == 0
    model_reports, amd_model_gate = evaluate_models(rtx, amd) if include_models else ([], None)
    return {
        "schema": 1,
        "experiment": "slot3_internal_fp8_mma_rtx5070_vs_rx9070xt",
        "status": "PASS" if pre_gate else "FAIL",
        "classification": "FIRST_INTERNAL_NUMERICAL_DIVERGENCE_LOCALIZATION",
        "input_fragment_gate": pre_gate,
        "mma_result_bitwise_gate": result_gate,
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
        "mma_count": MMA_COUNT,
        "pre_fragment_byte_mismatches": total_pre_bytes,
        "d_byte_mismatches": total_d_bytes,
        "d_half_mismatches": total_d_halves,
        "first_pre_fragment_mismatch_mma": first_pre,
        "first_pre_fragment_mismatches": first_pre_details,
        "first_d_mismatch_mma": first_d,
        "first_d_mismatches": first_d_details,
        "amd_scalar_model_gate": amd_model_gate,
        "candidate_model_count": len(model_reports),
        "candidate_model_basis": (
            "PTX leaves MMA accumulation order, rounding and subnormal handling unspecified; "
            "the candidates cover FP32/exact/tree orders, RN/RZ limited mantissas, staged FP16 "
            "products/dot sums and subnormal flushing."
        ),
        "candidate_accumulation_models": model_reports,
        "exact_candidate_models": [item["name"] for item in model_reports if item["exact"]],
        "per_mma": per_mma,
        "interpretation": (
            "A/B/C are bitwise identical; the first D mismatch directly identifies the "
            "first RTX-vs-emulated Tensor Core arithmetic difference."
            if pre_gate else
            "A/B/C already differ, so divergence precedes the corresponding MMA."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", required=True, type=Path)
    parser.add_argument("--amd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.rtx.resolve(), args.amd.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "input_fragment_gate", "mma_result_bitwise_gate",
        "d_half_mismatches", "first_d_mismatch_mma")}, separators=(",", ":")))
    return 0 if report["input_fragment_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
