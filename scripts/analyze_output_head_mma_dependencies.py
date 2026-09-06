#!/usr/bin/env python3
"""Recover output-head FP8 MMA fragment sources and accumulator chains.

The report joins the selected-CTA RTX MMA trace, the earlier E4M3 conversion
trace, and the decoded model arena.  It does not assume register names: exact
warp-wide fragment bytes are matched instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


HEAD_OFFSET = 147429888
HEAD_BYTES = 21808
LANES = 32
MMA_RECORD_BYTES = 40
E4_RECORD_BYTES = 8
FP8_MMAS = 256
E4_OPERATIONS = 388


def _mma_fragment(trace: bytes, mma: int, byte_offset: int, byte_count: int) -> bytes:
    return b"".join(
        trace[(mma * LANES + lane) * MMA_RECORD_BYTES + byte_offset:
              (mma * LANES + lane) * MMA_RECORD_BYTES + byte_offset + byte_count]
        for lane in range(LANES)
    )


def _e4_a_fragment(trace: bytes, first_operation: int) -> bytes:
    # Eight consecutive x2 conversions form the four 32-bit A registers in
    # each lane.  Each record stores the b16 conversion result at byte 4.
    return b"".join(
        b"".join(
            trace[(operation * LANES + lane) * E4_RECORD_BYTES + 4:
                  (operation * LANES + lane) * E4_RECORD_BYTES + 6]
            for operation in range(first_operation, first_operation + 8)
        )
        for lane in range(LANES)
    )


def _e4_b_fragment(trace: bytes, first_operation: int) -> bytes:
    return b"".join(
        b"".join(
            trace[(operation * LANES + lane) * E4_RECORD_BYTES + 4:
                  (operation * LANES + lane) * E4_RECORD_BYTES + 6]
            for operation in range(first_operation, first_operation + 4)
        )
        for lane in range(LANES)
    )


def _raw_half_a_fragment(trace: bytes, first_operation: int) -> bytes:
    return b"".join(
        b"".join(
            trace[(operation * LANES + lane) * E4_RECORD_BYTES:
                  (operation * LANES + lane) * E4_RECORD_BYTES + 4]
            for operation in range(first_operation, first_operation + 8)
        )
        for lane in range(LANES)
    )


def _e4_input_c_fragment(trace: bytes, first_operation: int) -> bytes:
    return b"".join(
        b"".join(
            trace[(operation * LANES + lane) * E4_RECORD_BYTES:
                  (operation * LANES + lane) * E4_RECORD_BYTES + 4]
            for operation in range(first_operation, first_operation + 2)
        )
        for lane in range(LANES)
    )


def _half_mul_bits(a_bits: int, b_bits: int) -> int:
    a = struct.unpack("<e", struct.pack("<H", a_bits))[0]
    b = struct.unpack("<e", struct.pack("<H", b_bits))[0]
    return struct.unpack("<H", struct.pack("<e", a * b))[0]


def _seed_fragment(raw: bytes, scales: bytes, mma_in_group: int) -> bytes:
    operation_pairs = ((0, 2), (1, 3), (4, 6), (5, 7))
    first_operation, second_operation = operation_pairs[mma_in_group]
    result = bytearray()
    for lane in range(LANES):
        scale_word = mma_in_group * 4 + lane % 4
        for operation in (first_operation, second_operation):
            raw_word_offset = lane * 32 + operation * 4
            scale_word_offset = scale_word * 4
            for pair_half in range(2):
                a_bits = struct.unpack_from("<H", raw, raw_word_offset + pair_half * 2)[0]
                scale_bits = struct.unpack_from(
                    "<H", scales, scale_word_offset + pair_half * 2)[0]
                result += struct.pack("<H", _half_mul_bits(a_bits, scale_bits))
    return bytes(result)


def _model_b_fragment(head: bytes, tile_offset: int, n_half: int) -> bytes:
    return b"".join(
        head[tile_offset + lane * 16 + n_half * 8:
             tile_offset + lane * 16 + n_half * 8 + 8]
        for lane in range(LANES)
    )


def analyze(mma_trace_path: Path, e4_trace_path: Path, model_arena_path: Path) -> dict:
    mma_trace = mma_trace_path.read_bytes()
    e4_trace = e4_trace_path.read_bytes()
    if len(mma_trace) < FP8_MMAS * LANES * MMA_RECORD_BYTES:
        raise ValueError("MMA trace does not contain all 256 FP8 operations")
    if len(e4_trace) < E4_OPERATIONS * LANES * E4_RECORD_BYTES:
        raise ValueError("E4 trace does not contain all 388 conversions")
    with model_arena_path.open("rb") as stream:
        stream.seek(HEAD_OFFSET)
        head = stream.read(HEAD_BYTES)
    if len(head) != HEAD_BYTES:
        raise ValueError("model arena does not contain the complete output head")

    raw_fused_groups = [_raw_half_a_fragment(e4_trace, 4 + group * 8)
                        for group in range(4)]
    residual_scales = head[8208:8272]
    seed_mismatches = 0
    for mma in range(16, 32):
        local = mma - 16
        group = local // 4
        expected = _seed_fragment(raw_fused_groups[group], residual_scales,
                                  local % 4)
        actual = _mma_fragment(mma_trace, mma, 24, 8)
        seed_mismatches += sum(got != want for got, want in zip(actual, expected))

    a_sources: dict[bytes, list[int]] = {}
    for operation in range(E4_OPERATIONS - 7):
        a_sources.setdefault(_e4_a_fragment(e4_trace, operation), []).append(operation)
    dynamic_b_sources: dict[bytes, list[int]] = {}
    for operation in range(E4_OPERATIONS - 3):
        dynamic_b_sources.setdefault(_e4_b_fragment(e4_trace, operation), []).append(operation)
    half_c_sources: dict[bytes, list[int]] = {}
    for operation in range(E4_OPERATIONS - 1):
        half_c_sources.setdefault(_e4_input_c_fragment(e4_trace, operation), []).append(operation)

    b_sources: dict[bytes, list[dict[str, int]]] = {}
    for tile_offset in range(0, HEAD_BYTES - 511, 16):
        for n_half in range(2):
            fragment = _model_b_fragment(head, tile_offset, n_half)
            b_sources.setdefault(fragment, []).append({
                "head_offset": tile_offset,
                "n_half": n_half,
            })

    d_sources: dict[bytes, list[int]] = {}
    operations = []
    zero_c = bytes(LANES * 8)
    for mma in range(FP8_MMAS):
        a = _mma_fragment(mma_trace, mma, 0, 16)
        b = _mma_fragment(mma_trace, mma, 16, 8)
        c = _mma_fragment(mma_trace, mma, 24, 8)
        d = _mma_fragment(mma_trace, mma, 32, 8)
        prior_d = [source for source in d_sources.get(c, []) if source < mma]
        operations.append({
            "mma": mma,
            "a_e4_first_operations": a_sources.get(a, []),
            "b_model_sources": b_sources.get(b, []),
            "b_e4_first_operations": dynamic_b_sources.get(b, []),
            "c_source": (
                {"kind": "zero"} if c == zero_c else
                {"kind": "prior_mma_d", "mma": prior_d[-1]} if prior_d else
                {"kind": "external_fp16_seed"}
            ),
            "c_e4_input_first_operations": half_c_sources.get(c, []),
        })
        d_sources.setdefault(d, []).append(mma)

    first128 = operations[:128]
    remaining = operations[128:]
    a_mapped = sum(bool(entry["a_e4_first_operations"]) for entry in first128)
    b_mapped = sum(bool(entry["b_model_sources"]) for entry in first128)
    c_zero = sum(entry["c_source"]["kind"] == "zero" for entry in first128)
    c_chained = sum(entry["c_source"]["kind"] == "prior_mma_d" for entry in first128)
    c_external = sum(entry["c_source"]["kind"] == "external_fp16_seed" for entry in first128)
    remaining_summary = {
        "a_fragments_mapped_to_e4_trace": sum(bool(entry["a_e4_first_operations"]) for entry in remaining),
        "b_fragments_mapped_to_model_arena": sum(bool(entry["b_model_sources"]) for entry in remaining),
        "b_fragments_mapped_to_e4_trace": sum(bool(entry["b_e4_first_operations"]) for entry in remaining),
        "b_fragments_resolved": sum(
            bool(entry["b_model_sources"] or entry["b_e4_first_operations"])
            for entry in remaining
        ),
        "zero_accumulators": sum(entry["c_source"]["kind"] == "zero" for entry in remaining),
        "prior_mma_accumulators": sum(entry["c_source"]["kind"] == "prior_mma_d" for entry in remaining),
        "external_fp16_seeds": sum(entry["c_source"]["kind"] == "external_fp16_seed" for entry in remaining),
    }
    chains = [
        {"mma_range": [0, 15], "c": "zero"},
        {"mma_range": [16, 31], "c": "external_fp16_seed"},
        {"mma_range": [32, 47], "c": "zero"},
        {"mma_range": [48, 63], "c": "mma_16_31_d"},
        {"mma_range": [64, 79], "c": "zero"},
        {"mma_range": [80, 95], "c": "mma_48_63_d"},
        {"mma_range": [96, 111], "c": "zero"},
        {"mma_range": [112, 127], "c": "mma_80_95_d"},
    ]
    expected_counts = (128, 128, 64, 48, 16)
    actual_counts = (a_mapped, b_mapped, c_zero, c_chained, c_external)
    passed = actual_counts == expected_counts and seed_mismatches == 0
    return {
        "schema": 1,
        "experiment": "output_head_first_128_mma_dependencies",
        "status": "PASS" if passed else "FAIL",
        "classification": "RTX_TRACE_VERIFIED_MMA_DATAFLOW_CONTRACT",
        "coverage": {"mma_range": [0, 127], "mma_count": 128},
        "summary": {
            "a_fragments_mapped_to_e4_trace": a_mapped,
            "b_fragments_mapped_to_model_arena": b_mapped,
            "zero_accumulators": c_zero,
            "prior_mma_accumulators": c_chained,
            "external_fp16_seeds": c_external,
            "raw_residual_seed_byte_mismatches": seed_mismatches,
        },
        "remaining_128_summary": remaining_summary,
        "accumulator_chains": chains,
        "operations": first128,
        "remaining_operations": remaining,
        "sha256": {
            "mma_trace": hashlib.sha256(mma_trace).hexdigest().upper(),
            "e4_trace": hashlib.sha256(e4_trace).hexdigest().upper(),
            "model_arena": hashlib.sha256(model_arena_path.read_bytes()).hexdigest().upper(),
        },
        "residual_seed": {
            "source": "raw fused main-plus-skip FP16 values",
            "scale_head_offset": 8208,
            "scale_shape": [32],
            "operation": "fp16_mul",
            "rtx_bytes_checked": 16 * LANES * 8,
            "byte_mismatches": seed_mismatches,
        },
        "next_gate": "Recover the Q/K/V spatial assignment and 48 FP16 seed fragments for dynamic MMAs 176 through 255.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mma_trace", type=Path)
    parser.add_argument("e4_trace", type=Path)
    parser.add_argument("model_arena", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.mma_trace, args.e4_trace, args.model_arena)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
