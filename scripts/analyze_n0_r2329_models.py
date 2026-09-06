#!/usr/bin/env python3
"""Enumerate FP16 rounding boundaries for the N0 r2329 square reduction."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import struct
from pathlib import Path

try:
    from scripts.analyze_slot3_mma_trace import candidate_inputs, decode_register_trace
except ModuleNotFoundError:
    from analyze_slot3_mma_trace import candidate_inputs, decode_register_trace


LANES = 32
MMA_LANE_BYTES = 40
MMA_INDICES = (140, 141, 142, 143)  # D0 registers r2152, r2156, r2160, r2164.


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def half_value(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def op(left: float, right: float, kind: str, rounded: bool) -> float:
    value = f32(left * right) if kind == "mul" else f32(left + right)
    return half_value(value) if rounded else value


def mma_d0(trace: bytes, mma: int, lane: int) -> tuple[float, float]:
    offset = (mma * LANES + lane) * MMA_LANE_BYTES + 32
    return struct.unpack("<2e", trace[offset:offset + 4])


def stage_words(path_trace: bytes, stage: int) -> list[int]:
    base = stage * LANES * 4
    return [struct.unpack("<I", path_trace[base + lane * 4:base + lane * 4 + 4])[0]
            for lane in range(LANES)]


def mma_accumulator(c_value: float, products: list[float], order: str) -> float:
    if order == "exact_fsum":
        return math.fsum([c_value, *products])
    if order.startswith("pairwise"):
        values = list(products) if order.endswith("c_last") else [c_value, *products]
        while len(values) > 1:
            if len(values) & 1:
                values.append(0.0)
            values = [f32(values[index] + values[index + 1]) for index in range(0, len(values), 2)]
        return f32(values[0] + c_value) if order.endswith("c_last") else values[0]
    if order.startswith("interleaved"):
        width = int(order.removeprefix("interleaved"))
        partials = [0.0] * width
        for index, product in enumerate(products):
            partials[index % width] = f32(partials[index % width] + product)
        while len(partials) > 1:
            partials = [f32(partials[index] + partials[index + 1]) for index in range(0, len(partials), 2)]
        return f32(partials[0] + c_value)
    sequence = list(reversed(products)) if order.startswith("reverse") else products
    accumulator = 0.0 if order.endswith("c_last") else c_value
    for product in sequence:
        accumulator = f32(accumulator + product)
    return f32(accumulator + c_value) if order.endswith("c_last") else accumulator


def evaluate(
    mma_trace: bytes,
    rtx_path: bytes,
    amd_path: bytes,
    *,
    mma_indices: tuple[int, ...] = MMA_INDICES,
    path_stage: int = 1,
    path_trace_bytes: int = 1280,
    experiment: str = "n0_r2329_rounding_boundary_enumeration",
) -> dict:
    if len(mma_trace) != 256 * LANES * MMA_LANE_BYTES:
        raise ValueError("unexpected all-MMA trace size")
    if len(rtx_path) != path_trace_bytes or len(amd_path) != path_trace_bytes:
        raise ValueError("unexpected normalization-path trace size")
    if len(mma_indices) != 4:
        raise ValueError("square reduction requires four MMA sources")
    visible_inputs = [[mma_d0(mma_trace, mma, lane) for lane in range(LANES)] for mma in mma_indices]
    decoded = decode_register_trace(mma_trace)
    mma_orders = (
        "sequential_c_first", "sequential_c_last", "reverse_c_first", "reverse_c_last",
        "interleaved2", "interleaved4", "interleaved8", "pairwise_c_first",
        "pairwise_c_last", "exact_fsum",
    )
    source_variants = {"visible_f16": visible_inputs}
    for order in mma_orders:
        order_inputs = []
        for mma in mma_indices:
            mma_lanes = []
            for lane in range(LANES):
                elements = []
                for element in range(2):
                    c_value, products, _ = candidate_inputs(decoded[mma], lane, element)
                    elements.append(mma_accumulator(c_value, products, order))
                mma_lanes.append(elements)
            order_inputs.append(mma_lanes)
        source_variants[f"mma_f32_{order}"] = order_inputs
    rtx_words = stage_words(rtx_path, path_stage)
    amd_words = stage_words(amd_path, path_stage)
    reports = []
    # Choices control rounding after square, pair sum, per-lane sum, xor-2 sum,
    # and xor-1 sum. The final cross-half sum is always rounded to FP16.
    for source_name, inputs in source_variants.items():
        for choices in itertools.product((False, True), repeat=5):
            square_round, pair_round, lane_round, xor2_round, xor1_round = choices
            lane_values = []
            for lane in range(LANES):
                squares = [[op(value, value, "mul", square_round) for value in inputs[reg][lane]]
                           for reg in range(4)]
                pair_a = [op(squares[1][half], squares[3][half], "add", pair_round) for half in range(2)]
                pair_b = [op(squares[0][half], squares[2][half], "add", pair_round) for half in range(2)]
                lane_values.append([op(pair_a[half], pair_b[half], "add", lane_round) for half in range(2)])
            xor2 = [[op(lane_values[lane][half], lane_values[lane ^ 2][half], "add", xor2_round)
                     for half in range(2)] for lane in range(LANES)]
            xor1 = [[op(xor2[lane][half], xor2[lane ^ 1][half], "add", xor1_round)
                     for half in range(2)] for lane in range(LANES)]
            predicted = []
            for lane in range(LANES):
                result = half_bits(op(xor1[lane][0], xor1[lane][1], "add", True))
                predicted.append(result | result << 16)
            rounded_names = "_".join(name for name, enabled in zip(
                ("square", "pair", "lane", "xor2", "xor1"), choices) if enabled)
            reports.append({
                "name": f"{source_name}_round_{rounded_names or 'final_only'}",
                "source_precision": source_name,
                "rounding": dict(zip(("square", "pair", "lane", "xor2", "xor1"), choices)),
                "word_mismatches_vs_rtx": sum(left != right for left, right in zip(predicted, rtx_words)),
                "word_mismatches_vs_amd": sum(left != right for left, right in zip(predicted, amd_words)),
                "predicted_words_hex": [f"{word:08X}" for word in predicted],
            })
    reports.sort(key=lambda item: (item["word_mismatches_vs_rtx"], item["word_mismatches_vs_amd"]))
    return {
        "schema": 1,
        "experiment": experiment,
        "status": "PASS",
        "candidate_count": len(reports),
        "exact_rtx_models": [item["name"] for item in reports if item["word_mismatches_vs_rtx"] == 0],
        "exact_amd_models": [item["name"] for item in reports if item["word_mismatches_vs_amd"] == 0],
        "best_models": reports[:10],
        "all_models": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mma-trace", required=True, type=Path)
    parser.add_argument("--rtx-path", required=True, type=Path)
    parser.add_argument("--amd-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.mma_trace.read_bytes(), args.rtx_path.read_bytes(), args.amd_path.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "exact_rtx_models", "exact_amd_models")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
