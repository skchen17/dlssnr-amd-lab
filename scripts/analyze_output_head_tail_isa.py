#!/usr/bin/env python3
"""Compare candidate RX 9070 XT output-tail arithmetic with a captured oracle."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path

import numpy as np


GRID_CTA_COUNT = 81 * 49
TOKENS = 64
INPUT_CHANNELS = 32
OUTPUT_CHANNELS = 4
HEAD_OFFSET = 147_429_888
HEAD_BYTES = 21_808


def load_weights(model_path: Path) -> np.ndarray:
    model = np.memmap(model_path, dtype=np.uint8, mode="r")
    head = model[HEAD_OFFSET : HEAD_OFFSET + HEAD_BYTES]
    weights = np.empty((INPUT_CHANNELS, OUTPUT_CHANNELS), dtype=np.float16)
    for k in range(INPUT_CHANNELS):
        tile = 20_784 + (k // 16) * 512
        kk = k % 16
        for channel in range(OUTPUT_CHANNELS):
            lane = channel * 4 + (kk % 8) // 2
            element = (kk % 2) + (2 if kk >= 8 else 0)
            offset = tile + lane * 16 + element * 2
            bits = int(head[offset]) | (int(head[offset + 1]) << 8)
            weights[k, channel] = np.array([bits], dtype=np.uint16).view(np.float16)[0]
    return weights


def load_fmaf():
    runtime = ctypes.CDLL("ucrtbase")
    fmaf = runtime.fmaf
    fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fmaf.restype = ctypes.c_float
    return fmaf


def fp32_fma(fmaf, a: float, b: float, c: float) -> float:
    return float(fmaf(float(a), float(b), float(c)))


def half_bits(value: float) -> int:
    with np.errstate(over="ignore", invalid="ignore"):
        return int(np.asarray([value], dtype=np.float16).view(np.uint16)[0])


def half_value(bits: int) -> float:
    return float(np.asarray([bits], dtype=np.uint16).view(np.float16)[0])


def project_isa(
    fmaf,
    values: np.ndarray,
    weights: np.ndarray,
    channel: int,
    first_round: str = "direct",
    dot_mode: str = "exact",
) -> int:
    # op_sel shows that gfx1201 consumes the low half and then the high half.
    acc = fp32_fma(fmaf, values[0], weights[0, channel], 0.0)
    acc = fp32_fma(fmaf, values[1], weights[1, channel], acc)
    for k in range(2, 14, 2):
        acc = fp32_fma(fmaf, values[k], weights[k, channel], acc)
        acc = fp32_fma(fmaf, values[k + 1], weights[k + 1, channel], acc)

    # v_fma_mixlo_f16 rounds the exact final FMA directly to binary16.
    acc = fp32_fma(fmaf, values[14], weights[14, channel], acc)
    exact_last = math.fma(float(values[15]), float(weights[15, channel]), acc)
    if first_round == "double":
        exact_last = float(np.float32(exact_last))
    acc = half_value(half_bits(exact_last))

    # The first two packed pairs in the second half remain explicit mix-FMAs.
    for k in (16, 18):
        acc = fp32_fma(fmaf, values[k], weights[k, channel], acc)
        acc = fp32_fma(fmaf, values[k + 1], weights[k + 1, channel], acc)

    # v_dot2_f32_f16 performs both exact half products and one FP32 accumulation.
    for k in range(20, 32, 2):
        low_product = float(values[k]) * float(weights[k, channel])
        high_product = float(values[k + 1]) * float(weights[k + 1, channel])
        if dot_mode == "exact":
            acc = float(np.float32(math.fsum((acc, low_product, high_product))))
        elif dot_mode == "pair_then_acc":
            pair = float(np.float32(low_product + high_product))
            acc = float(np.float32(acc + pair))
        elif dot_mode == "low_high":
            acc = fp32_fma(fmaf, values[k], weights[k, channel], acc)
            acc = fp32_fma(fmaf, values[k + 1], weights[k + 1, channel], acc)
        elif dot_mode == "high_low":
            acc = fp32_fma(fmaf, values[k + 1], weights[k + 1, channel], acc)
            acc = fp32_fma(fmaf, values[k], weights[k, channel], acc)
        elif dot_mode == "product_then_fma_low":
            acc = fp32_fma(fmaf, values[k], weights[k, channel], float(np.float32(acc + high_product)))
        elif dot_mode == "product_then_fma_high":
            acc = fp32_fma(fmaf, values[k + 1], weights[k + 1, channel], float(np.float32(acc + low_product)))
        else:
            raise ValueError(f"unknown dot mode: {dot_mode}")
    return half_bits(acc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("projected", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    projected = np.memmap(args.projected, dtype=np.float16, mode="r").reshape(
        GRID_CTA_COUNT, TOKENS, INPUT_CHANNELS
    )
    expected = np.memmap(args.expected, dtype=np.uint16, mode="r").reshape(
        GRID_CTA_COUNT, TOKENS, OUTPUT_CHANNELS
    )
    weights = load_weights(args.model)
    fmaf = load_fmaf()

    if args.trace:
        trace_bits = np.memmap(args.trace, dtype=np.uint32, mode="r").reshape(-1, 7)
        trace_values = trace_bits.view(np.float32)
        flat_input = projected.reshape(-1, INPUT_CHANNELS)
        transition_report = []
        for pair in range(6):
            k = 20 + pair * 2
            acc = trace_values[:, pair].astype(np.float64)
            low = np.repeat(flat_input[:, k], OUTPUT_CHANNELS).astype(np.float64)
            high = np.repeat(flat_input[:, k + 1], OUTPUT_CHANNELS).astype(np.float64)
            weight_low = np.tile(weights[k].astype(np.float64), flat_input.shape[0])
            weight_high = np.tile(weights[k + 1].astype(np.float64), flat_input.shape[0])
            low_product = low * weight_low
            high_product = high * weight_high
            actual = trace_bits[:, pair + 1]
            exact = (acc + low_product + high_product).astype(np.float32).view(np.uint32)
            pair_then_acc = (
                acc.astype(np.float32)
                + (low_product + high_product).astype(np.float32)
            ).view(np.uint32)
            low_high = (
                (acc + low_product).astype(np.float32).astype(np.float64) + high_product
            ).astype(np.float32).view(np.uint32)
            high_low = (
                (acc + high_product).astype(np.float32).astype(np.float64) + low_product
            ).astype(np.float32).view(np.uint32)
            half_products = (
                acc
                + low_product.astype(np.float16).astype(np.float64)
                + high_product.astype(np.float16).astype(np.float64)
            ).astype(np.float32).view(np.uint32)
            transition_report.append(
                {
                    "pair": pair,
                    "k": k,
                    "exact_mismatches": int(np.count_nonzero(exact != actual)),
                    "pair_then_acc_mismatches": int(np.count_nonzero(pair_then_acc != actual)),
                    "low_high_mismatches": int(np.count_nonzero(low_high != actual)),
                    "high_low_mismatches": int(np.count_nonzero(high_low != actual)),
                    "half_products_mismatches": int(np.count_nonzero(half_products != actual)),
                }
            )
        print(json.dumps({"trace_transitions": transition_report}, indent=2))
        return 0

    if args.candidate:
        candidate = np.memmap(args.candidate, dtype=np.uint16, mode="r").reshape(expected.shape)
        indices = np.argwhere(candidate != expected)
    else:
        indices = np.indices(expected.shape).reshape(3, -1).T
    if args.limit:
        indices = indices[: args.limit]

    modes = [("direct", "exact")]
    if args.sweep:
        modes = [
            (first_round, dot_mode)
            for first_round in ("direct", "double")
            for dot_mode in (
                "exact",
                "pair_then_acc",
                "low_high",
                "high_low",
                "product_then_fma_low",
                "product_then_fma_high",
            )
        ]
    mismatch_counts = {f"{first_round}/{dot_mode}": 0 for first_round, dot_mode in modes}
    mismatches = []
    checked = 0
    for cta, token, channel in indices:
        oracle = int(expected[cta, token, channel])
        checked += 1
        actual = 0
        for first_round, dot_mode in modes:
            actual = project_isa(
                fmaf,
                projected[cta, token],
                weights,
                int(channel),
                first_round,
                dot_mode,
            )
            mismatch_counts[f"{first_round}/{dot_mode}"] += actual != oracle
        if actual != oracle and not args.sweep:
            mismatches.append(
                {
                    "cta": int(cta),
                    "token": int(token),
                    "channel": int(channel),
                    "actual_bits": actual,
                    "expected_bits": oracle,
                }
            )

    report = {
        "checked": checked,
        "mismatches": mismatch_counts,
        "first_mismatches": mismatches[:16],
    }
    print(json.dumps(report, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
