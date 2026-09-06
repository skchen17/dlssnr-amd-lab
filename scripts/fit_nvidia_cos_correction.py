#!/usr/bin/env python3
"""Fit a segmented correction from the RX cosine SFU to the RTX 5070 SFU."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


SAMPLES = 80 * 48 * 64
TWO_PI = 2.0 * np.pi


def u32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def evaluate(base: np.ndarray, coefficients: np.ndarray, segment: np.ndarray,
             local: np.ndarray) -> np.ndarray:
    selected = coefficients[segment]
    value = selected[:, -1].copy()
    for index in range(selected.shape[1] - 2, -1, -1):
        value = np.float32(np.float32(value * local) + selected[:, index])
    return np.float32(base + value)


def fit(rtx_path: Path, amd_path: Path, segment_bits: int, degree: int,
        force_samples: list[int]) -> dict:
    if not 6 <= segment_bits <= 20:
        raise ValueError("segment_bits must be in [6,20]")
    if not 0 <= degree <= 3:
        raise ValueError("degree must be in [0,3]")
    rtx = np.fromfile(rtx_path, dtype="<u4").reshape(-1, 2)
    amd = np.fromfile(amd_path, dtype="<u4").reshape(-1, 2)
    if rtx.shape != (SAMPLES, 2) or amd.shape != rtx.shape:
        raise ValueError("expected 245760 cosine input/output pairs")
    if not np.array_equal(rtx[:, 0], amd[:, 0]):
        raise ValueError("RTX/RX cosine inputs are not bitwise identical")
    segments = 1 << segment_bits
    scale = np.float32(segments / TWO_PI)
    x = rtx[:, 0].view("<f4")
    target = rtx[:, 1].view("<f4")
    base = amd[:, 1].view("<f4")
    position = np.float32(x * scale)
    segment = np.minimum(position.astype(np.int32), segments - 1)
    local = np.float32(position - segment.astype("<f4"))
    residual = target.astype(np.float64) - base.astype(np.float64)
    coefficients = np.zeros((segments, degree + 1), dtype="<f4")
    order = np.argsort(segment, kind="stable")
    sorted_segment = segment[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_segment)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        current = int(sorted_segment[start])
        indexes = order[start:end]
        if len(indexes) <= degree:
            coefficients[current, 0] = np.float32(np.mean(residual[indexes]))
        else:
            coefficients[current] = np.polynomial.polynomial.polyfit(
                local[indexes].astype(np.float64), residual[indexes], degree
            ).astype("<f4")

    forced = []
    for sample in force_samples:
        if not 0 <= sample < SAMPLES:
            raise ValueError(f"force sample out of range: {sample}")
        current = int(segment[sample])
        if any(row["segment"] == current for row in forced):
            raise ValueError("forced samples must occupy distinct segments")
        before = evaluate(base, coefficients, segment, local)[sample]
        for _ in range(8):
            predicted = evaluate(base[sample:sample + 1], coefficients,
                                 segment[sample:sample + 1], local[sample:sample + 1])[0]
            if u32(float(predicted)) == int(rtx[sample, 1]):
                break
            error = np.float32(target[sample] - predicted)
            previous = coefficients[current, 0]
            coefficients[current, 0] = np.float32(previous + error)
            if coefficients[current, 0] == previous:
                direction = np.float32(np.inf if error > 0 else -np.inf)
                coefficients[current, 0] = np.nextafter(previous, direction, dtype=np.float32)
        after = evaluate(base[sample:sample + 1], coefficients,
                         segment[sample:sample + 1], local[sample:sample + 1])[0]
        forced.append({
            "sample": sample,
            "segment": current,
            "target_bits": f"0x{int(rtx[sample, 1]):08X}",
            "before_bits": f"0x{u32(float(before)):08X}",
            "after_bits": f"0x{u32(float(after)):08X}",
            "exact_after": u32(float(after)) == int(rtx[sample, 1]),
        })

    prediction = evaluate(base, coefficients, segment, local)
    prediction_bits = prediction.view("<u4")
    target_half = target.astype("<f2").view("<u2")
    prediction_half = prediction.astype("<f2").view("<u2")
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_cos_rx_correction_{segments}_segment_degree{degree}",
        "status": "PASS" if all(row["exact_after"] for row in forced) else "FAIL",
        "classification": "RTX5070_PRODUCTION_DOMAIN_EMPIRICAL_COS_SFU_CORRECTION",
        "segments": segments,
        "segment_bits": segment_bits,
        "degree": degree,
        "scale_u32": u32(float(scale)),
        "sample_count": SAMPLES,
        "forced_samples": forced,
        "fp32_exact_samples": int(np.sum(prediction_bits == rtx[:, 1])),
        "fp32_exact_fraction": float(np.mean(prediction_bits == rtx[:, 1])),
        "fp16_mismatch_samples": int(np.sum(prediction_half != target_half)),
        "coefficients_u32": [[u32(float(value)) for value in row] for row in coefficients],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-bits", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--force-sample", action="append", type=int, default=[])
    args = parser.parse_args()
    model = fit(args.rtx, args.amd, args.segment_bits, args.degree, args.force_sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "degree", "sample_count", "fp32_exact_fraction",
        "fp16_mismatch_samples", "forced_samples")}, separators=(",", ":")))
    return 0 if model["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
