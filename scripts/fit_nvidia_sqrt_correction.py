#!/usr/bin/env python3
"""Fit an RTX 5070 correction over RX N0 first-sqrt output values."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

try:
    from scripts.fit_nvidia_cos_correction import evaluate
except ModuleNotFoundError:
    from fit_nvidia_cos_correction import evaluate


SAMPLES = 80 * 48 * 64
DOMAIN_MAX = np.float32(8.0)


def u32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fit_sqrt(rtx_boundary: Path, amd_boundary: Path, segment_bits: int,
             degree: int) -> dict:
    if not 6 <= segment_bits <= 16:
        raise ValueError("segment_bits must be in [6,16]")
    if not 0 <= degree <= 3:
        raise ValueError("degree must be in [0,3]")
    rtx = np.fromfile(rtx_boundary, dtype="<u4").reshape(-1, 2)[:, 0]
    amd = np.fromfile(amd_boundary, dtype="<u4").reshape(-1, 2)[:, 0]
    if rtx.shape != (SAMPLES,) or amd.shape != rtx.shape:
        raise ValueError("expected 245760 sqrt boundary samples")
    target = rtx.view("<f4")
    base = amd.view("<f4")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(base)):
        raise ValueError("sqrt traces must be finite")
    if np.any(base < 0) or np.any(base >= DOMAIN_MAX):
        raise ValueError("sqrt outputs are outside fitted [0,8) domain")

    segments = 1 << segment_bits
    scale = np.float32(segments / DOMAIN_MAX)
    position = np.float32(base * scale)
    segment = np.minimum(position.astype(np.int32), segments - 1)
    local = np.float32(position - segment.astype("<f4"))
    residual = target.astype(np.float64) - base.astype(np.float64)
    coefficients = np.zeros((segments, degree + 1), dtype="<f4")
    for current in np.unique(segment):
        mask = segment == current
        if int(mask.sum()) <= degree:
            coefficients[current, 0] = np.float32(np.mean(residual[mask]))
        else:
            coefficients[current] = np.polynomial.polynomial.polyfit(
                local[mask].astype(np.float64), residual[mask], degree
            ).astype("<f4")
    prediction = evaluate(base, coefficients, segment, local)
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_sqrt0_rx_correction_{segments}_segment_degree{degree}",
        "status": "PASS",
        "classification": "RTX5070_PRODUCTION_DOMAIN_EMPIRICAL_SQRT_SFU_CORRECTION",
        "target_site": "first_n0_sqrt_r221_r220",
        "index_value": "amd_sqrt_output_r221",
        "domain_max_u32": u32(float(DOMAIN_MAX)),
        "segments": segments,
        "segment_bits": segment_bits,
        "degree": degree,
        "scale_u32": u32(float(scale)),
        "sample_count": SAMPLES,
        "fp32_exact_samples": int(np.sum(prediction.view("<u4") == rtx)),
        "fp32_exact_fraction": float(np.mean(prediction.view("<u4") == rtx)),
        "coefficients_u32": [[u32(float(value)) for value in row] for row in coefficients],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-boundary", type=Path, required=True)
    parser.add_argument("--amd-boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-bits", type=int, default=14)
    parser.add_argument("--degree", type=int, default=1)
    args = parser.parse_args()
    model = fit_sqrt(args.rtx_boundary, args.amd_boundary, args.segment_bits, args.degree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "degree", "sample_count", "fp32_exact_fraction"
    )}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
