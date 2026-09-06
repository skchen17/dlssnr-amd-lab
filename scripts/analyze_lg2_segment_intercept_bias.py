#!/usr/bin/env python3
"""Score local intercept ULP changes for selected NVIDIA lg2 model segments."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def f32_from_u32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def evaluate_segment(xbits: np.ndarray, y: np.ndarray, row: list[int], segment_bits: int) -> np.ndarray:
    local_bits = 23 - segment_bits
    mantissa = xbits & 0x7FFFFF
    t = ((mantissa & ((1 << local_bits) - 1)).astype(np.float32)
         * np.float32(2.0 ** -local_bits))
    exponent = (((xbits >> 23) & 255).astype(np.int32) - 127).astype(np.float32)
    coefficients = np.asarray([f32_from_u32(value) for value in row], dtype=np.float32)
    q = np.full(len(t), coefficients[-1], dtype=np.float32)
    for index in range(len(coefficients) - 2, -1, -1):
        q = np.float32(np.float32(q * t) + coefficients[index])
    return np.float32(q + exponent)


def analyze(trace: Path, model_path: Path, segments: list[int], radius: int) -> dict:
    raw = np.fromfile(trace, dtype="<u4").reshape(-1, 2)
    xbits = raw[:, 0]
    y = raw[:, 1].view("<f4")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    segment_bits = int(model["segment_bits"])
    local_bits = 23 - segment_bits
    ids = (xbits & 0x7FFFFF) >> local_bits
    reports = []
    for segment in segments:
        mask = ids == segment
        sx, sy = xbits[mask], y[mask]
        row = list(model["coefficients_u32"][segment])
        candidates = []
        for delta in range(-radius, radius + 1):
            candidate = list(row)
            candidate[0] = (candidate[0] + delta) & 0xFFFFFFFF
            prediction = evaluate_segment(sx, sy, candidate, segment_bits)
            half_mismatches = int(np.sum(prediction.astype("<f2").view("<u2") != sy.astype("<f2").view("<u2")))
            candidates.append({
                "delta_u32": delta,
                "fp16_mismatches": half_mismatches,
                "fp32_squared_error": float(np.sum((prediction.astype(np.float64) - sy.astype(np.float64)) ** 2)),
            })
        candidates.sort(key=lambda item: (item["fp16_mismatches"], item["fp32_squared_error"], abs(item["delta_u32"])))
        reports.append({
            "segment": segment,
            "samples": int(np.sum(mask)),
            "baseline": next(item for item in candidates if item["delta_u32"] == 0),
            "best": candidates[0],
            "nonpositive_no_extra_half_mismatch": [
                item for item in candidates
                if item["delta_u32"] < 0 and item["fp16_mismatches"]
                <= next(base["fp16_mismatches"] for base in candidates if base["delta_u32"] == 0)
            ],
            "candidates": sorted(candidates, key=lambda item: item["delta_u32"]),
        })
    return {"schema": 1, "experiment": "n0_lg2_segment_intercept_bias_search",
            "model": str(model_path), "segments": reports}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--segment", type=int, action="append", required=True)
    parser.add_argument("--radius", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.trace, args.model, args.segment, args.radius)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"segment": item["segment"], "baseline": item["baseline"],
                       "best": item["best"], "safe_negative_count": len(item["nonpositive_no_extra_half_mismatch"])}
                      for item in report["segments"]], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
