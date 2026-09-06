#!/usr/bin/env python3
"""Choose safe ULP shifts before N0's accepted sine polynomial correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.fit_nvidia_cos_correction import evaluate
except ModuleNotFoundError:
    from fit_nvidia_cos_correction import evaluate


SAMPLES = 80 * 48 * 64


def f32(value: int) -> np.float32:
    return np.array([value], dtype="<u4").view("<f4")[0]


def make(sin_model: dict, sqrt_model: dict, trace_root: Path,
         max_shift: int) -> dict:
    rtx_boundary = np.fromfile(trace_root / "rtx_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_boundary = np.fromfile(trace_root / "amd_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_sin = np.fromfile(trace_root / "amd_sin0_trace.raw", dtype="<u4").reshape(-1, 2)
    phase = amd_sin[:, 0].view("<f4")
    sin_bits = amd_sin[:, 1].copy()
    sin_base = sin_bits.view("<f4")
    segments = int(sin_model["segments"])
    degree = int(sin_model["degree"])
    coeff = np.asarray(sin_model["coefficients_u32"], dtype="<u4").view("<f4")
    position = np.float32(phase * f32(int(sin_model["scale_u32"])))
    segment = np.minimum(position.astype(np.int32), segments - 1)
    local = np.float32(position - segment.astype("<f4"))

    sqrt_bits = amd_boundary[:, 0].copy()
    sqrt_value = sqrt_bits.view("<f4")
    sqrt_position = np.float32(sqrt_value * f32(int(sqrt_model["scale_u32"])))
    sqrt_segment = np.minimum(sqrt_position.astype(np.int32), int(sqrt_model["segments"]) - 1)
    sqrt_shift = np.asarray(sqrt_model["ulp_shifts_i32"], dtype="<i4")[sqrt_segment]
    adjusted_sqrt = (sqrt_bits.astype(np.int64) + sqrt_shift).astype("<u4").view("<f4")

    corrected_sin = evaluate(sin_base, coeff, segment, local)
    target = (rtx_boundary[:, 1] & 0xFFFF).astype("<u2")
    baseline = np.float32(adjusted_sqrt * corrected_sin).astype("<f2").view("<u2")
    shifts = np.zeros(segments, dtype="<i4")
    selected = []
    order = np.argsort(segment, kind="stable")
    sorted_segment = segment[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_segment)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        indexes = order[start:end]
        current = int(sorted_segment[start])
        if not np.any(baseline[indexes] != target[indexes]):
            continue
        old, wanted = baseline[indexes], target[indexes]
        best = None
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0:
                continue
            shifted_base = (sin_bits[indexes].astype(np.int64) + shift).astype("<u4").view("<f4")
            candidate_sin = evaluate(shifted_base, coeff, segment[indexes], local[indexes])
            candidate = np.float32(adjusted_sqrt[indexes] * candidate_sin).astype("<f2").view("<u2")
            fixed = int(np.sum((old != wanted) & (candidate == wanted)))
            added = int(np.sum((old == wanted) & (candidate != wanted)))
            score = (added == 0 and fixed > 0, fixed, -int(np.sum(candidate != wanted)), -abs(shift))
            if best is None or score > best[0]:
                best = (score, shift, fixed, added)
        if best is not None and best[2] > 0 and best[3] == 0:
            shifts[current] = best[1]
            selected.append({"segment": int(current), "ulp_shift": int(best[1]),
                             "observed_fixes": best[2], "observed_additions": best[3]})
    return {"schema": 1,
            "experiment": f"nvidia_sm120_sin0_pre_polynomial_ulp_safe_{segments}",
            "status": "PASS",
            "classification": "FULL_GRID_NORMAL1_CONSERVATIVE_SIN0_PRE_CORRECTION_ULP_MODEL",
            "target_site": "first_n0_sine_r228_r226",
            "segments": segments, "segment_bits": int(sin_model["segment_bits"]),
            "scale_u32": int(sin_model["scale_u32"]), "max_abs_ulp_shift": max_shift,
            "sample_count": SAMPLES, "combined_baseline_mismatch_samples": int(np.sum(baseline != target)),
            "selected_segment_count": len(selected),
            "selected_model_fixed_boundaries": int(sum(row["observed_fixes"] for row in selected)),
            "selected_segments": selected, "ulp_shifts_i32": shifts.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-model", type=Path, required=True)
    parser.add_argument("--sqrt-model", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shift", type=int, default=64)
    args = parser.parse_args()
    model = make(json.loads(args.sin_model.read_text(encoding="utf-8")),
                 json.loads(args.sqrt_model.read_text(encoding="utf-8")),
                 args.trace_root, args.max_shift)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in ("status", "segments",
        "combined_baseline_mismatch_samples", "selected_segment_count",
        "selected_model_fixed_boundaries")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
