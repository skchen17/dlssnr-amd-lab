#!/usr/bin/env python3
"""Find phase-segmented joint cos0/sqrt0 shifts safe for normal0 and normal1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.make_n0_sqrt0_boundary_safe_model import corrected
except ModuleNotFoundError:
    from make_n0_sqrt0_boundary_safe_model import corrected


SAMPLES = 80 * 48 * 64


def f32_from_u32(value: int) -> np.float32:
    return np.array([value], dtype="<u4").view("<f4")[0]


def make(cos_model: dict, sin_model: dict, sqrt_model: dict,
         cos_trace_root: Path, normal0_root: Path, normal1_root: Path,
         max_shift: int) -> dict:
    cos_trace = np.fromfile(cos_trace_root / "amd_cos_trace.raw", dtype="<u4").reshape(-1, 2)
    sin_trace = np.fromfile(normal1_root / "amd_sin0_trace.raw", dtype="<u4").reshape(-1, 2)
    amd0 = np.fromfile(normal0_root / "amd_normal0_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    rtx0 = np.fromfile(normal0_root / "rtx_normal0_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd1 = np.fromfile(normal1_root / "amd_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    rtx1 = np.fromfile(normal1_root / "rtx_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    if any(array.shape != (SAMPLES, 2) for array in
           (cos_trace, sin_trace, amd0, rtx0, amd1, rtx1)):
        raise ValueError("trace shape mismatch")

    phase = cos_trace[:, 0].view("<f4")
    cos_bits = cos_trace[:, 1].copy()
    cos_value, segment = corrected(phase, cos_bits.view("<f4"), cos_model)
    sin_value, _ = corrected(
        sin_trace[:, 0].view("<f4"), sin_trace[:, 1].view("<f4"), sin_model
    )
    sqrt_bits = amd0[:, 0].copy()
    if not np.array_equal(sqrt_bits, amd1[:, 0]):
        raise ValueError("normal0 and normal1 do not share the same sqrt0 trace")
    sqrt_value = sqrt_bits.view("<f4")
    sqrt_scale = f32_from_u32(int(sqrt_model["scale_u32"]))
    sqrt_segment = np.minimum(
        np.float32(sqrt_value * sqrt_scale).astype(np.int32),
        int(sqrt_model["segments"]) - 1,
    )
    prior_sqrt_shift = np.asarray(sqrt_model["ulp_shifts_i32"], dtype="<i4")[sqrt_segment]
    adjusted_sqrt_bits = (sqrt_bits.astype(np.int64) + prior_sqrt_shift).astype("<u4")
    adjusted_sqrt = adjusted_sqrt_bits.view("<f4")

    target0 = (rtx0[:, 1] & 0xFFFF).astype("<u2")
    target1 = (rtx1[:, 1] & 0xFFFF).astype("<u2")
    baseline0 = np.float32(adjusted_sqrt * cos_value).astype("<f2").view("<u2")
    baseline1 = np.float32(adjusted_sqrt * sin_value).astype("<f2").view("<u2")
    segments = int(cos_model["segments"])
    cos_shifts = np.zeros(segments, dtype="<i4")
    sqrt_shifts = np.zeros(segments, dtype="<i4")
    selected = []
    order = np.argsort(segment, kind="stable")
    sorted_segment = segment[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_segment)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        indexes = order[start:end]
        current = int(sorted_segment[start])
        old0, old1 = baseline0[indexes], baseline1[indexes]
        wanted0, wanted1 = target0[indexes], target1[indexes]
        if not np.any((old0 != wanted0) | (old1 != wanted1)):
            continue
        best = None
        for cos_shift in range(-max_shift, max_shift + 1):
            shifted_cos_bits = (cos_bits[indexes].astype(np.int64) + cos_shift).astype("<u4")
            shifted_cos, _ = corrected(
                phase[indexes], shifted_cos_bits.view("<f4"), cos_model
            )
            for sqrt_shift in range(-max_shift, max_shift + 1):
                if cos_shift == 0 and sqrt_shift == 0:
                    continue
                shifted_sqrt = (
                    adjusted_sqrt_bits[indexes].astype(np.int64) + sqrt_shift
                ).astype("<u4").view("<f4")
                candidate0 = np.float32(shifted_sqrt * shifted_cos).astype("<f2").view("<u2")
                candidate1 = np.float32(shifted_sqrt * sin_value[indexes]).astype("<f2").view("<u2")
                fixed = int(np.sum((old0 != wanted0) & (candidate0 == wanted0)) +
                            np.sum((old1 != wanted1) & (candidate1 == wanted1)))
                added = int(np.sum((old0 == wanted0) & (candidate0 != wanted0)) +
                            np.sum((old1 == wanted1) & (candidate1 != wanted1)))
                remaining = int(np.sum(candidate0 != wanted0) + np.sum(candidate1 != wanted1))
                score = (added == 0 and fixed > 0, fixed, -remaining,
                         -(abs(cos_shift) + abs(sqrt_shift)))
                if best is None or score > best[0]:
                    best = (score, cos_shift, sqrt_shift, fixed, added)
        if best is not None and best[3] > 0 and best[4] == 0:
            cos_shifts[current] = best[1]
            sqrt_shifts[current] = best[2]
            selected.append({
                "segment": current,
                "cos0_raw_ulp_shift": int(best[1]),
                "sqrt0_post_ulp_shift": int(best[2]),
                "observed_fixes": best[3],
                "observed_additions": best[4],
                "sample_count": len(indexes),
            })
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_cos0_sqrt0_joint_ulp_{segments}",
        "status": "PASS",
        "classification": "FULL_GRID_NORMAL0_NORMAL1_CONSERVATIVE_JOINT_ULP_MODEL",
        "segments": segments,
        "segment_bits": int(cos_model["segment_bits"]),
        "scale_u32": int(cos_model["scale_u32"]),
        "max_abs_ulp_shift": max_shift,
        "sample_count": SAMPLES,
        "baseline_normal0_mismatches": int(np.sum(baseline0 != target0)),
        "baseline_normal1_mismatches": int(np.sum(baseline1 != target1)),
        "selected_segment_count": len(selected),
        "selected_model_fixed_boundaries": int(sum(row["observed_fixes"] for row in selected)),
        "selected_segments": selected,
        "cos0_raw_ulp_shifts_i32": cos_shifts.tolist(),
        "sqrt0_post_ulp_shifts_i32": sqrt_shifts.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cos-model", type=Path, required=True)
    parser.add_argument("--sin-model", type=Path, required=True)
    parser.add_argument("--sqrt-model", type=Path, required=True)
    parser.add_argument("--cos-trace-root", type=Path, required=True)
    parser.add_argument("--normal0-root", type=Path, required=True)
    parser.add_argument("--normal1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shift", type=int, default=4)
    args = parser.parse_args()
    model = make(
        json.loads(args.cos_model.read_text(encoding="utf-8")),
        json.loads(args.sin_model.read_text(encoding="utf-8")),
        json.loads(args.sqrt_model.read_text(encoding="utf-8")),
        args.cos_trace_root, args.normal0_root, args.normal1_root, args.max_shift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "baseline_normal0_mismatches",
        "baseline_normal1_mismatches", "selected_segment_count",
        "selected_model_fixed_boundaries",
    )}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
