#!/usr/bin/env python3
"""Select cosine-correction segments that fix and do not add FP16 boundaries."""

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


def f32_from_u32(value: int) -> np.float32:
    return np.array([value], dtype="<u4").view("<f4")[0]


def make(base_model: dict, boundary_root: Path, cos_root: Path) -> dict:
    rtx_boundary = np.fromfile(boundary_root / "rtx_normal0_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_boundary = np.fromfile(boundary_root / "amd_normal0_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    rtx_cos = np.fromfile(cos_root / "rtx_cos_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_cos = np.fromfile(cos_root / "amd_cos_trace.raw", dtype="<u4").reshape(-1, 2)
    if any(array.shape != (SAMPLES, 2) for array in (rtx_boundary, amd_boundary, rtx_cos, amd_cos)):
        raise ValueError("trace shape mismatch")
    if not np.array_equal(rtx_cos[:, 0], amd_cos[:, 0]):
        raise ValueError("cosine inputs differ")

    segments = int(base_model["segments"])
    degree = int(base_model["degree"])
    coefficient_bits = np.asarray(base_model["coefficients_u32"], dtype="<u4")
    if coefficient_bits.shape != (segments, degree + 1):
        raise ValueError("coefficient shape mismatch")
    coefficients = coefficient_bits.view("<f4")
    scale = f32_from_u32(int(base_model["scale_u32"]))
    position = np.float32(rtx_cos[:, 0].view("<f4") * scale)
    segment = np.minimum(position.astype(np.int32), segments - 1)
    local = np.float32(position - segment.astype("<f4"))
    corrected_cos = evaluate(amd_cos[:, 1].view("<f4"), coefficients, segment, local)
    corrected_normal = np.float32(
        amd_boundary[:, 0].view("<f4") * corrected_cos
    ).astype("<f2").view("<u2")
    target = (rtx_boundary[:, 1] & 0xFFFF).astype("<u2")
    baseline = (amd_boundary[:, 1] & 0xFFFF).astype("<u2")
    fixed = (baseline != target) & (corrected_normal == target)
    added = (baseline == target) & (corrected_normal != target)
    fixed_by_segment = np.bincount(segment, fixed.astype(np.int64), minlength=segments)
    added_by_segment = np.bincount(segment, added.astype(np.int64), minlength=segments)
    selected = (fixed_by_segment > 0) & (added_by_segment == 0)

    sparse_bits = coefficient_bits.copy()
    sparse_bits[~selected] = 0
    selected_rows = [
        {
            "segment": int(index),
            "observed_fixes": int(fixed_by_segment[index]),
            "observed_additions": int(added_by_segment[index]),
        }
        for index in np.flatnonzero(selected)
    ]
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_cos_boundary_safe_{segments}_segment_degree{degree}",
        "status": "PASS",
        "classification": "FULL_GRID_FP16_BOUNDARY_CONSERVATIVE_COS_CORRECTION_CANDIDATE",
        "segments": segments,
        "segment_bits": int(base_model["segment_bits"]),
        "degree": degree,
        "scale_u32": int(base_model["scale_u32"]),
        "sample_count": SAMPLES,
        "selection_rule": "observed_fixes > 0 and observed_additions == 0",
        "skip_zero_correction": True,
        "selected_segment_count": int(np.sum(selected)),
        "base_model_fixed_boundaries": int(np.sum(fixed)),
        "base_model_added_boundaries": int(np.sum(added)),
        "selected_segments": selected_rows,
        "coefficients_u32": sparse_bits.tolist(),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boundary-root", type=Path,
        default=repo / "results/20260904_110000_n0_normal0_boundary_full_grid_cross_vendor",
    )
    parser.add_argument(
        "--cos-root", type=Path,
        default=repo / "results/20260904_020000_n0_cos_full_grid_cross_vendor",
    )
    args = parser.parse_args()
    model = make(
        json.loads(args.base_model.read_text(encoding="utf-8")),
        args.boundary_root,
        args.cos_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "degree", "selected_segment_count",
        "base_model_fixed_boundaries", "base_model_added_boundaries",
    )}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
