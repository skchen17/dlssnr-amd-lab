#!/usr/bin/env python3
"""Select sqrt0 corrections safe after the accepted sine correction."""

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


def corrected(index_values: np.ndarray, base_values: np.ndarray,
              model: dict) -> tuple[np.ndarray, np.ndarray]:
    segments = int(model["segments"])
    degree = int(model["degree"])
    bits = np.asarray(model["coefficients_u32"], dtype="<u4")
    if bits.shape != (segments, degree + 1):
        raise ValueError("coefficient shape mismatch")
    position = np.float32(index_values * f32_from_u32(int(model["scale_u32"])))
    segment = np.minimum(position.astype(np.int32), segments - 1)
    local = np.float32(position - segment.astype("<f4"))
    return evaluate(base_values, bits.view("<f4"), segment, local), segment


def make(sqrt_model: dict, sin_model: dict, trace_root: Path) -> dict:
    rtx_boundary = np.fromfile(trace_root / "rtx_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_boundary = np.fromfile(trace_root / "amd_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_sin = np.fromfile(trace_root / "amd_sin0_trace.raw", dtype="<u4").reshape(-1, 2)
    if any(a.shape != (SAMPLES, 2) for a in (rtx_boundary, amd_boundary, amd_sin)):
        raise ValueError("trace shape mismatch")
    sqrt_base = amd_boundary[:, 0].view("<f4")
    sin_base = amd_sin[:, 1].view("<f4")
    corrected_sqrt, sqrt_segment = corrected(sqrt_base, sqrt_base, sqrt_model)
    corrected_sin, _ = corrected(
        amd_sin[:, 0].view("<f4"), sin_base, sin_model
    )
    target = (rtx_boundary[:, 1] & 0xFFFF).astype("<u2")
    sine_only = np.float32(sqrt_base * corrected_sin).astype("<f2").view("<u2")
    combined = np.float32(corrected_sqrt * corrected_sin).astype("<f2").view("<u2")
    fixed = (sine_only != target) & (combined == target)
    added = (sine_only == target) & (combined != target)
    segments = int(sqrt_model["segments"])
    fixed_by_segment = np.bincount(sqrt_segment, fixed.astype(np.int64), minlength=segments)
    added_by_segment = np.bincount(sqrt_segment, added.astype(np.int64), minlength=segments)
    selected = (fixed_by_segment > 0) & (added_by_segment == 0)
    sparse = np.asarray(sqrt_model["coefficients_u32"], dtype="<u4").copy()
    sparse[~selected] = 0
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_sqrt0_boundary_safe_after_sin_{segments}",
        "status": "PASS",
        "classification": "FULL_GRID_NORMAL1_BOUNDARY_CONSERVATIVE_SQRT0_CORRECTION_CANDIDATE",
        "target_site": "first_n0_sqrt_r221_r220",
        "index_value": "amd_sqrt_output_r221",
        "segments": segments,
        "segment_bits": int(sqrt_model["segment_bits"]),
        "degree": int(sqrt_model["degree"]),
        "scale_u32": int(sqrt_model["scale_u32"]),
        "sample_count": SAMPLES,
        "selection_rule": "relative to accepted sine correction: observed fixes > 0 and additions == 0",
        "skip_zero_correction": True,
        "selected_segment_count": int(np.sum(selected)),
        "sine_only_mismatch_samples": int(np.sum(sine_only != target)),
        "base_model_fixed_boundaries": int(np.sum(fixed)),
        "base_model_added_boundaries": int(np.sum(added)),
        "selected_model_fixed_boundaries": int(np.sum(fixed_by_segment[selected])),
        "selected_segments": [{
            "segment": int(index),
            "observed_fixes": int(fixed_by_segment[index]),
            "observed_additions": int(added_by_segment[index]),
        } for index in np.flatnonzero(selected)],
        "coefficients_u32": sparse.tolist(),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqrt-model", type=Path, required=True)
    parser.add_argument("--sin-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path,
                        default=repo / "results/20260904_999800_n0_normal1_full_grid_cross_vendor")
    args = parser.parse_args()
    model = make(json.loads(args.sqrt_model.read_text(encoding="utf-8")),
                 json.loads(args.sin_model.read_text(encoding="utf-8")), args.trace_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "degree", "selected_segment_count",
        "sine_only_mismatch_samples", "base_model_fixed_boundaries",
        "base_model_added_boundaries", "selected_model_fixed_boundaries"
    )}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
