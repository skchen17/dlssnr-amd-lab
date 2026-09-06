#!/usr/bin/env python3
"""Build a sparse high-resolution refinement over the accepted sine model."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

try:
    from scripts.fit_nvidia_cos_correction import evaluate
    from scripts.make_n0_sqrt0_boundary_safe_model import corrected
except ModuleNotFoundError:
    from fit_nvidia_cos_correction import evaluate
    from make_n0_sqrt0_boundary_safe_model import corrected


SAMPLES = 80 * 48 * 64
TWO_PI = np.float32(2.0 * np.pi)


def u32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def make(sin_model: dict, sqrt_model: dict, trace_root: Path,
         segment_bits: int) -> dict:
    rtx_boundary = np.fromfile(trace_root / "rtx_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_boundary = np.fromfile(trace_root / "amd_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    rtx_sin = np.fromfile(trace_root / "rtx_sin0_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_sin = np.fromfile(trace_root / "amd_sin0_trace.raw", dtype="<u4").reshape(-1, 2)
    phase = amd_sin[:, 0].view("<f4")
    accepted_sin, _ = corrected(phase, amd_sin[:, 1].view("<f4"), sin_model)

    sqrt_bits = amd_boundary[:, 0].copy()
    sqrt_value = sqrt_bits.view("<f4")
    sqrt_scale = np.array([sqrt_model["scale_u32"]], dtype="<u4").view("<f4")[0]
    sqrt_segment = np.minimum(np.float32(sqrt_value * sqrt_scale).astype(np.int32),
                              int(sqrt_model["segments"]) - 1)
    sqrt_shift = np.asarray(sqrt_model["ulp_shifts_i32"], dtype="<i4")[sqrt_segment]
    accepted_sqrt = (sqrt_bits.astype(np.int64) + sqrt_shift).astype("<u4").view("<f4")

    segments = 1 << segment_bits
    scale = np.float32(segments / TWO_PI)
    position = np.float32(phase * scale)
    segment = np.minimum(position.astype(np.int32), segments - 1)
    target_sin = rtx_sin[:, 1].view("<f4")
    residual = target_sin.astype(np.float64) - accepted_sin.astype(np.float64)
    coefficients = np.zeros(segments, dtype="<f4")
    order = np.argsort(segment, kind="stable")
    sorted_segment = segment[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_segment)) + 1]
    ends = np.r_[starts[1:], len(order)]
    target_normal = (rtx_boundary[:, 1] & 0xFFFF).astype("<u2")
    baseline_normal = np.float32(accepted_sqrt * accepted_sin).astype("<f2").view("<u2")
    selected_rows = []
    for start, end in zip(starts, ends):
        indexes = order[start:end]
        current = int(sorted_segment[start])
        correction = np.float32(np.mean(residual[indexes]))
        if correction == 0:
            continue
        candidate_sin = np.float32(accepted_sin[indexes] + correction)
        candidate_normal = np.float32(accepted_sqrt[indexes] * candidate_sin).astype("<f2").view("<u2")
        old, wanted = baseline_normal[indexes], target_normal[indexes]
        fixed = int(np.sum((old != wanted) & (candidate_normal == wanted)))
        added = int(np.sum((old == wanted) & (candidate_normal != wanted)))
        if fixed > 0 and added == 0:
            coefficients[current] = correction
            selected_rows.append({"segment": current, "observed_fixes": fixed,
                                  "observed_additions": added})
    return {"schema": 1,
            "experiment": f"nvidia_sm120_sin0_refinement_{segments}_segment_degree0",
            "status": "PASS",
            "classification": "FULL_GRID_NORMAL1_BOUNDARY_SPARSE_SIN0_REFINEMENT",
            "target_site": "after_accepted_sine_correction",
            "segments": segments, "segment_bits": segment_bits, "degree": 0,
            "scale_u32": u32(float(scale)), "skip_zero_correction": True,
            "sample_count": SAMPLES,
            "baseline_mismatch_samples": int(np.sum(baseline_normal != target_normal)),
            "selected_segment_count": len(selected_rows),
            "selected_model_fixed_boundaries": int(sum(row["observed_fixes"] for row in selected_rows)),
            "selected_segments": selected_rows,
            "coefficients_u32": [[u32(float(value))] for value in coefficients]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-model", type=Path, required=True)
    parser.add_argument("--sqrt-model", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-bits", type=int, default=19)
    args = parser.parse_args()
    model = make(json.loads(args.sin_model.read_text(encoding="utf-8")),
                 json.loads(args.sqrt_model.read_text(encoding="utf-8")),
                 args.trace_root, args.segment_bits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in ("status", "segments",
        "baseline_mismatch_samples", "selected_segment_count",
        "selected_model_fixed_boundaries")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
