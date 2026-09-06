#!/usr/bin/env python3
"""Choose per-segment sqrt0 ULP shifts safe after the accepted sine fix."""

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
DOMAIN_MAX = np.float32(8.0)


def make(sin_model: dict, trace_root: Path, segment_bits: int,
         max_shift: int) -> dict:
    rtx_boundary = np.fromfile(trace_root / "rtx_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_boundary = np.fromfile(trace_root / "amd_normal1_boundary_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_sin = np.fromfile(trace_root / "amd_sin0_trace.raw", dtype="<u4").reshape(-1, 2)
    sqrt_bits = amd_boundary[:, 0].copy()
    sqrt_values = sqrt_bits.view("<f4")
    if any(a.shape != (SAMPLES, 2) for a in (rtx_boundary, amd_boundary, amd_sin)):
        raise ValueError("trace shape mismatch")
    if np.any(sqrt_values <= 0) or np.any(sqrt_values >= DOMAIN_MAX):
        raise ValueError("sqrt values outside positive [0,8) domain")
    sin_values, _ = corrected(
        amd_sin[:, 0].view("<f4"), amd_sin[:, 1].view("<f4"), sin_model
    )
    target = (rtx_boundary[:, 1] & 0xFFFF).astype("<u2")
    baseline = np.float32(sqrt_values * sin_values).astype("<f2").view("<u2")
    segments = 1 << segment_bits
    scale = np.float32(segments / DOMAIN_MAX)
    segment = np.minimum(np.float32(sqrt_values * scale).astype(np.int32), segments - 1)
    shifts = np.zeros(segments, dtype="<i4")
    selected = []
    for current in np.unique(segment):
        indexes = np.flatnonzero(segment == current)
        old = baseline[indexes]
        wanted = target[indexes]
        best = None
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0:
                continue
            adjusted_bits = (sqrt_bits[indexes].astype(np.int64) + shift).astype("<u4")
            adjusted = np.float32(adjusted_bits.view("<f4") * sin_values[indexes]).astype("<f2").view("<u2")
            fixed = int(np.sum((old != wanted) & (adjusted == wanted)))
            added = int(np.sum((old == wanted) & (adjusted != wanted)))
            remaining = int(np.sum(adjusted != wanted))
            score = (added == 0 and fixed > 0, fixed, -remaining, -abs(shift))
            if best is None or score > best[0]:
                best = (score, shift, fixed, added, remaining)
        if best is not None and best[2] > 0 and best[3] == 0:
            shifts[current] = best[1]
            selected.append({"segment": int(current), "ulp_shift": int(best[1]),
                             "observed_fixes": best[2], "observed_additions": best[3]})
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_sqrt0_ulp_boundary_safe_after_sin_{segments}",
        "status": "PASS",
        "classification": "FULL_GRID_NORMAL1_BOUNDARY_CONSERVATIVE_SQRT0_ULP_CORRECTION",
        "target_site": "first_n0_sqrt_r221_r220",
        "index_value": "amd_sqrt_output_r221",
        "segments": segments,
        "segment_bits": segment_bits,
        "scale_u32": int(scale.view("<u4")),
        "max_abs_ulp_shift": max_shift,
        "sample_count": SAMPLES,
        "selection_rule": "relative to accepted sine correction: fixes > 0 and additions == 0",
        "sine_only_mismatch_samples": int(np.sum(baseline != target)),
        "selected_segment_count": len(selected),
        "selected_model_fixed_boundaries": int(sum(row["observed_fixes"] for row in selected)),
        "selected_segments": selected,
        "ulp_shifts_i32": shifts.tolist(),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-bits", type=int, default=14)
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--trace-root", type=Path,
                        default=repo / "results/20260904_999800_n0_normal1_full_grid_cross_vendor")
    args = parser.parse_args()
    model = make(json.loads(args.sin_model.read_text(encoding="utf-8")),
                 args.trace_root, args.segment_bits, args.max_shift)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "selected_segment_count", "sine_only_mismatch_samples",
        "selected_model_fixed_boundaries")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
