#!/usr/bin/env python3
"""Flatten the accepted cos1 model and add safe high-resolution overrides."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

try:
    from scripts.make_n0_sqrt0_boundary_safe_model import corrected
except ModuleNotFoundError:
    from make_n0_sqrt0_boundary_safe_model import corrected


SAMPLES = 80 * 48 * 64


def u32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def make(base_model: dict, override_model: dict, trace_root: Path,
         site: str = "second", cos_trace_root: Path | None = None) -> dict:
    if site == "first":
        boundary_name = "normal0"
        cos_name = "cos"
        target_site = "first_n0_cosine_r229_r226"
    elif site == "second":
        boundary_name = "normal2"
        cos_name = "cos1"
        target_site = "second_n0_cosine_r230_r227"
    else:
        raise ValueError(f"unknown cosine site: {site}")
    rtx_boundary = np.fromfile(
        trace_root / f"rtx_{boundary_name}_boundary_trace.raw", dtype="<u4"
    ).reshape(-1, 2)
    amd_boundary = np.fromfile(
        trace_root / f"amd_{boundary_name}_boundary_trace.raw", dtype="<u4"
    ).reshape(-1, 2)
    cos_root = cos_trace_root or trace_root
    rtx_cos = np.fromfile(cos_root / f"rtx_{cos_name}_trace.raw", dtype="<u4").reshape(-1, 2)
    amd_cos = np.fromfile(cos_root / f"amd_{cos_name}_trace.raw", dtype="<u4").reshape(-1, 2)
    if any(array.shape != (SAMPLES, 2) for array in
           (rtx_boundary, amd_boundary, rtx_cos, amd_cos)):
        raise ValueError("trace shape mismatch")

    phase = amd_cos[:, 0].view("<f4")
    amd_value = amd_cos[:, 1].view("<f4")
    target_cos = rtx_cos[:, 1].view("<f4")
    accepted_cos, _ = corrected(phase, amd_value, base_model)
    sqrt_value = amd_boundary[:, 0].view("<f4")
    target_normal = (rtx_boundary[:, 1] & 0xFFFF).astype("<u2")
    baseline_normal = np.float32(sqrt_value * accepted_cos).astype("<f2").view("<u2")

    segments = int(override_model["segments"])
    scale = np.array([override_model["scale_u32"]], dtype="<u4").view("<f4")[0]
    segment = np.minimum(np.float32(phase * scale).astype(np.int32), segments - 1)
    raw_override, _ = corrected(phase, amd_value, override_model)
    coefficients = np.zeros(segments, dtype="<f4")
    order = np.argsort(segment, kind="stable")
    sorted_segment = segment[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_segment)) + 1]
    ends = np.r_[starts[1:], len(order)]
    selected = []
    for start, end in zip(starts, ends):
        indexes = order[start:end]
        current = int(sorted_segment[start])
        candidates = {
            np.float32(0.0).view("<u4").item(),
            np.float32(np.mean(target_cos[indexes].astype(np.float64) -
                               amd_value[indexes].astype(np.float64))).view("<u4").item(),
            np.float32(raw_override[indexes[0]] - amd_value[indexes[0]]).view("<u4").item(),
        }
        for index in indexes:
            candidates.add(np.float32(accepted_cos[index] - amd_value[index]).view("<u4").item())
            candidates.add(np.float32(target_cos[index] - amd_value[index]).view("<u4").item())
        best = None
        for bits in candidates:
            correction = np.array([bits], dtype="<u4").view("<f4")[0]
            candidate_cos = np.float32(amd_value[indexes] + correction)
            candidate_normal = np.float32(sqrt_value[indexes] * candidate_cos).astype("<f2").view("<u2")
            old, wanted = baseline_normal[indexes], target_normal[indexes]
            fixed = int(np.sum((old != wanted) & (candidate_normal == wanted)))
            added = int(np.sum((old == wanted) & (candidate_normal != wanted)))
            preserved = int(np.sum(candidate_cos.view("<u4") == accepted_cos[indexes].view("<u4")))
            score = (-added, fixed, preserved, -int(np.sum(candidate_normal != wanted)))
            if best is None or score > best[0]:
                best = (score, correction, fixed, added, preserved)
        assert best is not None
        coefficients[current] = best[1]
        if best[1] != 0:
            selected.append({
                "segment": current,
                "observed_fixes": best[2],
                "observed_additions": best[3],
                "accepted_cos_preserved": best[4],
                "sample_count": len(indexes),
            })

    prediction = np.float32(amd_value + coefficients[segment])
    predicted_normal = np.float32(sqrt_value * prediction).astype("<f2").view("<u2")
    return {
        "schema": 1,
        "experiment": f"nvidia_sm120_{cos_name}_hierarchical_{segments}_segment_degree0",
        "status": "PASS",
        "classification": "TRACE_DOMAIN_FLATTENED_32K_BASE_WITH_HIGH_RES_SAFE_OVERRIDES",
        "target_site": target_site,
        "segments": segments,
        "segment_bits": int(override_model["segment_bits"]),
        "degree": 0,
        "scale_u32": int(override_model["scale_u32"]),
        "skip_zero_correction": True,
        "sample_count": SAMPLES,
        "selected_segment_count": len(selected),
        "baseline_mismatch_samples": int(np.sum(baseline_normal != target_normal)),
        "predicted_mismatch_samples": int(np.sum(predicted_normal != target_normal)),
        "predicted_additions_vs_base": int(np.sum((baseline_normal == target_normal) &
                                                   (predicted_normal != target_normal))),
        "predicted_fixes_vs_base": int(np.sum((baseline_normal != target_normal) &
                                               (predicted_normal == target_normal))),
        "selected_segments": selected,
        "coefficients_u32": [[u32(float(value))] for value in coefficients],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--override-model", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--cos-trace-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", choices=("first", "second"), default="second")
    args = parser.parse_args()
    model = make(
        json.loads(args.base_model.read_text(encoding="utf-8")),
        json.loads(args.override_model.read_text(encoding="utf-8")),
        args.trace_root,
        args.site,
        args.cos_trace_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "selected_segment_count", "baseline_mismatch_samples",
        "predicted_mismatch_samples", "predicted_fixes_vs_base",
        "predicted_additions_vs_base",
    )}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
