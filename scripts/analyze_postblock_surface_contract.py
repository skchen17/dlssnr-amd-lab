#!/usr/bin/env python3
"""Analyze slot-154 RGBA16F surface deltas and simple layout hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np


WIDTH = 640
HEIGHT = 360
CHANNELS = 4


def read_blob(spec: str) -> bytes:
    """Read PATH or ZIP::ENTRY without extracting an archive."""
    archive, separator, entry = spec.partition("::")
    if not separator:
        return Path(spec).read_bytes()
    with zipfile.ZipFile(archive) as handle:
        return handle.read(entry)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest().upper()


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def row_segments(mask: np.ndarray) -> list[dict]:
    """Run-length encode the number of true pixels in each row."""
    counts = np.count_nonzero(mask, axis=1)
    segments = []
    start = 0
    for row in range(1, len(counts) + 1):
        if row == len(counts) or counts[row] != counts[start]:
            segments.append({
                "first_y": start,
                "last_y": row - 1,
                "pixels": int(counts[start]),
            })
            start = row
    return segments


def numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = reference.astype(np.float32).reshape(-1)
    cand = candidate.astype(np.float32).reshape(-1)
    finite = np.isfinite(ref) & np.isfinite(cand)
    ref = ref[finite]
    cand = cand[finite]
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff * diff))) if len(diff) else None
    ref_std = float(np.std(ref)) if len(ref) else 0.0
    correlation = float(np.corrcoef(ref, cand)[0, 1]) if len(ref) and np.std(cand) and ref_std else None
    return {
        "finite_pairs": int(finite.sum()),
        "nonfinite_pairs": int((~finite).sum()),
        "pearson_correlation": correlation,
        "rmse": rmse,
        "nrmse_vs_reference_stddev": rmse / ref_std if rmse is not None and ref_std else None,
    }


def exact_report(left_bits: np.ndarray, right_bits: np.ndarray) -> dict:
    exact = left_bits == right_bits
    pixel_exact = np.all(exact, axis=2)
    return {
        "exact_components": int(exact.sum()),
        "total_components": int(exact.size),
        "exact_component_fraction": float(exact.mean()),
        "exact_pixels": int(pixel_exact.sum()),
        "total_pixels": int(pixel_exact.size),
        "exact_pixel_fraction": float(pixel_exact.mean()),
        "mismatch_bbox_xyxy": bbox(~pixel_exact),
        "per_channel_exact_fraction": [float(exact[:, :, channel].mean()) for channel in range(CHANNELS)],
    }


def top_layout_hypotheses(reference: np.ndarray, candidate: np.ndarray) -> list[dict]:
    reports = []
    transforms = {
        "identity": candidate,
        "flip_x": candidate[:, ::-1, :],
        "flip_y": candidate[::-1, :, :],
        "flip_xy": candidate[::-1, ::-1, :],
    }
    for transform_name, transformed in transforms.items():
        for permutation in itertools.permutations(range(CHANNELS)):
            arranged = transformed[:, :, permutation]
            exact = float((arranged == reference).mean())
            numeric = numeric_metrics(
                reference[:, :, :3].view("<f2"), arranged[:, :, :3].view("<f2"))
            reports.append({
                "transform": transform_name,
                "channel_permutation": list(permutation),
                "exact_component_fraction": exact,
                "rgb_pearson_correlation": numeric["pearson_correlation"],
                "rgb_nrmse": numeric["nrmse_vs_reference_stddev"],
            })
    reports.sort(key=lambda item: (
        item["exact_component_fraction"],
        item["rgb_pearson_correlation"] if item["rgb_pearson_correlation"] is not None else -2.0,
    ), reverse=True)
    return reports[:8]


def analyze(initial_blob: bytes, reference_blob: bytes, candidate_blob: bytes) -> dict:
    expected = WIDTH * HEIGHT * CHANNELS * 2
    for name, blob in (("initial", initial_blob), ("reference", reference_blob), ("candidate", candidate_blob)):
        if len(blob) != expected:
            raise ValueError(f"{name} size is {len(blob)}, expected {expected}")
    initial_bits = np.frombuffer(initial_blob, dtype="<u2").reshape(HEIGHT, WIDTH, CHANNELS)
    reference_bits = np.frombuffer(reference_blob, dtype="<u2").reshape(HEIGHT, WIDTH, CHANNELS)
    candidate_bits = np.frombuffer(candidate_blob, dtype="<u2").reshape(HEIGHT, WIDTH, CHANNELS)
    initial = initial_bits.view("<f2")
    reference = reference_bits.view("<f2")
    candidate = candidate_bits.view("<f2")

    reference_changed = reference_bits != initial_bits
    candidate_changed = candidate_bits != initial_bits
    reference_pixel_changed = np.any(reference_changed, axis=2)
    candidate_pixel_changed = np.any(candidate_changed, axis=2)
    overlap = reference_changed & candidate_changed
    union = reference_changed | candidate_changed
    ref_changed_count = int(reference_changed.sum())
    ref_unchanged = ~reference_changed

    return {
        "schema": 1,
        "experiment": "postblock_surface_contract_delta",
        "dimensions": [WIDTH, HEIGHT, CHANNELS],
        "sha256": {
            "initial": sha256(initial_blob),
            "reference": sha256(reference_blob),
            "candidate": sha256(candidate_blob),
        },
        "reference_vs_initial": exact_report(reference_bits, initial_bits) | {
            "changed_components": ref_changed_count,
            "changed_pixels": int(reference_pixel_changed.sum()),
            "changed_pixel_bbox_xyxy": bbox(reference_pixel_changed),
            "changed_pixel_row_segments": row_segments(reference_pixel_changed),
        },
        "candidate_vs_initial": exact_report(candidate_bits, initial_bits) | {
            "changed_components": int(candidate_changed.sum()),
            "changed_pixels": int(candidate_pixel_changed.sum()),
            "changed_pixel_bbox_xyxy": bbox(candidate_pixel_changed),
            "changed_pixel_row_segments": row_segments(candidate_pixel_changed),
        },
        "candidate_vs_reference": exact_report(candidate_bits, reference_bits) | {
            "rgba_numeric": numeric_metrics(reference, candidate),
            "rgb_numeric": numeric_metrics(reference[:, :, :3], candidate[:, :, :3]),
            "exact_fraction_where_reference_changed": (
                float((candidate_bits[reference_changed] == reference_bits[reference_changed]).mean())
                if ref_changed_count else None
            ),
            "exact_fraction_where_reference_unchanged": float(
                (candidate_bits[ref_unchanged] == reference_bits[ref_unchanged]).mean()
            ),
            "exact_pixel_row_segments": row_segments(
                np.all(candidate_bits == reference_bits, axis=2)),
        },
        "write_mask": {
            "component_intersection": int(overlap.sum()),
            "component_union": int(union.sum()),
            "component_iou": float(overlap.sum() / union.sum()) if union.any() else 1.0,
            "reference_changed_candidate_unchanged": int((reference_changed & ~candidate_changed).sum()),
            "candidate_changed_reference_unchanged": int((candidate_changed & ~reference_changed).sum()),
        },
        "top_layout_hypotheses": top_layout_hypotheses(reference_bits, candidate_bits),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True, help="PATH or ZIP::ENTRY")
    parser.add_argument("--reference", required=True, help="PATH or ZIP::ENTRY")
    parser.add_argument("--candidate", required=True, help="PATH or ZIP::ENTRY")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(read_blob(args.initial), read_blob(args.reference), read_blob(args.candidate))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
