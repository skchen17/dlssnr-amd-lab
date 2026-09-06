#!/usr/bin/env python3
"""Compare slot-154 replay outputs against a frame-aligned RGBA16F oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.analyze_full_graph_integrated import compare_rgba16f_rgb, sha256
except ModuleNotFoundError:
    from analyze_full_graph_integrated import compare_rgba16f_rgb, sha256


def analyze(reference_path: Path, candidates: list[tuple[str, Path]]) -> dict:
    reference = reference_path.read_bytes()
    reports = {}
    for name, path in candidates:
        candidate = path.read_bytes()
        reports[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(candidate),
            "bitwise_exact": candidate == reference,
            "comparison": compare_rgba16f_rgb(reference, candidate),
        }
    return {
        "schema": 1,
        "experiment": "frame_aligned_slot154_output_reanalysis",
        "reference": str(reference_path.resolve()),
        "reference_sha256": sha256(reference),
        "candidates": reports,
        "all_pass": all(item["comparison"]["parity_gate"]["pass"]
                        for item in reports.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[],
                        help="NAME=PATH; may be repeated")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = []
    for spec in args.candidate:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            parser.error("--candidate must be NAME=PATH")
        candidates.append((name, Path(path)))
    if not candidates:
        parser.error("at least one --candidate is required")
    report = analyze(args.reference, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
