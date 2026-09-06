#!/usr/bin/env python3
"""Rank executed slot-3 candidate rewrites against the exact RTX output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.analyze_n0_cross_vendor import compare, raw_summary
except ModuleNotFoundError:
    from analyze_n0_cross_vendor import compare, raw_summary


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze(candidate_root: Path, reference_path: Path, baseline_path: Path) -> dict:
    aggregate = json.loads((candidate_root / "summary.json").read_text(encoding="utf-8-sig"))
    reference = reference_path.read_bytes()
    baseline = baseline_path.read_bytes()
    baseline_comparison = compare(reference, baseline)
    reports = []
    for execution in aggregate["candidates"]:
        model = execution["candidate_model"]
        candidate = (candidate_root / model / "output.raw").read_bytes()
        comparison = compare(reference, candidate)
        reports.append({
            "candidate_model": model,
            "execution_status": execution["status"],
            "device_name": execution["device_name"],
            "output_sha256": sha256(candidate),
            "equals_current_baseline": candidate == baseline,
            "comparison": comparison,
        })
    reports.sort(key=lambda item: (
        item["comparison"]["nrmse_vs_rtx_stddev"],
        -item["comparison"]["pearson_correlation"],
    ))
    best = reports[0]
    execution_gate = all(
        item["execution_status"] == "PASS"
        and "AMD Radeon RX 9070 XT" in item["device_name"]
        for item in reports
    )
    return {
        "schema": 1,
        "experiment": "slot3_candidate_rx9070xt_exact_state_rtx_ranking",
        "status": "PASS" if execution_gate else "FAIL",
        "classification": "DIAGNOSTIC_CANDIDATE_RANKING_NOT_S7",
        "counts_as_s7": False,
        "execution_gate": execution_gate,
        "reference": {"path": str(reference_path), **raw_summary(reference)},
        "current_baseline": {
            "path": str(baseline_path),
            "sha256": sha256(baseline),
            "comparison": baseline_comparison,
        },
        "best_candidate": best["candidate_model"],
        "best_candidate_improves_nrmse": (
            best["comparison"]["nrmse_vs_rtx_stddev"]
            < baseline_comparison["nrmse_vs_rtx_stddev"]
        ),
        "candidates": reports,
        "limitation": (
            "Each candidate has one real RX execution against an exact-state RTX output; "
            "only the returned internal RTX A/B/C/D trace can select the correct arithmetic "
            "model, and full-frame S7 remains open."
        ),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument(
        "--reference", type=Path,
        default=repo / "results/20260901_000000_slot3_accumulation_diagnostic/case/output_reference.raw",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=repo / "results/20260901_000000_slot3_accumulation_diagnostic/runs/baseline/output.raw",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.candidate_root / "numerical_comparison.json"
    report = analyze(args.candidate_root.resolve(), args.reference.resolve(), args.baseline.resolve())
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "best_candidate": report["best_candidate"],
        "best_candidate_improves_nrmse": report["best_candidate_improves_nrmse"],
        "baseline_nrmse": report["current_baseline"]["comparison"]["nrmse_vs_rtx_stddev"],
        "candidate_nrmse": {
            item["candidate_model"]: item["comparison"]["nrmse_vs_rtx_stddev"]
            for item in report["candidates"]
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
