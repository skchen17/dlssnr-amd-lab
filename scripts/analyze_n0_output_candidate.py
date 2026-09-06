#!/usr/bin/env python3
"""Compare one or two RX N0 output candidates with a captured RTX output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.analyze_n0_cross_vendor import compare, raw_summary
except ModuleNotFoundError:
    from analyze_n0_cross_vendor import compare, raw_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rtx = args.rtx.read_bytes()
    candidate = args.candidate.read_bytes()
    candidate_metrics = compare(rtx, candidate)
    report = {
        "schema": 1,
        "experiment": "n0_output_candidate_vs_rtx5070",
        "status": "PASS",
        "rtx": raw_summary(rtx),
        "candidate": raw_summary(candidate),
        "comparison": candidate_metrics,
        "mismatching_bytes": len(rtx) - candidate_metrics["exact"],
    }
    if args.baseline:
        baseline = args.baseline.read_bytes()
        baseline_metrics = compare(rtx, baseline)
        fixed = added = 0
        for reference, before, after in zip(rtx, baseline, candidate):
            fixed += before != reference and after == reference
            added += before == reference and after != reference
        report["baseline"] = raw_summary(baseline)
        report["baseline_comparison"] = baseline_metrics
        report["delta"] = {
            "baseline_mismatching_bytes": len(rtx) - baseline_metrics["exact"],
            "candidate_mismatching_bytes": len(rtx) - candidate_metrics["exact"],
            "net_mismatching_bytes": baseline_metrics["exact"] - candidate_metrics["exact"],
            "fixed_old_mismatches": fixed,
            "added_new_mismatches": added,
            "nrmse_delta": candidate_metrics["nrmse_vs_rtx_stddev"]
            - baseline_metrics["nrmse_vs_rtx_stddev"],
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mismatching_bytes": report["mismatching_bytes"],
                      "nrmse": candidate_metrics["nrmse_vs_rtx_stddev"],
                      "delta": report.get("delta")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
