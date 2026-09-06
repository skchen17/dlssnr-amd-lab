#!/usr/bin/env python3
"""Compare a full-graph numerical variant with its RX baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(baseline: dict, candidate: dict) -> dict:
    slots = sorted(set(baseline["boundaries"]) & set(candidate["boundaries"]), key=int)
    boundaries = []
    for slot in slots:
        before = baseline["boundaries"][slot]
        after = candidate["boundaries"][slot]
        before_nrmse = before["comparison"].get("nrmse_vs_rtx_stddev")
        after_nrmse = after["comparison"].get("nrmse_vs_rtx_stddev")
        boundaries.append({
            "slot": int(slot),
            "baseline_sha256": before["amd_sha256"],
            "candidate_sha256": after["amd_sha256"],
            "bitwise_equal": before["amd_sha256"] == after["amd_sha256"],
            "baseline_nrmse": before_nrmse,
            "candidate_nrmse": after_nrmse,
            "nrmse_delta": None if before_nrmse is None else after_nrmse - before_nrmse,
        })
    before_final = baseline["final_comparison"]["nrmse_vs_rtx_stddev"]
    after_final = candidate["final_comparison"]["nrmse_vs_rtx_stddev"]
    improved_slots = [row["slot"] for row in boundaries
                      if row["nrmse_delta"] is not None and row["nrmse_delta"] < 0]
    worsened_slots = [row["slot"] for row in boundaries
                      if row["nrmse_delta"] is not None and row["nrmse_delta"] > 0]
    final_delta = after_final - before_final
    direction = "improves" if final_delta < 0 else ("worsens" if final_delta > 0 else "does not change")
    verdict = (
        f"The variant improves NRMSE at boundaries {improved_slots}"
        f"{f' and worsens boundaries {worsened_slots}' if worsened_slots else ' with no worsened measured boundary'}; "
        f"it {direction} final NRMSE and "
        f"{'satisfies' if candidate['counts_as_s7'] else 'does not satisfy'} the full-frame gate."
    )
    return {
        "schema": 1,
        "experiment": "full_graph_variant_delta_rx9070xt",
        "status": "PASS",
        "classification": "DETERMINISTIC_FULL_GRAPH_VARIANT_COMPARISON_NOT_S7",
        "counts_as_s7": False,
        "baseline_execution_gate": baseline["execution_gate"],
        "candidate_execution_gate": candidate["execution_gate"],
        "baseline_final_sha256": baseline["final_amd_sha256"],
        "candidate_final_sha256": candidate["final_amd_sha256"],
        "baseline_final_nrmse": before_final,
        "candidate_final_nrmse": after_final,
        "final_nrmse_delta": final_delta,
        "final_relative_nrmse_improvement": (before_final - after_final) / before_final,
        "candidate_counts_as_s7": candidate["counts_as_s7"],
        "improved_boundary_slots": improved_slots,
        "worsened_boundary_slots": worsened_slots,
        "boundaries": boundaries,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8-sig"))
    report = analyze(baseline, candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
