#!/usr/bin/env python3
"""Rank slot-3/4 reduction candidates by same-capture full-graph propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MEASURED_SLOTS = (3, 4, 5, 6, 7, 8, 9, 15, 154)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze(baseline_path: Path, candidate_paths: list[Path]) -> dict:
    baseline = _read(baseline_path)
    if baseline.get("execution_gate") is not True:
        raise ValueError("baseline execution gate failed")
    if baseline.get("evidence", {}).get("rtx_intermediate_state_injection") is not False:
        raise ValueError("baseline contains RTX state injection")
    base_metrics = {
        slot: float(baseline["boundaries"][str(slot)]["comparison"]["nrmse_vs_rtx_stddev"])
        for slot in MEASURED_SLOTS
    }

    rows = []
    seen = set()
    for analysis_path in candidate_paths:
        report = _read(analysis_path)
        execution = _read(analysis_path.parent / "execution.json")
        plan = _read(Path(execution["plan"]))
        targets = plan.get("variant", {}).get("targets", [])
        if not targets:
            raise ValueError(f"candidate {analysis_path} has no reduction targets")
        target = "+".join(targets)
        if target in seen:
            raise ValueError(f"duplicate target {target}")
        seen.add(target)
        if report.get("execution_gate") is not True:
            raise ValueError(f"candidate {target} execution gate failed")
        if report.get("amd_repeat_determinism_gate") is not True:
            raise ValueError(f"candidate {target} is not repeat deterministic")
        if report.get("evidence", {}).get("rtx_intermediate_state_injection") is not False:
            raise ValueError(f"candidate {target} contains RTX state injection")
        metrics = {}
        for slot in MEASURED_SLOTS:
            base_boundary = baseline["boundaries"][str(slot)]
            candidate = report["boundaries"][str(slot)]
            if candidate["rtx_sha256"] != base_boundary["rtx_sha256"]:
                raise ValueError(f"candidate {target} slot {slot} reference differs")
            nrmse = float(candidate["comparison"]["nrmse_vs_rtx_stddev"])
            metrics[str(slot)] = {
                "nrmse": nrmse,
                "delta_vs_baseline": nrmse - base_metrics[slot],
                "pass": candidate["comparison"]["parity_gate"]["pass"],
            }
        final_nrmse = float(report["final_comparison"]["nrmse_vs_rtx_stddev"])
        rows.append({
            "target": target,
            "targets": targets,
            "analysis": str(analysis_path),
            "metrics": metrics,
            "final_nrmse": final_nrmse,
            "final_delta_vs_baseline": final_nrmse
                - float(baseline["final_comparison"]["nrmse_vs_rtx_stddev"]),
            "final_sha256": report["final_amd_sha256"],
        })

    rows.sort(key=lambda row: (
        row["metrics"]["6"]["nrmse"],
        row["metrics"]["7"]["nrmse"],
        row["metrics"]["8"]["nrmse"],
    ))
    best = rows[0] if rows else None
    return {
        "schema": 1,
        "experiment": "slot34_reduction_same_capture_propagation_search",
        "status": "PASS",
        "classification": "REAL_AMD_FULL_GRAPH_NUMERICAL_SEARCH_NOT_S7",
        "counts_as_s7": False,
        "candidate_count": len(rows),
        "primary_objective": "minimize same-capture slot-6 NRMSE",
        "baseline": {
            "analysis": str(baseline_path),
            "metrics": {str(slot): base_metrics[slot] for slot in MEASURED_SLOTS},
            "final_nrmse": baseline["final_comparison"]["nrmse_vs_rtx_stddev"],
        },
        "best": best,
        "slot6_passing_targets": [
            row["target"] for row in rows if row["metrics"]["6"]["pass"]
        ],
        "candidates": rows,
        "verdict": (
            f"Best single target is {best['target']} at slot-6 NRMSE "
            f"{best['metrics']['6']['nrmse']:.12f}; "
            + ("it crosses the strict gate." if best["metrics"]["6"]["pass"]
               else "no single target crosses the strict gate.")
        ) if best else "No candidates were supplied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--glob", default="20260904_39*_full_graph_slot34_single_*_rx9070xt/analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [path.resolve(strict=True) for path in args.candidate]
    if args.results_root:
        candidates.extend(path.resolve(strict=True) for path in args.results_root.glob(args.glob))
    candidates = sorted(set(candidates))
    report = analyze(args.baseline.resolve(strict=True), candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "best_target": report["best"]["target"] if report["best"] else None,
        "best_slot6_nrmse": report["best"]["metrics"]["6"]["nrmse"]
            if report["best"] else None,
        "slot6_passing_targets": report["slot6_passing_targets"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
