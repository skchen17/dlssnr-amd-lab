#!/usr/bin/env python3
"""Summarize a same-capture partial RTX-state injection sweep.

This is causal diagnostic evidence only.  Every candidate contains RTX state and
therefore can never count as end-to-end/S7 AMD evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MEASURED_SLOTS = (4, 5, 6, 7, 8, 9, 10, 15)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nrmse(report: dict, slot: int) -> float:
    return float(report["boundaries"][str(slot)]["comparison"]["nrmse_vs_rtx_stddev"])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze(baseline_path: Path, candidate_paths: list[Path]) -> dict:
    baseline_path = baseline_path.resolve(strict=True)
    baseline = _read(baseline_path)
    baseline_execution = _read(baseline_path.parent / "execution.json")
    if baseline.get("execution_gate") is not True or baseline_execution.get("pass") is not True:
        raise ValueError("baseline execution gate failed")
    if baseline_execution.get("rtx_intermediate_state_injection") is not False:
        raise ValueError("baseline contains RTX state injection")

    baseline_metrics = {slot: _nrmse(baseline, slot) for slot in MEASURED_SLOTS}
    baseline_final = float(baseline["final_comparison"]["nrmse_vs_rtx_stddev"])
    baseline_slot3 = (baseline_path.parent / "run1" / "slot3.raw").read_bytes()

    rows = []
    seen_ranges: set[tuple[int, int]] = set()
    reference_identity: tuple[Path, int] | None = None
    for analysis_path in sorted(path.resolve(strict=True) for path in candidate_paths):
        report = _read(analysis_path)
        execution = _read(analysis_path.parent / "execution.json")
        if execution.get("pass") is not True:
            raise ValueError(f"candidate execution failed: {analysis_path}")
        if execution.get("rtx_intermediate_state_injection") is not True:
            raise ValueError(f"candidate has no declared RTX injection: {analysis_path}")
        runs = execution.get("runs", [])
        if len(runs) != 2 or runs[0].get("final_sha256") != runs[1].get("final_sha256"):
            raise ValueError(f"candidate is not two-run deterministic: {analysis_path}")
        injections = execution.get("diagnostic_injections", [])
        if len(injections) != 1 or int(injections[0]["after_slot"]) != 3:
            raise ValueError(f"candidate is not a single slot-3 injection: {analysis_path}")
        injection = injections[0]
        offset = int(injection["reference_offset"])
        size = int(injection["bytes"])
        key = (offset, size)
        if key in seen_ranges:
            raise ValueError(f"duplicate injection range {key}")
        seen_ranges.add(key)

        plan_path = Path(execution["plan"]).resolve(strict=True)
        plan = _read(plan_path)
        plan_injection = plan["diagnostic_injections"][0]
        reference_path = (plan_path.parent / plan_injection["reference"]["path"]).resolve(strict=True)
        logical_bytes = int(plan["checkpoints"]["3"]["logical_bytes"])
        identity = (reference_path, logical_bytes)
        if reference_identity is None:
            reference_identity = identity
        elif reference_identity != identity:
            raise ValueError("candidates do not use one slot-3 reference tensor")
        reference = reference_path.read_bytes()
        reference_slice = reference[offset:offset + size]
        if len(reference_slice) != size:
            raise ValueError(f"reference slice is short for range {key}")
        if _sha256(reference_slice) != injection["sha256"]:
            raise ValueError(f"reference slice hash differs for range {key}")
        native_slice = baseline_slot3[offset:offset + size]
        if len(native_slice) != size:
            raise ValueError(f"baseline slot-3 slice is short for range {key}")
        differing = sum(a != b for a, b in zip(native_slice, reference_slice))

        metrics = {}
        for slot in MEASURED_SLOTS:
            candidate_boundary = report["boundaries"][str(slot)]
            baseline_boundary = baseline["boundaries"][str(slot)]
            if candidate_boundary["rtx_sha256"] != baseline_boundary["rtx_sha256"]:
                raise ValueError(f"candidate reference differs at slot {slot}: {analysis_path}")
            value = _nrmse(report, slot)
            metrics[str(slot)] = {
                "nrmse": value,
                "delta_vs_baseline": value - baseline_metrics[slot],
                "pass": candidate_boundary["comparison"]["parity_gate"]["pass"],
            }
        final = float(report["final_comparison"]["nrmse_vs_rtx_stddev"])
        rows.append({
            "analysis": str(analysis_path),
            "reference_offset": offset,
            "bytes": size,
            "end_offset_exclusive": offset + size,
            "slot3_differing_bytes_replaced": differing,
            "slot3_differing_fraction_in_slice": differing / size,
            "metrics": metrics,
            "final_nrmse": final,
            "final_delta_vs_baseline": final - baseline_final,
            "final_sha256": report["final_amd_sha256"],
        })

    rows.sort(key=lambda row: (
        row["metrics"]["6"]["nrmse"],
        row["metrics"]["8"]["nrmse"],
        row["reference_offset"],
    ))
    by_offset = sorted(rows, key=lambda row: row["reference_offset"])
    contiguous = bool(by_offset) and by_offset[0]["reference_offset"] == 0
    for left, right in zip(by_offset, by_offset[1:]):
        contiguous &= left["end_offset_exclusive"] == right["reference_offset"]
    covered_bytes = sum(row["bytes"] for row in by_offset)
    full_coverage = bool(reference_identity) and contiguous and covered_bytes == reference_identity[1]
    passing = [row for row in rows if row["metrics"]["6"]["pass"]]
    best = rows[0] if rows else None
    return {
        "schema": 1,
        "experiment": "slot3_partial_rtx_state_injection_sensitivity_sweep",
        "status": "PASS",
        "classification": "REAL_AMD_CAUSAL_DIAGNOSTIC_WITH_RTX_STATE_NOT_S7",
        "counts_as_s7": False,
        "candidate_count": len(rows),
        "all_candidates_two_run_deterministic": True,
        "range_coverage": {
            "contiguous": contiguous,
            "covered_bytes": covered_bytes,
            "logical_bytes": reference_identity[1] if reference_identity else None,
            "full_tensor": full_coverage,
        },
        "baseline": {
            "analysis": str(baseline_path),
            "metrics": {str(slot): baseline_metrics[slot] for slot in MEASURED_SLOTS},
            "final_nrmse": baseline_final,
        },
        "primary_objective": "localize slot-3 state errors by downstream slot-6 NRMSE",
        "best_slot6_range": best,
        "slot6_passing_range_count": len(passing),
        "all_ranges_make_slot6_pass": bool(rows) and len(passing) == len(rows),
        "candidates_ranked_by_slot6": rows,
        "verdict": (
            "All disjoint ranges independently make slot 6 pass; slot-3 error sensitivity "
            "is distributed across the tensor rather than isolated to one coarse byte range."
            if rows and len(passing) == len(rows)
            else "Only a subset of ranges changes the slot-6 gate; refine the best range."
        ),
        "limitations": [
            "Every candidate injects RTX intermediate state and can never count as S7.",
            "Byte ranges are not labeled as channels or tiles until tensor layout is proven.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--glob",
        default="20260904_402*_full_graph_partial_inject_slot3_chunk*_rx9070xt/analysis.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = list(args.candidate)
    if args.results_root:
        candidates.extend(args.results_root.glob(args.glob))
    report = analyze(args.baseline, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "full_tensor_coverage": report["range_coverage"]["full_tensor"],
        "all_ranges_make_slot6_pass": report["all_ranges_make_slot6_pass"],
        "best_offset": report["best_slot6_range"]["reference_offset"]
            if report["best_slot6_range"] else None,
        "best_slot6_nrmse": report["best_slot6_range"]["metrics"]["6"]["nrmse"]
            if report["best_slot6_range"] else None,
        "counts_as_s7": False,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
