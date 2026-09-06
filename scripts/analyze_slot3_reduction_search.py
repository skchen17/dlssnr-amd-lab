#!/usr/bin/env python3
"""Consolidate executed slot-3 reduction candidates against the RTX reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.analyze_n0_cross_vendor import compare
except ModuleNotFoundError:
    from analyze_n0_cross_vendor import compare


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze(roots: list[Path], reference_path: Path, baseline_path: Path) -> dict:
    reference = reference_path.read_bytes()
    baseline = baseline_path.read_bytes()
    baseline_metrics = compare(reference, baseline)
    candidates = []
    for root in roots:
        ablation_targets = {}
        ablation_path = root / "ablation.json"
        if ablation_path.is_file():
            ablation = json.loads(ablation_path.read_text(encoding="utf-8-sig"))
            ablation_targets = {
                item["target"].removeprefix("r").removeprefix("%r"): item["target"]
                for item in ablation.get("candidates", [])
            }
        for output_path in root.rglob("output.raw"):
            directory = output_path.parent
            reduction_paths = list(directory.glob("*reduction.json"))
            probe_path = directory / "probe.json"
            if not probe_path.is_file():
                continue
            if len(reduction_paths) == 1:
                reduction = json.loads(reduction_paths[0].read_text(encoding="utf-8-sig"))
                targets = [item["destination"] for item in reduction["replacements"]]
            elif directory.name.removeprefix("r") in ablation_targets:
                targets = [ablation_targets[directory.name.removeprefix("r")]]
            else:
                continue
            probe = json.loads(probe_path.read_text(encoding="utf-8-sig"))
            output = output_path.read_bytes()
            metrics = compare(reference, output)
            execution_pass = (
                probe.get("pass") is True
                and "AMD Radeon RX 9070 XT" in str(probe.get("device_name"))
                and probe.get("module_loaded") is True
                and probe.get("function_resolved") is True
                and probe.get("kernel_launched") is True
                and probe.get("execution_verified") is True
            )
            candidates.append({
                "path": str(directory),
                "targets": targets,
                "execution_pass": execution_pass,
                "output_sha256": sha256(output),
                "nrmse": metrics["nrmse_vs_rtx_stddev"],
                "correlation": metrics["pearson_correlation"],
                "exact_fraction": metrics["exact_fraction"],
                "exact_or_adjacent_fraction": metrics["exact_or_adjacent_fraction"],
                "sign_mismatches": metrics["sign_mismatches"],
                "nrmse_delta_vs_baseline": (
                    metrics["nrmse_vs_rtx_stddev"]
                    - baseline_metrics["nrmse_vs_rtx_stddev"]
                ),
            })
    candidates.sort(key=lambda item: (item["nrmse"], -item["correlation"]))
    best = candidates[0]
    return {
        "schema": 1,
        "experiment": "slot3_selective_f16_reduction_search_rx9070xt",
        "status": "PASS" if candidates and all(x["execution_pass"] for x in candidates) else "FAIL",
        "classification": "REAL_AMD_GPU_NUMERICAL_SEARCH_NOT_S7",
        "counts_as_s7": False,
        "reference_sha256": sha256(reference),
        "baseline_sha256": sha256(baseline),
        "candidate_count": len(candidates),
        "baseline": {
            "nrmse": baseline_metrics["nrmse_vs_rtx_stddev"],
            "correlation": baseline_metrics["pearson_correlation"],
            "exact_fraction": baseline_metrics["exact_fraction"],
            "exact_or_adjacent_fraction": baseline_metrics["exact_or_adjacent_fraction"],
            "sign_mismatches": baseline_metrics["sign_mismatches"],
        },
        "best": best,
        "relative_nrmse_improvement": (
            -best["nrmse_delta_vs_baseline"] / baseline_metrics["nrmse_vs_rtx_stddev"]
        ),
        "candidates": candidates,
        "limitation": (
            "Search is measured on exact captured slot-3 state and is not a full-graph or "
            "in-game S7 result; selected targets are an empirical local optimum."
        ),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, default=repo / "results/20260901_000000_slot3_accumulation_diagnostic/case/output_reference.raw")
    parser.add_argument("--baseline", type=Path, default=repo / "results/20260901_000000_slot3_accumulation_diagnostic/runs/baseline/output.raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze([path.resolve() for path in args.root], args.reference.resolve(), args.baseline.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidate_count": report["candidate_count"],
        "best": report["best"], "relative_nrmse_improvement": report["relative_nrmse_improvement"],
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
