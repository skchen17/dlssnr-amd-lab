#!/usr/bin/env python3
"""Run one-target-at-a-time slot-3 reduction fusion ablations on RX 9070 XT."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from scripts.analyze_n0_cross_vendor import compare
    from scripts.lower_ptx_f16x2_local_reduction import TARGETS, lower
    from scripts.lower_ptx_fp8_mma import FP8_MMA, lower_compact
    from scripts.run_slot3_candidate_rx import probe_arguments
except ModuleNotFoundError:
    from analyze_n0_cross_vendor import compare
    from lower_ptx_f16x2_local_reduction import TARGETS, lower
    from lower_ptx_fp8_mma import FP8_MMA, lower_compact
    from run_slot3_candidate_rx import probe_arguments


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"output already exists: {root}")
    root.mkdir(parents=True)

    source_path = repo / "results/20260831_144500_swin1h_slots3_5_lowering/chained_movmatrix.ptx"
    reference_path = repo / "results/20260901_000000_slot3_accumulation_diagnostic/case/output_reference.raw"
    baseline_path = repo / "results/20260901_000000_slot3_accumulation_diagnostic/runs/baseline/output.raw"
    source = source_path.read_text(encoding="utf-8")
    reference = reference_path.read_bytes()
    baseline = baseline_path.read_bytes()
    baseline_metrics = compare(reference, baseline)
    reports = []

    for index, target in enumerate(TARGETS):
        name = target.removeprefix("%")
        candidate_dir = root / name
        candidate_dir.mkdir()
        fused, replacements = lower(source, (target,))
        candidate, count = lower_compact(fused)
        remaining = len(FP8_MMA.findall(candidate))
        if len(replacements) != 1 or count != 256 or remaining:
            raise RuntimeError(f"invalid lowering for {target}")
        ptx = candidate_dir / "candidate.ptx"
        ptx.write_text(candidate, encoding="utf-8", newline="\n")
        process = subprocess.run(
            probe_arguments(repo, ptx, candidate_dir), cwd=repo,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False,
        )
        (candidate_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (candidate_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
        probe = json.loads((candidate_dir / "probe.json").read_text(encoding="utf-8-sig"))
        output = (candidate_dir / "output.raw").read_bytes()
        metrics = compare(reference, output)
        execution_pass = (
            process.returncode == 0 and probe.get("pass") is True
            and "AMD Radeon RX 9070 XT" in str(probe.get("device_name"))
            and probe.get("kernel_launched") is True
            and probe.get("execution_verified") is True
        )
        item = {
            "target": target,
            "status": "PASS" if execution_pass else "FAIL",
            "output_sha256": sha256(output),
            "equals_baseline": output == baseline,
            "nrmse": metrics["nrmse_vs_rtx_stddev"],
            "correlation": metrics["pearson_correlation"],
            "exact_fraction": metrics["exact_fraction"],
            "nrmse_delta_vs_baseline": (
                metrics["nrmse_vs_rtx_stddev"]
                - baseline_metrics["nrmse_vs_rtx_stddev"]
            ),
        }
        reports.append(item)
        print(f"[{index + 1:02d}/{len(TARGETS)}] {target}: nrmse={item['nrmse']:.12f} "
              f"delta={item['nrmse_delta_vs_baseline']:+.12f}", flush=True)

    reports.sort(key=lambda item: item["nrmse"])
    report = {
        "schema": 1,
        "experiment": "slot3_local_reduction_single_target_ablation_rx9070xt",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "classification": "REAL_AMD_GPU_NUMERICAL_ABLATION_NOT_S7",
        "counts_as_s7": False,
        "device_name": "AMD Radeon RX 9070 XT [ZLUDA]",
        "reference_sha256": sha256(reference),
        "baseline_sha256": sha256(baseline),
        "baseline": {
            "nrmse": baseline_metrics["nrmse_vs_rtx_stddev"],
            "correlation": baseline_metrics["pearson_correlation"],
            "exact_fraction": baseline_metrics["exact_fraction"],
        },
        "beneficial_targets": [
            item["target"] for item in reports if item["nrmse_delta_vs_baseline"] < 0
        ],
        "candidates": reports,
    }
    (root / "ablation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "beneficial_targets": report["beneficial_targets"],
        "best": reports[0],
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
