#!/usr/bin/env python3
"""Execute N0 FP8 accumulation candidates on RX 9070 XT and rank against RTX."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from scripts.analyze_n0_cross_vendor import compare
    from scripts.lower_ptx_fp8_mma import FP8_MMA, candidate_model_options, lower_compact
except ModuleNotFoundError:
    from analyze_n0_cross_vendor import compare
    from lower_ptx_fp8_mma import FP8_MMA, candidate_model_options, lower_compact


FUNCTION = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def probe_arguments(repo: Path, ptx: Path, output: Path) -> list[str]:
    payload = repo / "deliverables/n0_full_reference_20260831_130219/payload"
    return [
        str(repo / "build/zluda_ptx_probe.exe"),
        "--nvcuda", str(repo / ".tools/zluda-v7-preview.3/zluda/nvcuda.dll"),
        "--ptx", str(ptx), "--function", FUNCTION,
        "--json", str(output / "probe.json"),
        "--n0-input", str(payload / "input_rgba16f.raw"),
        "--n0-weights", str(payload / "weights.raw"),
        "--n0-params", str(payload / "params.raw"),
        "--n0-scratch-out", str(output / "scratch.raw"),
        "--n0-output", str(output / "output.raw"),
        "--n0-grid-x", "80", "--n0-grid-y", "48",
    ]


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"output already exists: {root}")
    root.mkdir(parents=True)

    source_path = repo / "results/20260831_140000_n0_full_numeric_corrected/neural_f16_corrected.ptx"
    source = source_path.read_text(encoding="utf-8")
    rtx_root = repo / "results/20260831_130611_rtx5070_n0_same_input"
    baseline_root = repo / "results/20260831_140500_amd_n0_full_grid_corrected"
    references = {name: (rtx_root / f"{name}.raw").read_bytes() for name in ("scratch", "output")}
    baselines = {name: (baseline_root / f"{name}.raw").read_bytes() for name in ("scratch", "output")}
    baseline_metrics = {name: compare(references[name], baselines[name]) for name in references}
    reports = []

    for index, model in enumerate(args.candidate, 1):
        candidate_dir = root / model
        candidate_dir.mkdir()
        options = candidate_model_options(model)
        lowered, count = lower_compact(source, **options)
        remaining = len(FP8_MMA.findall(lowered))
        if count != 256 or remaining:
            raise RuntimeError(f"invalid FP8 lowering for {model}: {count}/{remaining}")
        ptx = candidate_dir / "candidate.ptx"
        ptx.write_text(lowered, encoding="utf-8", newline="\n")
        (candidate_dir / "lowering.json").write_text(json.dumps({
            "schema": 1, "status": "PASS", "candidate_model": model,
            "source_sha256": sha256(source.encode("utf-8")),
            "ptx_sha256": sha256(lowered.encode("utf-8")),
            "fp8_mma_lowered": count, "remaining_fp8_mma": remaining, **options,
        }, indent=2) + "\n", encoding="utf-8")
        process = subprocess.run(
            probe_arguments(repo, ptx, candidate_dir), cwd=repo,
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        (candidate_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (candidate_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
        probe = json.loads((candidate_dir / "probe.json").read_text(encoding="utf-8-sig"))
        execution_pass = (
            process.returncode == 0 and probe.get("pass") is True
            and "AMD Radeon RX 9070 XT" in str(probe.get("device_name"))
            and probe.get("module_loaded") is True and probe.get("function_resolved") is True
            and probe.get("kernel_launched") is True and probe.get("execution_verified") is True
        )
        tensors = {}
        for name in references:
            data = (candidate_dir / f"{name}.raw").read_bytes()
            metrics = compare(references[name], data)
            tensors[name] = {
                "sha256": sha256(data), "equals_baseline": data == baselines[name],
                "nrmse": metrics["nrmse_vs_rtx_stddev"],
                "correlation": metrics["pearson_correlation"],
                "exact_fraction": metrics["exact_fraction"],
                "nrmse_delta_vs_baseline": (
                    metrics["nrmse_vs_rtx_stddev"]
                    - baseline_metrics[name]["nrmse_vs_rtx_stddev"]
                ),
            }
        item = {"candidate_model": model, "execution_pass": execution_pass,
                "device_name": probe.get("device_name"), "tensors": tensors}
        reports.append(item)
        print(f"[{index:02d}/{len(args.candidate)}] {model}: "
              f"output={tensors['output']['nrmse']:.12f} "
              f"delta={tensors['output']['nrmse_delta_vs_baseline']:+.12f}", flush=True)

    reports.sort(key=lambda item: (
        item["tensors"]["output"]["nrmse"], item["tensors"]["scratch"]["nrmse"]
    ))
    report = {
        "schema": 1, "experiment": "n0_fp8_accumulation_candidate_search_rx9070xt",
        "status": "PASS" if all(item["execution_pass"] for item in reports) else "FAIL",
        "classification": "REAL_AMD_GPU_N0_NUMERICAL_SEARCH_NOT_S7",
        "counts_as_s7": False, "candidate_count": len(reports),
        "baseline": {name: {
            "nrmse": baseline_metrics[name]["nrmse_vs_rtx_stddev"],
            "correlation": baseline_metrics[name]["pearson_correlation"],
            "exact_fraction": baseline_metrics[name]["exact_fraction"],
        } for name in references},
        "best_candidate": reports[0]["candidate_model"],
        "best_improves_output": reports[0]["tensors"]["output"]["nrmse_delta_vs_baseline"] < 0,
        "candidates": reports,
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "best_candidate": report["best_candidate"],
                      "best_improves_output": report["best_improves_output"],
                      "best": reports[0]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
