#!/usr/bin/env python3
"""Execute cosine-correction N0 candidates on RX 9070 XT and rank full output."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from scripts.analyze_n0_cross_vendor import compare
    from scripts.lower_ptx_nvidia_cos_correction import lower
except ModuleNotFoundError:
    from analyze_n0_cross_vendor import compare
    from lower_ptx_nvidia_cos_correction import lower


FUNCTION = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"output already exists: {root}")
    root.mkdir(parents=True)
    source_path = repo / "results/20260902_150000_n0_square_pair_fusion_candidate/full.ptx"
    source = source_path.read_text(encoding="utf-8")
    reference_path = repo / "results/20260831_151635_rtx5070_feature18_v17/n0_output_after.raw"
    baseline_path = repo / "results/20260902_150000_n0_square_pair_fusion_candidate/output.raw"
    reference = reference_path.read_bytes()
    baseline = baseline_path.read_bytes()
    payload = repo / "deliverables/n0_full_reference_20260831_130219/payload"
    input_path = repo / "results/20260901_153000_n0_zero_input/input_rgba16f_zero.raw"
    reports = []
    for index, model_path in enumerate(args.model, 1):
        model = json.loads(model_path.read_text(encoding="utf-8"))
        name = f"seg{model['segments']}_deg{model['degree']}"
        candidate = root / name
        candidate.mkdir()
        text, count = lower(source, model)
        if count != 1:
            raise RuntimeError(f"invalid lowering count for {name}: {count}")
        ptx = candidate / "candidate.ptx"
        ptx.write_text(text, encoding="utf-8", newline="\n")
        probe_path = candidate / "probe.json"
        output_path = candidate / "output.raw"
        command = [
            str(repo / "build/zluda_ptx_probe.exe"),
            "--nvcuda", str(repo / ".tools/zluda-v7-preview.3/zluda/nvcuda.dll"),
            "--ptx", str(ptx), "--function", FUNCTION,
            "--json", str(probe_path),
            "--n0-input", str(input_path),
            "--n0-weights", str(payload / "weights.raw"),
            "--n0-params", str(payload / "params.raw"),
            "--n0-scratch-out", str(candidate / "scratch.raw"),
            "--n0-output", str(output_path),
            "--n0-grid-x", "80", "--n0-grid-y", "48",
        ]
        process = subprocess.run(command, cwd=repo, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", check=False)
        (candidate / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (candidate / "stderr.log").write_text(process.stderr, encoding="utf-8")
        probe = json.loads(probe_path.read_text(encoding="utf-8-sig"))
        output = output_path.read_bytes()
        metrics = compare(reference, output)
        fixed = sum(before != expected and after == expected
                    for expected, before, after in zip(reference, baseline, output))
        added = sum(before == expected and after != expected
                    for expected, before, after in zip(reference, baseline, output))
        report = {
            "candidate": name,
            "model": str(model_path),
            "model_status": model.get("status"),
            "execution_pass": process.returncode == 0 and probe.get("pass") is True,
            "ptx_sha256": sha256(text.encode("utf-8")),
            "output_sha256": sha256(output),
            "mismatching_bytes": len(reference) - metrics["exact"],
            "nrmse": metrics["nrmse_vs_rtx_stddev"],
            "fixed_old_mismatches": fixed,
            "added_new_mismatches": added,
        }
        (candidate / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
        print(f"[{index}/{len(args.model)}] {name}: mismatch={report['mismatching_bytes']} "
              f"fixed={fixed} added={added} nrmse={report['nrmse']:.12f}", flush=True)
    reports.sort(key=lambda row: (row["mismatching_bytes"], row["nrmse"]))
    summary = {
        "schema": 1,
        "experiment": "n0_cos_correction_candidate_search_rx9070xt",
        "status": "PASS" if all(row["execution_pass"] for row in reports) else "FAIL",
        "classification": "REAL_RX9070XT_FULL_N0_NUMERICAL_SEARCH_NOT_S7",
        "baseline_mismatching_bytes": 241,
        "candidate_count": len(reports),
        "best_candidate": reports[0]["candidate"],
        "best_improves_output": reports[0]["mismatching_bytes"] < 241,
        "candidates": reports,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
