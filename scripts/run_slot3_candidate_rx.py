#!/usr/bin/env python3
"""Generate and execute slot-3 FP8 MMA candidate rewrites on RX 9070 XT."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from scripts.lower_ptx_fp8_mma import FP8_MMA, candidate_model_options, lower_compact
except ModuleNotFoundError:
    from lower_ptx_fp8_mma import FP8_MMA, candidate_model_options, lower_compact


FUNCTION = "cc_tinlayout_fused_swin_1h_32_1_chained_fp8"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def probe_arguments(repo: Path, ptx: Path, output: Path) -> list[str]:
    case = repo / "results/20260901_000000_slot3_accumulation_diagnostic/case"
    plan = repo / "results/20260831_234000_full_graph_integrated_plan"
    return [
        str(repo / "build/zluda_ptx_probe.exe"),
        "--nvcuda", str(repo / ".tools/zluda-v7-preview.3/zluda/nvcuda.dll"),
        "--ptx", str(ptx), "--function", FUNCTION,
        "--json", str(output / "probe.json"),
        "--n1-arena", str(case / "activation_arena_before.raw"),
        "--n1-arena-out", str(output / "arena_after.raw"),
        "--n1-arena-input-offset", "9940992",
        "--n1-arena-output-offset", "11907072",
        "--n1-weights", str(plan / "model_arena.raw"),
        "--n1-params", str(plan / "params/slot3.raw"),
        "--n1-output", str(output / "output.raw"),
        "--n1-sync-out", str(output / "sync.raw"),
        "--n1-input-param-offset", "0", "--n1-output-param-offset", "8",
        "--n1-no-weights-param", "--n1-no-release", "--n1-expected-releases", "0",
        "--n1-grid-x", "41", "--n1-grid-y", "25", "--n1-grid-z", "1",
        "--n1-block-x", "32", "--n1-block-y", "1", "--n1-block-z", "1",
        "--n1-arena-param-view", "0:9940992",
        "--n1-arena-param-view", "8:11907072",
        "--n1-arena-param-view", "40:0",
        "--n1-arena-param-view", "56:4096",
        "--n1-weight-param-view", "16:4512256",
    ]


def run_candidate(repo: Path, model: str, root: Path) -> dict:
    output = root / model
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.mkdir(parents=True)
    source_path = repo / "results/20260831_144500_swin1h_slots3_5_lowering/chained_movmatrix.ptx"
    source = source_path.read_text(encoding="utf-8")
    options = candidate_model_options(model)
    lowered, count = lower_compact(source, **options)
    remaining = len(FP8_MMA.findall(lowered))
    if count != 256 or remaining:
        raise ValueError(f"candidate lowering incomplete: lowered={count}, remaining={remaining}")
    ptx = output / "slot3_candidate.ptx"
    ptx.write_text(lowered, encoding="utf-8", newline="\n")
    lowering = {
        "schema": 1,
        "status": "PASS",
        "candidate_model": model,
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "ptx_sha256": file_sha256(ptx),
        "fp8_mma_lowered": count,
        "remaining_fp8_mma": remaining,
        **options,
    }
    (output / "lowering.json").write_text(
        json.dumps(lowering, indent=2) + "\n", encoding="utf-8"
    )

    process = subprocess.run(
        probe_arguments(repo, ptx, output), cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    (output / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(process.stderr, encoding="utf-8")
    probe_path = output / "probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8-sig")) if probe_path.exists() else {}
    passed = (
        process.returncode == 0
        and probe.get("pass") is True
        and "AMD Radeon RX 9070 XT" in str(probe.get("device_name"))
        and probe.get("module_loaded") is True
        and probe.get("function_resolved") is True
        and probe.get("kernel_launched") is True
        and probe.get("execution_verified") is True
    )
    summary = {
        "schema": 1,
        "experiment": "slot3_candidate_rewrite_rx9070xt_execution",
        "status": "PASS" if passed else "FAIL",
        "classification": "REAL_AMD_GPU_CANDIDATE_EXECUTION",
        "candidate_model": model,
        "device_name": probe.get("device_name"),
        "process_exit": process.returncode,
        "module_loaded": probe.get("module_loaded", False),
        "function_resolved": probe.get("function_resolved", False),
        "kernel_launched": probe.get("kernel_launched", False),
        "execution_verified": probe.get("execution_verified", False),
        "output_nonzero_bytes": probe.get("n1_output_nonzero_bytes", 0),
        "ptx_sha256": lowering["ptx_sha256"],
        "output_sha256": file_sha256(output / "output.raw")
        if (output / "output.raw").is_file() else None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or (
        repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_slot3_candidate_rx_smoke"
    )
    if root.exists():
        parser.error(f"output already exists: {root}")
    summaries = []
    try:
        for model in args.candidate:
            print(f"Running {model} on RX 9070 XT...", flush=True)
            summaries.append(run_candidate(repo, model, root))
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    aggregate = {
        "schema": 1,
        "experiment": "slot3_candidate_rewrite_rx9070xt_smoke_set",
        "status": "PASS" if all(item["status"] == "PASS" for item in summaries) else "FAIL",
        "candidates": summaries,
    }
    (root / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))
    print(f"Result: {root.resolve()}")
    return 0 if aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
