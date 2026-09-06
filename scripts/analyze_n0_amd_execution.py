#!/usr/bin/env python3
"""Independently verify the single-CTA and full-grid AMD N0 executions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCRATCH_BYTES = 7_864_320
OUTPUT_BYTES = 1_966_080


def raw_summary(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "nonzero_bytes": sum(value != 0 for value in data),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def successful_step(probe: dict, name: str) -> bool:
    return any(
        step.get("name") == name and step.get("code") == 0
        for step in probe.get("steps", [])
    )


def verify_run(path: Path, expected_grid: list[int]) -> tuple[dict, bool]:
    probe = json.loads((path / "probe.json").read_text(encoding="utf-8"))
    scratch = raw_summary(path / "scratch.raw")
    output = raw_summary(path / "output.raw")
    checks = {
        "rx9070xt_device": "RX 9070 XT" in str(probe.get("device_name")),
        "module_loaded": probe.get("module_loaded") is True,
        "function_resolved": probe.get("function_resolved") is True,
        "n0_synthetic_mode": probe.get("execution_mode") == "n0_synthetic",
        "expected_grid": probe.get("n0_grid") == expected_grid,
        "kernel_launched": probe.get("kernel_launched") is True,
        "execution_verified": probe.get("execution_verified") is True,
        "launch_api_success": successful_step(probe, "cuLaunchKernel_n0"),
        "synchronize_api_success": successful_step(probe, "cuCtxSynchronize_n0"),
        "scratch_size": scratch["bytes"] == SCRATCH_BYTES,
        "output_size": output["bytes"] == OUTPUT_BYTES,
        "scratch_nonzero": scratch["nonzero_bytes"] > 0,
        "output_nonzero": output["nonzero_bytes"] > 0,
        "scratch_count_matches_probe": (
            scratch["nonzero_bytes"] == probe.get("n0_scratch_nonzero_bytes")
        ),
        "output_count_matches_probe": (
            output["nonzero_bytes"] == probe.get("n0_output_nonzero_bytes")
        ),
    }
    steps = {
        step["name"]: {
            "code": step.get("code"),
            "milliseconds": step.get("milliseconds"),
        }
        for step in probe.get("steps", [])
        if step.get("name") in {
            "cuModuleLoadData",
            "cuLaunchKernel_n0",
            "cuCtxSynchronize_n0",
        }
    }
    return {
        "path": path.as_posix(),
        "device_name": probe.get("device_name"),
        "grid": probe.get("n0_grid"),
        "block": [32, 1, 1],
        "scratch": scratch,
        "output": output,
        "timings": steps,
        "checks": checks,
    }, all(checks.values())


def analyze(root: Path) -> dict:
    single, single_ok = verify_run(
        root / "results/20260831_134000_amd_n0_single_cta", [1, 1, 1]
    )
    full, full_ok = verify_run(
        root / "results/20260831_135000_amd_n0_full_grid", [80, 48, 1]
    )
    full_density_ok = (
        full["scratch"]["nonzero_bytes"] > SCRATCH_BYTES * 0.99
        and full["output"]["nonzero_bytes"] > OUTPUT_BYTES * 0.99
    )
    status = "PASS" if single_ok and full_ok and full_density_ok else "FAIL"
    return {
        "schema": 1,
        "experiment": "amd_n0_real_execution",
        "status": status,
        "classification": "REAL_AMD_FULL_N0_EXECUTION_PENDING_RTX_ORACLE",
        "counts_as_s6": False,
        "single_cta": single,
        "full_grid": full,
        "full_grid_density_gate": {
            "threshold": 0.99,
            "scratch_nonzero_fraction": full["scratch"]["nonzero_bytes"] / SCRATCH_BYTES,
            "output_nonzero_fraction": full["output"]["nonzero_bytes"] / OUTPUT_BYTES,
            "pass": full_density_ok,
        },
        "warning": (
            "This proves real full-grid N0 execution on RX 9070 XT. S6 remains "
            "open until original RTX PTX is run over the identical synthetic "
            "input, weights and parameter block and the tensors are compared."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
