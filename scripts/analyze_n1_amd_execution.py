#!/usr/bin/env python3
"""Verify the isolated slot-2 N1 execution on RX 9070 XT."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


TENSOR_BYTES = 192 * 320 * 32
SYNC_WORDS = 27648
EXPECTED_RELEASES = 40 * 24


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sync_summary(data: bytes) -> dict:
    if len(data) != SYNC_WORDS * 4:
        raise ValueError(f"expected {SYNC_WORDS * 4} sync bytes, got {len(data)}")
    words = struct.unpack(f"<{SYNC_WORDS}I", data)
    zero = sum(value == 0 for value in words)
    sentinel = sum(value == 0xFFFFFFFF for value in words)
    unexpected = len(words) - zero - sentinel
    return {"bytes": len(data), "zero_words": zero,
            "sentinel_words": sentinel, "unexpected_words": unexpected,
            "sha256": sha256(data)}


def analyze(result_dir: Path, input_path: Path, weights_path: Path,
            repeat_dir: Path | None = None) -> dict:
    probe = json.loads((result_dir / "probe.json").read_text(encoding="utf-8"))
    params_manifest = json.loads((result_dir / "params.json").read_text(encoding="utf-8"))
    output = (result_dir / "output.raw").read_bytes()
    sync = (result_dir / "sync.raw").read_bytes()
    params = (result_dir / "params.raw").read_bytes()
    sync_info = sync_summary(sync)
    output_info = {"bytes": len(output),
                   "nonzero_bytes": sum(value != 0 for value in output),
                   "sha256": sha256(output)}
    repeat = None
    if repeat_dir is not None:
        repeat_output = (repeat_dir / "output.raw").read_bytes()
        repeat_sync = (repeat_dir / "sync.raw").read_bytes()
        repeat = {
            "output_sha256": sha256(repeat_output),
            "sync_sha256": sha256(repeat_sync),
            "bitwise_deterministic": repeat_output == output and repeat_sync == sync,
        }
    checks = {
        "rx9070xt": probe.get("device_name") == "AMD Radeon RX 9070 XT [ZLUDA]",
        "module_loaded": probe.get("module_loaded") is True,
        "function_resolved": probe.get("function_resolved") is True,
        "kernel_launched": probe.get("kernel_launched") is True,
        "execution_verified": probe.get("execution_verified") is True,
        "mode": probe.get("execution_mode") == "n1_slot2",
        "output_size": len(output) == TENSOR_BYTES,
        "output_nonzero": output_info["nonzero_bytes"] > 0,
        "release_count": sync_info["zero_words"] == EXPECTED_RELEASES,
        "untouched_sync_sentinel": sync_info["sentinel_words"] == SYNC_WORDS - EXPECTED_RELEASES,
        "no_unexpected_sync_words": sync_info["unexpected_words"] == 0,
        "captured_param_integrity": params_manifest.get("param_sha256") == sha256(params),
        "captured_slot": params_manifest.get("slot") == 2,
        "repeat_deterministic": repeat is None or repeat["bitwise_deterministic"],
    }
    passed = all(checks.values())
    return {
        "schema": 1,
        "experiment": "amd_n1_slot2_real_execution",
        "status": "PASS" if passed else "FAIL",
        "classification": "REAL_AMD_NEURAL_GRAPH_SUCCESSOR_EXECUTION_PENDING_RTX_ORACLE",
        "counts_as_s7": False,
        "function": probe.get("function"),
        "grid": [40, 24, 1],
        "block": [32, 1, 1],
        "input": {"bytes": input_path.stat().st_size, "sha256": sha256(input_path.read_bytes())},
        "weights": {"bytes": weights_path.stat().st_size, "sha256": sha256(weights_path.read_bytes()),
                    "view_offset": 0x5600},
        "params": {"bytes": len(params), "sha256": sha256(params)},
        "output": output_info,
        "sync": sync_info,
        "repeat": repeat,
        "checks": checks,
        "next_gate": "run original N1 PTX over the identical payload on RTX 5070 and compare output.raw",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--repeat-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.result_dir.resolve(), args.input.resolve(), args.weights.resolve(),
                     args.repeat_dir.resolve() if args.repeat_dir else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
