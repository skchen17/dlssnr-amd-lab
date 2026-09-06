#!/usr/bin/env python3
"""Verify same-input N1 slot-2 parity between RTX 5070 and RX 9070 XT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
from pathlib import Path


TENSOR_BYTES = 192 * 320 * 32
SYNC_WORDS = 27_648
EXPECTED_RELEASES = 40 * 24
EXPECTED_DEVICE_RTX = "NVIDIA GeForce RTX 5070"
EXPECTED_DEVICE_AMD = "AMD Radeon RX 9070 XT [ZLUDA]"
EXPECTED_GRID = [40, 24, 1]
EXPECTED_BLOCK = [32, 1, 1]


def _load_tensor_comparator():
    path = Path(__file__).with_name("analyze_n0_cross_vendor.py")
    spec = importlib.util.spec_from_file_location("_n0_tensor_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tensor comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMPARE = _load_tensor_comparator()
compare = _COMPARE.compare
raw_summary = _COMPARE.raw_summary
sha256 = _COMPARE.sha256


def sync_summary(data: bytes) -> dict:
    if len(data) != SYNC_WORDS * 4:
        raise ValueError(f"expected {SYNC_WORDS * 4} sync bytes, got {len(data)}")
    words = struct.unpack(f"<{SYNC_WORDS}I", data)
    zero = sum(value == 0 for value in words)
    sentinel = sum(value == 0xFFFFFFFF for value in words)
    return {
        "bytes": len(data),
        "zero_words": zero,
        "sentinel_words": sentinel,
        "unexpected_words": len(words) - zero - sentinel,
        "sha256": sha256(data),
    }


def analyze(rtx_dir: Path, amd_dir: Path, repeat_dir: Path, payload_dir: Path) -> dict:
    manifest = json.loads((rtx_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    rtx_probe = json.loads((rtx_dir / "probe.json").read_text(encoding="utf-8-sig"))
    amd_probe = json.loads((amd_dir / "probe.json").read_text(encoding="utf-8-sig"))
    amd_verification = json.loads((amd_dir / "verification.json").read_text(encoding="utf-8-sig"))

    rtx_output = (rtx_dir / "output.raw").read_bytes()
    amd_output = (amd_dir / "output.raw").read_bytes()
    repeat_output = (repeat_dir / "output.raw").read_bytes()
    rtx_sync = (rtx_dir / "sync.raw").read_bytes()
    amd_sync = (amd_dir / "sync.raw").read_bytes()
    repeat_sync = (repeat_dir / "sync.raw").read_bytes()

    payload_hashes = {
        name: sha256((payload_dir / name).read_bytes())
        for name in ("input.raw", "weights.raw", "params.raw")
    }
    payload_integrity = all(
        payload_hashes[name] == manifest.get("payload_sha256", {}).get(name)
        for name in payload_hashes
    )
    rtx_sync_info = sync_summary(rtx_sync)
    amd_sync_info = sync_summary(amd_sync)
    comparison = compare(rtx_output, amd_output)
    deterministic = amd_output == repeat_output and amd_sync == repeat_sync
    sync_exact = rtx_sync == amd_sync

    rtx_checks = {
        "manifest_pass": manifest.get("status") == "PASS",
        "device": manifest.get("device_name") == EXPECTED_DEVICE_RTX,
        "grid": manifest.get("grid") == EXPECTED_GRID,
        "block": manifest.get("block") == EXPECTED_BLOCK,
        "kernel_launched": manifest.get("kernel_launched") is True,
        "probe_pass": rtx_probe.get("pass") is True,
        "probe_execution_verified": rtx_probe.get("execution_verified") is True,
        "output_size": len(rtx_output) == TENSOR_BYTES,
        "output_hash": manifest.get("output_sha256") == sha256(rtx_output),
        "sync_release_count": rtx_sync_info["zero_words"] == EXPECTED_RELEASES,
        "sync_sentinel": rtx_sync_info["sentinel_words"] == SYNC_WORDS - EXPECTED_RELEASES,
        "sync_no_unexpected": rtx_sync_info["unexpected_words"] == 0,
    }
    amd_checks = {
        "verification_pass": amd_verification.get("status") == "PASS",
        "device": amd_probe.get("device_name") == EXPECTED_DEVICE_AMD,
        "kernel_launched": amd_probe.get("kernel_launched") is True,
        "execution_verified": amd_probe.get("execution_verified") is True,
        "output_size": len(amd_output) == TENSOR_BYTES,
        "sync_release_count": amd_sync_info["zero_words"] == EXPECTED_RELEASES,
        "sync_sentinel": amd_sync_info["sentinel_words"] == SYNC_WORDS - EXPECTED_RELEASES,
        "sync_no_unexpected": amd_sync_info["unexpected_words"] == 0,
        "repeat_deterministic": deterministic,
    }
    execution_gate = all(rtx_checks.values()) and all(amd_checks.values())
    parity_gate = comparison["parity_gate"]["pass"] and sync_exact
    passed = execution_gate and payload_integrity and parity_gate
    return {
        "schema": 1,
        "experiment": "n1_slot2_same_input_rtx5070_vs_rx9070xt",
        "status": "PASS" if passed else "FAIL",
        "classification": "CROSS_VENDOR_N1_SLOT2_NUMERICAL_PARITY",
        "counts_as_s7": False,
        "execution_gate": execution_gate,
        "payload_integrity_gate": payload_integrity,
        "parity_gate": parity_gate,
        "amd_deterministic_two_runs": deterministic,
        "payload_sha256": payload_hashes,
        "rtx_checks": rtx_checks,
        "amd_checks": amd_checks,
        "output": {
            "rtx": raw_summary(rtx_output),
            "amd": raw_summary(amd_output),
            "comparison": comparison,
        },
        "sync": {
            "rtx": rtx_sync_info,
            "amd": amd_sync_info,
            "bitwise_exact": sync_exact,
        },
        "qualification": (
            "second consecutive DLSSNR-originated learned neural graph kernel executed "
            "on RX 9070 XT with identical-input RTX 5070 numerical parity"
        ),
        "limitations": [
            "isolated slot-2 harness; slots 3-5 and the complete 156-slot graph remain pending",
            "controlled captured N0 output rather than an end-to-end rendered frame",
            "scalar correctness lowering is not yet an efficient AMD matrix implementation",
        ],
        "next_gate": "capture the slot 3-5 weight views and execute the chained dependency sequence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-dir", required=True, type=Path)
    parser.add_argument("--amd-dir", required=True, type=Path)
    parser.add_argument("--repeat-dir", required=True, type=Path)
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.rtx_dir.resolve(), args.amd_dir.resolve(), args.repeat_dir.resolve(),
        args.payload_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
