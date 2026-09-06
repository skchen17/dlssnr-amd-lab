#!/usr/bin/env python3
"""Verify deterministic RX 9070 XT execution of Swin graph slots 3-5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TENSOR_BYTES = 1_966_080
EXPECTED_RELEASES = {3: 1025, 4: 984, 5: 0}
EXPECTED_GRIDS = {3: [41, 25, 1], 4: [41, 24, 1], 5: [40, 25, 1]}
WEIGHT_NAMES = {3: "slot3_weights", 4: "slot4_weights", 5: "slot5_weights"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def analyze(capture_dir: Path, run_dir: Path, repeat_dir: Path) -> dict:
    capture_summary = json.loads((capture_dir / "summary.json").read_text(encoding="utf-8-sig"))
    capture = json.loads((capture_dir / "n0_preblock_capture.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    repeat_manifest = json.loads((repeat_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    capture_windows = {item["name"]: item for item in capture["windows"]}

    capture_checks = {
        "package_pass": capture_summary.get("status") == "PASS",
        "capture_pass": capture.get("status") == "PASS",
        "all_weight_windows_present": all(name in capture_windows for name in WEIGHT_NAMES.values()),
        "all_weight_windows_64k": all(
            capture_windows.get(name, {}).get("capture_bytes") == 65_536
            for name in WEIGHT_NAMES.values()
        ),
        "all_weight_windows_immutable": all(
            capture_windows.get(name, {}).get("changed_bytes") == 0
            for name in WEIGHT_NAMES.values()
        ),
    }
    run_slots = {item["slot"]: item for item in manifest.get("slots", [])}
    repeat_slots = {item["slot"]: item for item in repeat_manifest.get("slots", [])}
    slot_reports = {}
    deterministic = True
    chain_integrity = True
    for slot in (3, 4, 5):
        current = run_slots.get(slot, {})
        repeated = repeat_slots.get(slot, {})
        output = run_dir / f"slot{slot}" / "output.raw"
        repeat_output = repeat_dir / f"slot{slot}" / "output.raw"
        sync = run_dir / f"slot{slot}" / "sync.raw"
        repeat_sync = repeat_dir / f"slot{slot}" / "sync.raw"
        output_same = output.read_bytes() == repeat_output.read_bytes()
        sync_same = sync.read_bytes() == repeat_sync.read_bytes()
        extra_same = True
        if slot == 5:
            extra = run_dir / "slot5" / "extra_output.raw"
            repeat_extra = repeat_dir / "slot5" / "extra_output.raw"
            extra_same = extra.read_bytes() == repeat_extra.read_bytes()
        deterministic &= output_same and sync_same and extra_same
        if slot > 3:
            prior_hash = run_slots[slot - 1]["output_sha256"]
            chain_integrity &= current.get("input_sha256") == prior_hash
        slot_reports[str(slot)] = {
            "pass": current.get("pass") is True,
            "grid": current.get("grid"),
            "expected_grid": EXPECTED_GRIDS[slot],
            "observed_releases": current.get("observed_releases"),
            "expected_releases": EXPECTED_RELEASES[slot],
            "output_bytes": output.stat().st_size,
            "output_nonzero_bytes": current.get("output_nonzero_bytes"),
            "output_sha256": sha256(output),
            "repeat_output_sha256": sha256(repeat_output),
            "sync_sha256": sha256(sync),
            "repeat_sync_sha256": sha256(repeat_sync),
            "bitwise_deterministic": output_same and sync_same and extra_same,
        }
    execution_gate = (
        manifest.get("status") == "PASS"
        and repeat_manifest.get("status") == "PASS"
        and manifest.get("device") == "AMD Radeon RX 9070 XT [ZLUDA]"
        and all(
            run_slots.get(slot, {}).get("pass") is True
            and run_slots[slot].get("grid") == EXPECTED_GRIDS[slot]
            and run_slots[slot].get("observed_releases") == EXPECTED_RELEASES[slot]
            and run_slots[slot].get("output_bytes") == TENSOR_BYTES
            and run_slots[slot].get("output_nonzero_bytes", 0) > 0
            for slot in (3, 4, 5)
        )
        and run_slots[5].get("extra_output_bytes") == TENSOR_BYTES
        and run_slots[5].get("extra_output_nonzero_bytes", 0) > 0
    )
    passed = all(capture_checks.values()) and execution_gate and chain_integrity and deterministic
    return {
        "schema": 1,
        "experiment": "amd_swin_slots3_5_dependency_execution",
        "status": "PASS" if passed else "FAIL",
        "classification": "REAL_AMD_NEURAL_DEPENDENCY_EXECUTION_PENDING_RTX_ORACLE",
        "counts_as_s7": False,
        "capture_gate": all(capture_checks.values()),
        "execution_gate": execution_gate,
        "chain_integrity_gate": chain_integrity,
        "determinism_gate": deterministic,
        "capture_checks": capture_checks,
        "slots": slot_reports,
        "next_gate": "return the packaged original-PTX RTX slots 3-5 outputs and compare E4M3 tensors",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repeat-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.capture_dir.resolve(), args.run_dir.resolve(), args.repeat_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
