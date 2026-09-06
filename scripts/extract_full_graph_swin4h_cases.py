#!/usr/bin/env python3
"""Materialize exact RTX before/after states for full-graph slots 10-15."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


TENSOR_BYTES = 192 * 320 * 32
SYNC_BYTES = 27_648 * 4
MAIN_LOGICAL_BYTES = 48 * 80 * 128
EXTRA_LOGICAL_BYTES = 24 * 80 * 64 * 2
EXPECTED_RELEASES = {10: 60, 11: 77, 12: 66, 13: 70, 14: 60, 15: 0}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def fit_state(data: bytes, size: int) -> bytes:
    """Preserve captured bytes and zero-extend short bounded resource views."""
    if len(data) >= size:
        return data[:size]
    return data + bytes(size - len(data))


def write_asset(directory: Path, name: str, data: bytes) -> dict:
    path = directory / name
    path.write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256(data)}


def extract(
    archive_path: Path,
    inventory_path: Path,
    params_root: Path,
    output_dir: Path,
) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    inventory_slots = {int(item["slot"]): item for item in inventory["slots"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []

    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(
            archive.read(members["full_graph_capture.json"]).decode("utf-8-sig")
        )
        windows = {
            (int(item["slot"]), int(item["param_offset"])): item
            for item in capture["windows"]
            if 10 <= int(item["slot"]) <= 15
        }

        def read_window(slot: int, offset: int, phase: str) -> tuple[bytes, dict]:
            window = windows.get((slot, offset))
            if window is None:
                raise ValueError(f"slot {slot} parameter +{offset} has no captured window")
            if window.get("write_ok") is not True:
                raise ValueError(f"slot {slot} parameter +{offset} capture failed")
            blob_name = window[f"{phase}_blob"].replace("\\", "/")
            data = archive.read(members[blob_name])
            expected_hash = window[f"{phase}_sha256"].upper()
            if len(data) != int(window["capture_bytes"]) or sha256(data) != expected_hash:
                raise ValueError(f"slot {slot} parameter +{offset} {phase} blob mismatch")
            return data, window

        for slot in range(10, 16):
            case_dir = output_dir / f"slot{slot}"
            case_dir.mkdir()
            static = inventory_slots[slot]
            params_source = params_root / f"slot{slot}" / "params.raw"
            params = params_source.read_bytes()
            if sha256(params) != static["frame1_param_sha256"].upper():
                raise ValueError(f"slot {slot} parameter hash differs from frame-1 RTX ABI")

            input_before, _ = read_window(slot, 0, "before")
            weights_before, weights_window = read_window(slot, 16, "before")
            output_before, output_window = read_window(slot, 8, "before")
            output_after, _ = read_window(slot, 8, "after")
            assets = {
                "input": write_asset(case_dir, "input.raw", fit_state(input_before, TENSOR_BYTES)),
                "weights": write_asset(case_dir, "weights.raw", weights_before),
                "params": write_asset(case_dir, "params.raw", params),
                "output_initial": write_asset(
                    case_dir, "output_initial.raw", fit_state(output_before, TENSOR_BYTES)
                ),
                "output_reference": write_asset(
                    case_dir, "output_reference.raw", fit_state(output_after, TENSOR_BYTES)
                ),
            }

            wait_window = None
            if slot >= 11:
                wait_before, wait_window = read_window(slot, 48, "before")
                assets["wait_sync_initial"] = write_asset(
                    case_dir, "wait_sync_initial.raw", fit_state(wait_before, SYNC_BYTES)
                )

            release_window = None
            if slot <= 14:
                release_before, release_window = read_window(slot, 64, "before")
                release_after, _ = read_window(slot, 64, "after")
                assets["sync_initial"] = write_asset(
                    case_dir, "sync_initial.raw", fit_state(release_before, SYNC_BYTES)
                )
                assets["sync_reference"] = write_asset(
                    case_dir, "sync_reference.raw", fit_state(release_after, SYNC_BYTES)
                )

            extra_window = None
            if slot == 15:
                extra_before, extra_window = read_window(slot, 72, "before")
                extra_after, _ = read_window(slot, 72, "after")
                assets["extra_output_initial"] = write_asset(
                    case_dir,
                    "extra_output_initial.raw",
                    fit_state(extra_before, TENSOR_BYTES),
                )
                assets["extra_output_reference"] = write_asset(
                    case_dir,
                    "extra_output_reference.raw",
                    fit_state(extra_after, TENSOR_BYTES),
                )

            records.append(
                {
                    "slot": slot,
                    "function": static["function"],
                    "grid": static["grid"],
                    "block": static["block"],
                    "param_size": static["param_size"],
                    "expected_releases": EXPECTED_RELEASES[slot],
                    "main_tensor_type": "fp16",
                    "extra_tensor_type": "e4m3" if slot == 15 else None,
                    "main_logical_bytes": MAIN_LOGICAL_BYTES,
                    "extra_logical_bytes": EXTRA_LOGICAL_BYTES if slot == 15 else 0,
                    "output_native_changed_bytes": int(output_window["changed_bytes"]),
                    "release_native_changed_bytes": (
                        int(release_window["changed_bytes"]) if release_window else 0
                    ),
                    "weight_capture_bytes": int(weights_window["capture_bytes"]),
                    "wait_capture_bytes": (
                        int(wait_window["capture_bytes"]) if wait_window else 0
                    ),
                    "extra_native_changed_bytes": (
                        int(extra_window["changed_bytes"]) if extra_window else 0
                    ),
                    "assets": assets,
                }
            )

    manifest = {
        "schema": 1,
        "experiment": "full_graph_swin4h_slots10_15_exact_state_cases",
        "status": "PASS",
        "classification": "RTX_NATIVE_BEFORE_AFTER_STATE_FOR_AMD_REPLAY",
        "family": "swin4h_slots10_15",
        "next_gate": "lower and replay the 8h/256 successor family beginning at slot 16",
        "source_archive": {
            "path": str(archive_path.resolve()),
            "bytes": archive_path.stat().st_size,
        },
        "slots": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--params-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = extract(
        args.archive.resolve(),
        args.inventory.resolve(),
        args.params_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
