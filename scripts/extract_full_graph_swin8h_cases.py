#!/usr/bin/env python3
"""Materialize exact RTX before/after states for full-graph slots 16-23."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


TENSOR_BYTES = 192 * 320 * 32
SYNC_BYTES = 27_648 * 4
MAIN_LOGICAL_BYTES = 24 * 40 * 128 * 2
EXTRA_LOGICAL_BYTES = 12 * 40 * 256


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def fit_state(data: bytes, size: int) -> bytes:
    return data[:size] if len(data) >= size else data + bytes(size - len(data))


def write_asset(directory: Path, name: str, data: bytes) -> dict:
    (directory / name).write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256(data)}


def extract(archive_path: Path, inventory_path: Path, output_dir: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    inventory_slots = {int(item["slot"]): item for item in inventory["slots"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(archive.read(members["full_graph_capture.json"]).decode("utf-8-sig"))
        windows = {
            (int(item["slot"]), int(item["param_offset"])): item
            for item in capture["windows"] if 16 <= int(item["slot"]) <= 23
        }
        params_by_slot = {}
        events_by_slot = {}
        with archive.open(members["module_trace.jsonl"]) as stream:
            for raw in stream:
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                slot = event.get("slot")
                if (
                    event.get("ev") == "nvapi_launch_cu_kernel"
                    and event.get("frame") == 1
                    and event.get("index") == 0
                    and isinstance(slot, int)
                    and 16 <= slot <= 23
                ):
                    params_by_slot[slot] = bytes.fromhex(event["param_hex"])
                    events_by_slot[slot] = event

        def read_window(slot: int, offset: int, phase: str) -> tuple[bytes, dict]:
            window = windows.get((slot, offset))
            if window is None or window.get("write_ok") is not True:
                raise ValueError(f"slot {slot} parameter +{offset} has no valid capture")
            blob_name = window[f"{phase}_blob"].replace("\\", "/")
            data = archive.read(members[blob_name])
            if len(data) != int(window["capture_bytes"]) or sha256(data) != window[f"{phase}_sha256"].upper():
                raise ValueError(f"slot {slot} parameter +{offset} {phase} blob mismatch")
            return data, window

        for slot in range(16, 24):
            case_dir = output_dir / f"slot{slot}"
            case_dir.mkdir()
            static = inventory_slots[slot]
            params = params_by_slot.get(slot)
            event = events_by_slot.get(slot)
            if params is None or len(params) != int(static["param_size"]):
                raise ValueError(f"slot {slot} frame-1 parameter bytes missing")
            if event.get("grid") != static["grid"] or event.get("block") != static["block"]:
                raise ValueError(f"slot {slot} frame-1 launch geometry differs from inventory")
            input_before, _ = read_window(slot, 0, "before")
            weights_before, weights_window = read_window(slot, 16, "before")
            output_before, output_window = read_window(slot, 8, "before")
            output_after, _ = read_window(slot, 8, "after")
            assets = {
                "input": write_asset(case_dir, "input.raw", fit_state(input_before, TENSOR_BYTES)),
                "weights": write_asset(case_dir, "weights.raw", weights_before),
                "params": write_asset(case_dir, "params.raw", params),
                "output_initial": write_asset(case_dir, "output_initial.raw", fit_state(output_before, TENSOR_BYTES)),
                "output_reference": write_asset(case_dir, "output_reference.raw", fit_state(output_after, TENSOR_BYTES)),
            }
            wait_window = release_window = extra_window = None
            if slot >= 17:
                wait_before, wait_window = read_window(slot, 48, "before")
                assets["wait_sync_initial"] = write_asset(case_dir, "wait_sync_initial.raw", fit_state(wait_before, SYNC_BYTES))
            if slot <= 22:
                release_before, release_window = read_window(slot, 64, "before")
                release_after, _ = read_window(slot, 64, "after")
                assets["sync_initial"] = write_asset(case_dir, "sync_initial.raw", fit_state(release_before, SYNC_BYTES))
                assets["sync_reference"] = write_asset(case_dir, "sync_reference.raw", fit_state(release_after, SYNC_BYTES))
            if slot == 23:
                extra_before, extra_window = read_window(slot, 72, "before")
                extra_after, _ = read_window(slot, 72, "after")
                assets["extra_output_initial"] = write_asset(case_dir, "extra_output_initial.raw", fit_state(extra_before, TENSOR_BYTES))
                assets["extra_output_reference"] = write_asset(case_dir, "extra_output_reference.raw", fit_state(extra_after, TENSOR_BYTES))
            expected_releases = static["grid"][0] * static["grid"][1] if slot <= 22 else 0
            records.append({
                "slot": slot, "function": static["function"], "grid": static["grid"],
                "block": static["block"], "param_size": static["param_size"],
                "current_frame1_param_sha256": sha256(params),
                "inventory_dynamic_address_param_sha256": static["frame1_param_sha256"].upper(),
                "expected_releases": expected_releases, "main_tensor_type": "fp16",
                "extra_tensor_type": "e4m3" if slot == 23 else None,
                "main_logical_bytes": MAIN_LOGICAL_BYTES,
                "extra_logical_bytes": EXTRA_LOGICAL_BYTES if slot == 23 else 0,
                "output_native_changed_bytes": int(output_window["changed_bytes"]),
                "release_native_changed_bytes": int(release_window["changed_bytes"]) if release_window else 0,
                "weight_capture_bytes": int(weights_window["capture_bytes"]),
                "wait_capture_bytes": int(wait_window["capture_bytes"]) if wait_window else 0,
                "extra_native_changed_bytes": int(extra_window["changed_bytes"]) if extra_window else 0,
                "assets": assets,
            })
    manifest = {
        "schema": 1, "experiment": "full_graph_swin8h_slots16_23_exact_state_cases",
        "status": "PASS", "classification": "RTX_NATIVE_BEFORE_AFTER_STATE_FOR_AMD_REPLAY",
        "family": "swin8h_slots16_23",
        "next_gate": "lower and replay the split 16h/512 block at slots 24-56",
        "source_archive": {"path": str(archive_path.resolve()), "bytes": archive_path.stat().st_size},
        "slots": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = extract(args.archive.resolve(), args.inventory.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
