#!/usr/bin/env python3
"""Materialize exact RTX before/after states for split-16h slots 24-56."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


TENSOR_BYTES = 192 * 320 * 32
SYNC_BYTES = 27_648 * 4
MAIN_LOGICAL_BYTES = 122_880
POOL_EXTRA_LOGICAL_BYTES = 49_152
FINAL_HEAD_LOGICAL_BYTES = 98_304


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def fit_state(data: bytes, size: int) -> bytes:
    return data[:size] if len(data) >= size else data + bytes(size - len(data))


def write_asset(directory: Path, name: str, data: bytes) -> dict:
    (directory / name).write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256(data)}


def abi_for(slot: int) -> dict:
    if slot == 24:
        return {"input": 0, "output": 8, "weights": 16, "wait": None,
                "release": 48, "extra": None, "ptx_tag": "ffwd_inpview"}
    if slot == 25:
        return {"input": 0, "input2": 8, "output": 16, "weights": 24,
                "wait": 56, "release": 48, "extra": None,
                "ptx_tag": "ffwd_proj_inpview"}
    if slot == 55:
        return {"input": 0, "input2": 8, "output": 16, "extra": 24,
                "weights": 32, "wait": 40, "release": 48,
                "ptx_tag": "proj_pool"}
    if slot == 56:
        return {"input": 0, "output": 8, "weights": 16, "wait": 24,
                "release": None, "extra": None, "ptx_tag": "final_head"}
    phase = (slot - 26) % 4
    if phase == 0:
        return {"input": 0, "output": 8, "weights": 16, "wait": 40,
                "release": 48, "extra": None, "ptx_tag": "qkv"}
    if phase == 1:
        return {"input": 0, "input2": 8, "output": 16, "weights": 24,
                "wait": 56, "release": 48, "extra": None, "ptx_tag": "proj"}
    if phase == 2:
        return {"input": 0, "output": 8, "weights": 16, "wait": 32,
                "release": 48, "extra": None, "ptx_tag": "ffwd"}
    return {"input": 0, "input2": 8, "output": 16, "weights": 24,
            "wait": 56, "release": 48, "extra": None,
            "ptx_tag": "ffwd_proj"}


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
            for item in capture["windows"] if 24 <= int(item["slot"]) <= 56
        }
        for slot in range(24, 57):
            (output_dir / f"slot{slot}").mkdir()
        params_by_slot: dict[int, bytes] = {}
        events_by_slot: dict[int, dict] = {}
        with archive.open(members["module_trace.jsonl"]) as stream:
            for raw in stream:
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                slot = event.get("slot")
                if (event.get("ev") == "nvapi_launch_cu_kernel" and
                        event.get("frame") == 1 and event.get("index") == 0 and
                        isinstance(slot, int) and 24 <= slot <= 56):
                    params_by_slot[slot] = bytes.fromhex(event["param_hex"])
                    events_by_slot[slot] = event

        def checked_blob(window: dict, phase: str) -> bytes:
            blob_name = window[f"{phase}_blob"].replace("\\", "/")
            data = archive.read(members[blob_name])
            if (len(data) != int(window["capture_bytes"]) or
                    sha256(data) != window[f"{phase}_sha256"].upper()):
                raise ValueError(
                    f"slot {window['slot']} parameter +{window['param_offset']} "
                    f"{phase} blob mismatch")
            return data

        def read_window(slot: int, offset: int, phase: str) -> tuple[bytes, dict]:
            window = windows.get((slot, offset))
            if window is None or window.get("write_ok") is not True:
                raise ValueError(f"slot {slot} parameter +{offset} has no valid capture")
            return checked_blob(window, phase), window

        all_capture_windows = capture["windows"]
        activation_resource = windows[(24, 0)]["resource"]
        activation_windows = [
            item for item in all_capture_windows
            if item.get("resource") == activation_resource and int(item["slot"]) <= 56
        ]
        arena_bytes = max(
            int(item["resource_offset"]) + int(item["capture_bytes"])
            for item in activation_windows
        )
        arena = bytearray(arena_bytes)
        coverage: list[tuple[int, int]] = []

        def add_coverage(start: int, end: int) -> None:
            coverage.append((start, end))
            coverage.sort()
            merged: list[tuple[int, int]] = []
            for left, right in coverage:
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            coverage[:] = merged

        def range_covered(start: int, end: int) -> bool:
            return any(left <= start and end <= right for left, right in coverage)

        activation_arena_assets: dict[int, dict] = {}
        by_slot: dict[int, list[dict]] = {}
        for item in activation_windows:
            by_slot.setdefault(int(item["slot"]), []).append(item)
        for graph_slot in range(0, 57):
            current = by_slot.get(graph_slot, [])
            for item in current:
                start = int(item["resource_offset"])
                data = checked_blob(item, "before")
                arena[start:start + len(data)] = data
                add_coverage(start, start + len(data))
            if 24 <= graph_slot <= 56:
                case_abi = abi_for(graph_slot)
                required_offsets = [
                    int(windows[(graph_slot, case_abi[name])]["resource_offset"])
                    for name in ("input", "output")
                ]
                if "input2" in case_abi:
                    required_offsets.append(
                        int(windows[(graph_slot, case_abi["input2"])]["resource_offset"]))
                if case_abi["extra"] is not None:
                    required_offsets.append(
                        int(windows[(graph_slot, case_abi["extra"])]["resource_offset"]))
                if not all(range_covered(offset, offset + TENSOR_BYTES)
                           for offset in required_offsets):
                    raise ValueError(f"slot {graph_slot} activation arena view lacks coverage")
                case_dir = output_dir / f"slot{graph_slot}"
                activation_arena_assets[graph_slot] = write_asset(
                    case_dir, "activation_arena_initial.raw", bytes(arena))
            for item in current:
                start = int(item["resource_offset"])
                data = checked_blob(item, "after")
                arena[start:start + len(data)] = data

        model_resource = windows[(24, 16)]["resource"]
        model_windows = [
            item for item in all_capture_windows if item.get("resource") == model_resource
        ]
        model_bytes = max(
            int(item["resource_offset"]) + int(item["capture_bytes"])
            for item in model_windows
        )
        model_arena = bytearray(model_bytes)
        model_coverage: list[tuple[int, int]] = []
        for item in model_windows:
            start = int(item["resource_offset"])
            data = checked_blob(item, "before")
            model_arena[start:start + len(data)] = data
            model_coverage.append((start, start + len(data)))
        model_coverage.sort()
        model_end = 0
        for start, end in model_coverage:
            if start > model_end:
                raise ValueError(f"model arena capture gap at {model_end}:{start}")
            model_end = max(model_end, end)
        if model_end != model_bytes:
            raise ValueError("model arena capture does not reach its declared extent")
        model_arena_asset = write_asset(output_dir, "model_arena.raw", bytes(model_arena))

        for slot in range(24, 57):
            static = inventory_slots[slot]
            params = params_by_slot.get(slot)
            event = events_by_slot.get(slot)
            if params is None or len(params) != int(static["param_size"]):
                raise ValueError(f"slot {slot} frame-1 parameter bytes missing")
            if event.get("grid") != static["grid"] or event.get("block") != static["block"]:
                raise ValueError(f"slot {slot} frame-1 launch geometry differs from inventory")
            abi = abi_for(slot)
            case_dir = output_dir / f"slot{slot}"
            input_before, input_window = read_window(slot, abi["input"], "before")
            weights_before, weights_window = read_window(slot, abi["weights"], "before")
            output_before, output_window = read_window(slot, abi["output"], "before")
            output_after, _ = read_window(slot, abi["output"], "after")
            assets = {
                "activation_arena_initial": activation_arena_assets[slot],
                "input": write_asset(case_dir, "input.raw", fit_state(input_before, TENSOR_BYTES)),
                "params": write_asset(case_dir, "params.raw", params),
                "output_initial": write_asset(case_dir, "output_initial.raw", fit_state(output_before, TENSOR_BYTES)),
                "output_reference": write_asset(case_dir, "output_reference.raw", fit_state(output_after, TENSOR_BYTES)),
            }
            main_oracle = "immediate_after"
            # Launch capture is intentionally non-blocking.  For chained graph
            # edges, the next consumer's before snapshot is taken after its wait
            # dependency and is therefore the settled producer result.  The
            # producer's immediate after snapshot can still contain an old or
            # partially written buffer.
            if slot < 55:
                next_abi = abi_for(slot + 1)
                next_input, next_input_window = read_window(
                    slot + 1, next_abi["input"], "before")
                if (next_input_window["resource"] != output_window["resource"] or
                        int(next_input_window["resource_offset"]) !=
                        int(output_window["resource_offset"])):
                    raise ValueError(
                        f"slot {slot} output is not the aliased input of slot {slot + 1}")
                assets["settled_output_reference"] = write_asset(
                    case_dir, "settled_output_reference.raw",
                    fit_state(next_input, TENSOR_BYTES))
                main_oracle = "downstream_input_before"
            input2_window = wait_window = release_window = extra_window = None
            if "input2" in abi:
                input2_before, input2_window = read_window(slot, abi["input2"], "before")
                assets["input2"] = write_asset(case_dir, "input2.raw", fit_state(input2_before, TENSOR_BYTES))
            if abi["wait"] is not None:
                wait_before, wait_window = read_window(slot, abi["wait"], "before")
                assets["wait_sync_initial"] = write_asset(
                    case_dir, "wait_sync_initial.raw", fit_state(wait_before, SYNC_BYTES))
            if abi["release"] is not None:
                release_before, release_window = read_window(slot, abi["release"], "before")
                release_after, _ = read_window(slot, abi["release"], "after")
                assets["sync_initial"] = write_asset(
                    case_dir, "sync_initial.raw", fit_state(release_before, SYNC_BYTES))
                assets["sync_reference"] = write_asset(
                    case_dir, "sync_reference.raw", fit_state(release_after, SYNC_BYTES))
            if abi["extra"] is not None:
                extra_before, extra_window = read_window(slot, abi["extra"], "before")
                extra_after, _ = read_window(slot, abi["extra"], "after")
                assets["extra_output_initial"] = write_asset(
                    case_dir, "extra_output_initial.raw", fit_state(extra_before, TENSOR_BYTES))
                assets["extra_output_reference"] = write_asset(
                    case_dir, "extra_output_reference.raw", fit_state(extra_after, TENSOR_BYTES))
                next_abi = abi_for(slot + 1)
                next_input, next_input_window = read_window(
                    slot + 1, next_abi["input"], "before")
                if (next_input_window["resource"] != extra_window["resource"] or
                        int(next_input_window["resource_offset"]) !=
                        int(extra_window["resource_offset"])):
                    raise ValueError(
                        f"slot {slot} extra output is not the aliased input of slot {slot + 1}")
                assets["settled_extra_output_reference"] = write_asset(
                    case_dir, "settled_extra_output_reference.raw",
                    fit_state(next_input, TENSOR_BYTES))
            expected_releases = (static["grid"][0] * static["grid"][1] * static["grid"][2]
                                 if abi["release"] is not None else 0)
            main_type = "e4m3" if slot == 56 else "fp16"
            main_bytes = FINAL_HEAD_LOGICAL_BYTES if slot == 56 else MAIN_LOGICAL_BYTES
            arena_views = {
                "input": int(input_window["resource_offset"]),
                "output": int(output_window["resource_offset"]),
                "input2": int(input2_window["resource_offset"]) if input2_window else None,
                "extra": int(extra_window["resource_offset"]) if extra_window else None,
            }
            for name, window in (("input", input_window), ("output", output_window),
                                 ("input2", input2_window), ("extra", extra_window)):
                if window is not None and window["resource"] != activation_resource:
                    raise ValueError(f"slot {slot} {name} is outside the activation arena")
            if weights_window["resource"] != model_resource:
                raise ValueError(f"slot {slot} weights are outside the model arena")
            records.append({
                "slot": slot, "function": static["function"], "grid": static["grid"],
                "block": static["block"], "param_size": static["param_size"],
                "current_frame1_param_sha256": sha256(params), "abi": abi,
                "activation_arena_resource": activation_resource,
                "activation_arena_bytes": arena_bytes,
                "arena_views": arena_views,
                "weight_view_offset": int(weights_window["resource_offset"]),
                "expected_releases": expected_releases, "main_tensor_type": main_type,
                "main_oracle": main_oracle,
                "extra_oracle": ("downstream_input_before"
                                 if abi["extra"] is not None else None),
                "extra_tensor_type": "e4m3" if slot == 55 else None,
                "main_logical_bytes": main_bytes,
                "extra_logical_bytes": POOL_EXTRA_LOGICAL_BYTES if slot == 55 else 0,
                "input_capture_bytes": int(input_window["capture_bytes"]),
                "input2_capture_bytes": int(input2_window["capture_bytes"]) if input2_window else 0,
                "output_native_changed_bytes": int(output_window["changed_bytes"]),
                "output_capture_bytes": int(output_window["capture_bytes"]),
                "release_native_changed_bytes": int(release_window["changed_bytes"]) if release_window else 0,
                "weight_capture_bytes": int(weights_window["capture_bytes"]),
                "wait_capture_bytes": int(wait_window["capture_bytes"]) if wait_window else 0,
                "extra_native_changed_bytes": int(extra_window["changed_bytes"]) if extra_window else 0,
                "assets": assets,
            })
    manifest = {
        "schema": 1, "experiment": "full_graph_split16h_slots24_56_exact_state_cases",
        "status": "PASS", "classification": "RTX_NATIVE_BEFORE_AFTER_STATE_FOR_AMD_REPLAY",
        "family": "split16h_slots24_56", "next_gate": "replay ViT slots 57-104",
        "source_archive": {"path": str(archive_path.resolve()), "bytes": archive_path.stat().st_size},
        "model_arena": {**model_arena_asset, "resource": model_resource},
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
