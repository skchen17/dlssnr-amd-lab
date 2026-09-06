#!/usr/bin/env python3
"""Materialize exact RTX states and settled graph-edge oracles for ViT slots 57-98."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def write_asset(directory: Path, name: str, data: bytes) -> dict:
    (directory / name).write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256(data)}


def tag_for(function: str) -> str:
    return {
        "cc_vit_1d_attention_chained_fp8": "attention",
        "cc_vit_1d_ffn_contract_chained_fp8": "ffn_contract",
        "cc_vit_1d_ffn_expand_chained_fp8": "ffn_expand",
        "cc_vit_1d_ffn_expand_publish_fp8": "ffn_expand_publish",
        "cc_vit_1d_projection_chained_fp8": "projection",
        "cc_vit_1d_projection_wait_fp8": "projection_wait",
        "cc_vit_1d_qkv_chained_fp8": "qkv",
        "cc_vit_1d_repack_1d_to_2d_fp8": "repack_1d_to_2d",
        "cc_vit_1d_repack_2d_to_1d_fp8": "repack_2d_to_1d",
    }[function]


def output_specs(slot: int, tag: str) -> list[tuple[int, int, str]]:
    if tag in ("repack_2d_to_1d", "repack_1d_to_2d"):
        return [(8, 98_304, "e4m3")]
    if tag in ("ffn_expand", "ffn_expand_publish"):
        # Both entries convert f16x2 to e4m3x2 before packing/storing. The
        # captured 393216 bytes represent 96x4096 FP8, not 96x2048 FP16.
        return [(16, 393_216, "e4m3")]
    if tag == "ffn_contract":
        return [(16, 98_304, "e4m3"), (40, 196_608, "fp16")]
    if tag == "qkv":
        return [(8, 98_304, "e4m3"), (16, 98_304, "e4m3"),
                (24, 98_304, "e4m3"), (48, 589_824, "fp16")]
    if tag == "attention":
        return [(24, 98_304, "e4m3")]
    if tag in ("projection", "projection_wait"):
        return [(16, 98_304, "e4m3"), (40, 196_608, "fp16")]
    raise ValueError(f"no output specification for slot {slot} tag {tag}")


def extract(archive_path: Path, inventory_path: Path, output_dir: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    inventory_slots = {int(item["slot"]): item for item in inventory["slots"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(archive.read(members["full_graph_capture.json"]).decode("utf-8-sig"))
        all_windows = capture["windows"]
        windows = {
            (int(item["slot"]), int(item["param_offset"])): item
            for item in all_windows
        }

        def checked_blob(window: dict, phase: str) -> bytes:
            name = window[f"{phase}_blob"].replace("\\", "/")
            data = archive.read(members[name])
            if (len(data) != int(window["capture_bytes"]) or
                    sha256(data) != window[f"{phase}_sha256"].upper()):
                raise ValueError(
                    f"slot {window['slot']} +{window['param_offset']} {phase} mismatch")
            return data

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
                        isinstance(slot, int) and 57 <= slot <= 98):
                    params_by_slot[slot] = bytes.fromhex(event["param_hex"])
                    events_by_slot[slot] = event

        activation_resource = windows[(57, 0)]["resource"]
        activation_windows = [
            item for item in all_windows
            if item.get("resource") == activation_resource and int(item["slot"]) <= 99
        ]
        arena_bytes = max(
            int(item["resource_offset"]) + int(item["capture_bytes"])
            for item in activation_windows
        )
        arena = bytearray(arena_bytes)
        by_slot: dict[int, list[dict]] = {}
        for item in activation_windows:
            by_slot.setdefault(int(item["slot"]), []).append(item)
        arena_assets: dict[int, dict] = {}
        for slot in range(0, 99):
            current = by_slot.get(slot, [])
            for item in current:
                start = int(item["resource_offset"])
                data = checked_blob(item, "before")
                arena[start:start + len(data)] = data
            if 57 <= slot <= 98:
                case_dir = output_dir / f"slot{slot}"
                case_dir.mkdir()
                arena_assets[slot] = write_asset(
                    case_dir, "activation_arena_initial.raw", bytes(arena))
            for item in current:
                start = int(item["resource_offset"])
                data = checked_blob(item, "after")
                arena[start:start + len(data)] = data

        model_resource = next(
            item["resource"] for item in all_windows
            if int(item["slot"]) == 58 and int(item["param_offset"]) == 24
        )
        model_windows = [item for item in all_windows if item.get("resource") == model_resource]
        model_bytes = max(
            int(item["resource_offset"]) + int(item["capture_bytes"])
            for item in model_windows
        )
        model_arena = bytearray(model_bytes)
        coverage = []
        for item in model_windows:
            start = int(item["resource_offset"])
            data = checked_blob(item, "before")
            model_arena[start:start + len(data)] = data
            coverage.append((start, start + len(data)))
        coverage.sort()
        end = 0
        for start, stop in coverage:
            if start > end:
                raise ValueError(f"model arena capture gap at {end}:{start}")
            end = max(end, stop)
        if end != model_bytes:
            raise ValueError("model arena capture is incomplete")
        model_asset = write_asset(output_dir, "model_arena.raw", bytes(model_arena))

        records = []
        for slot in range(57, 99):
            static = inventory_slots[slot]
            params = params_by_slot.get(slot)
            event = events_by_slot.get(slot)
            if params is None or len(params) != int(static["param_size"]):
                raise ValueError(f"slot {slot} parameter bytes missing")
            if event.get("grid") != static["grid"] or event.get("block") != static["block"]:
                raise ValueError(f"slot {slot} launch geometry differs")
            case_dir = output_dir / f"slot{slot}"
            slot_windows = sorted(
                (item for item in all_windows if int(item["slot"]) == slot),
                key=lambda item: int(item["param_offset"]),
            )
            activation_views = [
                {"param_offset": int(item["param_offset"]),
                 "arena_offset": int(item["resource_offset"])}
                for item in slot_windows if item["resource"] == activation_resource
            ]
            weight_windows = [item for item in slot_windows if item["resource"] == model_resource]
            tag = tag_for(static["function"])
            if len(weight_windows) != (0 if tag.startswith("repack_") else 1):
                raise ValueError(f"slot {slot} has an unexpected model-pointer count")
            weight_window = weight_windows[0] if weight_windows else None
            outputs = []
            for index, (param_offset, logical_bytes, tensor_type) in enumerate(
                    output_specs(slot, tag)):
                window = windows[(slot, param_offset)]
                arena_offset = int(window["resource_offset"])
                oracle = None
                oracle_source = "immediate_after"
                # A pointer-identical future consumer before snapshot is the
                # dependency-settled producer result.
                future_slot = slot + 1
                candidates = [
                    item for item in all_windows
                    if int(item["slot"]) == future_slot
                    and item["resource"] == activation_resource
                    and int(item["resource_offset"]) == arena_offset
                ]
                if candidates:
                    oracle = checked_blob(candidates[0], "before")[:logical_bytes]
                    oracle_source = f"slot{future_slot}_input_before"
                if oracle is None:
                    oracle = checked_blob(window, "after")[:logical_bytes]
                asset_name = f"output{index}_reference.raw"
                outputs.append({
                    "param_offset": param_offset,
                    "arena_offset": arena_offset,
                    "logical_bytes": logical_bytes,
                    "tensor_type": tensor_type,
                    "oracle_source": oracle_source,
                    "asset": write_asset(case_dir, asset_name, oracle),
                })
            records.append({
                "slot": slot,
                "function": static["function"],
                "ptx_tag": tag,
                "grid": static["grid"],
                "block": static["block"],
                "param_size": static["param_size"],
                "params": write_asset(case_dir, "params.raw", params),
                "activation_arena": arena_assets[slot],
                "activation_arena_bytes": arena_bytes,
                "activation_param_views": activation_views,
                "weight_param_offset": (
                    int(weight_window["param_offset"]) if weight_window else None),
                "weight_view_offset": (
                    int(weight_window["resource_offset"]) if weight_window else 0),
                "outputs": outputs,
            })

    manifest = {
        "schema": 1,
        "experiment": "full_graph_vit1d_slots57_98_exact_state_cases",
        "status": "PASS",
        "classification": "RTX_NATIVE_EXACT_STATE_AND_SETTLED_GRAPH_EDGE_ORACLES",
        "source_archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size},
        "activation_resource": activation_resource,
        "model_resource": model_resource,
        "model_arena": model_asset,
        "slots": records,
        "limitations": [
            "side accumulator outputs without a pointer-identical future consumer use immediate-after diagnostics"
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = extract(args.archive.resolve(), args.inventory.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "slots": len(report["slots"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
