#!/usr/bin/env python3
"""Materialize exact RTX states and settled oracles for decoder slots 99-154."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


FUNCTION_TAGS = {
    "cc_dec_input_upsample_1024_512_tilesync_fp8": "dec_input",
    "cc_split_swin_16h_ffwd_512_chained_fp8": "split_ffwd",
    "cc_split_swin_16h_ffwd_proj_512_chained_fp8": "split_ffwd_proj",
    "cc_split_swin_16h_qkv_512_chained_fp8": "split_qkv",
    "cc_split_swin_16h_proj_512_chained_fp8": "split_proj",
    "cc_split_swin_16h_proj_512_outview_wait_fp8": "split_proj_outview",
    "cc_tinlayout_fused_swin_8h_256_8_upsample_tilesync_fp8": "swin8_upsample",
    "cc_tinlayout_fused_swin_8h_256_8_chained_fp8": "swin8_chained",
    "cc_tinlayout_fused_swin_8h_256_8_outview_wait_fp8": "swin8_outview",
    "cc_tinlayout_fused_swin_4h_128_4_upsample_tilesync_fp8": "swin4_upsample",
    "cc_tinlayout_fused_swin_4h_128_4_chained_fp8": "swin4_chained",
    "cc_tinlayout_fused_swin_4h_128_4_outview_wait_fp8": "swin4_outview",
    "cc_tinlayout_fused_swin_2h_64_2_upsample_tilesync_fp8": "swin2_upsample",
    "cc_tinlayout_fused_swin_2h_64_2_chained_fp8": "swin2_chained",
    "cc_tinlayout_fused_swin_2h_64_2_outview_wait_fp8": "swin2_outview",
    "cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8": "swin1_upsample",
    "cc_tinlayout_fused_swin_1h_32_1_chained_fp8": "swin1_chained",
    "cc_tinlayout_fused_swin_1h_32_1_outview_wait_fp8": "swin1_outview",
    "cc_tinlayout_fused_post_block_swin_1h_32_fp8": "post_block",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_asset(directory: Path, name: str, data: bytes) -> dict:
    (directory / name).write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256(data)}


def output_spec(tag: str) -> tuple[int, int, str]:
    if tag == "dec_input":
        return 16, 122_880, "e4m3"
    if tag in ("split_ffwd", "split_qkv"):
        return 8, 122_880, "e4m3"
    if tag in ("split_ffwd_proj", "split_proj", "split_proj_outview"):
        return 16, 122_880, "e4m3"
    if tag.startswith("swin8_"):
        return 8, 245_760, "e4m3"
    if tag.startswith("swin4_"):
        return 8, 491_520, "e4m3"
    if tag.startswith("swin2_"):
        return 8, 983_040, "e4m3"
    if tag.startswith("swin1_"):
        return 8, 1_966_080, "e4m3"
    if tag == "post_block":
        return 16, 1_843_200, "rgba16f"
    raise ValueError(f"no output specification for {tag}")


def extract(archive_path: Path, inventory_path: Path, output_dir: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    inventory_slots = {int(item["slot"]): item for item in inventory["slots"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(archive.read(members["full_graph_capture.json"]).decode("utf-8-sig"))
        all_windows = capture["windows"]
        windows = {(int(item["slot"]), int(item["param_offset"])): item
                   for item in all_windows}

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
                        isinstance(slot, int) and 99 <= slot <= 154):
                    params_by_slot[slot] = bytes.fromhex(event["param_hex"])
                    events_by_slot[slot] = event

        activation_resource = windows[(99, 0)]["resource"]
        activation_windows = [item for item in all_windows
                              if item.get("resource") == activation_resource
                              and int(item["slot"]) <= 154]
        activation_end = max(int(item["resource_offset"]) + int(item["capture_bytes"])
                             for item in activation_windows)
        surface_arena_offset = (activation_end + 255) & ~255
        surface_padded_bytes = 640 * 384 * 8
        arena_bytes = surface_arena_offset + surface_padded_bytes
        arena = bytearray(arena_bytes)
        by_slot: dict[int, list[dict]] = {}
        for item in activation_windows:
            by_slot.setdefault(int(item["slot"]), []).append(item)
        arena_assets: dict[int, dict] = {}
        for slot in range(0, 155):
            current = by_slot.get(slot, [])
            for item in current:
                start = int(item["resource_offset"])
                data = checked_blob(item, "before")
                arena[start:start + len(data)] = data
            if 99 <= slot <= 154:
                case_dir = output_dir / f"slot{slot}"
                case_dir.mkdir()
                arena_assets[slot] = write_asset(
                    case_dir, "activation_arena_initial.raw", bytes(arena))
            for item in current:
                start = int(item["resource_offset"])
                data = checked_blob(item, "after")
                arena[start:start + len(data)] = data

        model_resource = windows[(99, 56)]["resource"]
        model_windows = [item for item in all_windows
                         if item.get("resource") == model_resource]
        model_bytes = max(int(item["resource_offset"]) + int(item["capture_bytes"])
                          for item in model_windows)
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

        final_reference = archive.read(members["copy_input.raw"])
        if len(final_reference) != 1_843_200:
            raise ValueError("slot 155 final-copy input size differs")
        if final_reference != archive.read(members["copy_output.raw"]):
            raise ValueError("slot 155 final-copy input/output differ")

        records = []
        for slot in range(99, 155):
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
                key=lambda item: int(item["param_offset"]))
            activation_views = [
                {"param_offset": int(item["param_offset"]),
                 "arena_offset": int(item["resource_offset"])}
                for item in slot_windows if item["resource"] == activation_resource]
            weight_views = [
                {"param_offset": int(item["param_offset"]),
                 "weight_offset": int(item["resource_offset"])}
                for item in slot_windows if item["resource"] == model_resource]
            tag = FUNCTION_TAGS[static["function"]]
            param_offset, logical_bytes, tensor_type = output_spec(tag)
            if tag == "post_block":
                arena_offset = surface_arena_offset
                reference = final_reference
                oracle_source = "slot155_copy_input"
            else:
                window = windows[(slot, param_offset)]
                arena_offset = int(window["resource_offset"])
                candidates = [item for item in all_windows
                              if int(item["slot"]) == slot + 1
                              and item["resource"] == activation_resource
                              and int(item["resource_offset"]) == arena_offset]
                if not candidates:
                    raise ValueError(f"slot {slot} main output lacks adjacent consumer")
                reference = checked_blob(candidates[0], "before")[:logical_bytes]
                if len(reference) != logical_bytes:
                    raise ValueError(f"slot {slot} settled oracle is truncated")
                oracle_source = f"slot{slot + 1}_input_before"
            output_asset = write_asset(case_dir, "output0_reference.raw", reference)
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
                "weight_param_views": weight_views,
                "outputs": [{
                    "param_offset": param_offset,
                    "arena_offset": arena_offset,
                    "logical_bytes": logical_bytes,
                    "tensor_type": tensor_type,
                    "oracle_source": oracle_source,
                    "asset": output_asset,
                }],
            })

    manifest = {
        "schema": 1,
        "experiment": "full_graph_decoder_slots99_154_exact_state_cases",
        "status": "PASS",
        "classification": "RTX_NATIVE_EXACT_STATE_AND_SETTLED_GRAPH_EDGE_ORACLES",
        "source_archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size,
                           "sha256": sha256_file(archive_path)},
        "activation_resource": activation_resource,
        "model_resource": model_resource,
        "activation_arena_bytes": arena_bytes,
        "surface_arena_offset": surface_arena_offset,
        "surface_padded_bytes": surface_padded_bytes,
        "model_arena": model_asset,
        "slots": records,
        "limitations": [
            "function-isolated states are exact RTX snapshots; integrated scheduling is a later gate",
            "formatted CUDA surface output is represented by an explicit linear RGBA16F ABI",
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
    print(json.dumps({"status": report["status"], "slots": len(report["slots"]),
                      "activation_arena_bytes": report["activation_arena_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
