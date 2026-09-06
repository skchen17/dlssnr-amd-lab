#!/usr/bin/env python3
"""Build a literal 156-launch, single-state RX replay plan from the RTX capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile


DECODER_TAGS = {
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

SPLIT_TAGS = {
    "cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8": "ffwd_inpview",
    "cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8": "ffwd_proj_inpview",
    "cc_split_swin_16h_qkv_512_chained_fp8": "qkv",
    "cc_split_swin_16h_proj_512_chained_fp8": "proj",
    "cc_split_swin_16h_ffwd_512_chained_fp8": "ffwd",
    "cc_split_swin_16h_ffwd_proj_512_chained_fp8": "ffwd_proj",
    "cc_split_swin_16h_proj_pool_512_chained_fp8": "proj_pool",
    "cc_split_swin_16h_final_head_512_wait_fp8": "final_head",
}

VIT_TAGS = {
    "cc_vit_1d_repack_2d_to_1d_fp8": "repack_2d_to_1d",
    "cc_vit_1d_ffn_expand_publish_fp8": "ffn_expand_publish",
    "cc_vit_1d_ffn_contract_chained_fp8": "ffn_contract",
    "cc_vit_1d_qkv_chained_fp8": "qkv",
    "cc_vit_1d_attention_chained_fp8": "attention",
    "cc_vit_1d_projection_chained_fp8": "projection",
    "cc_vit_1d_ffn_expand_chained_fp8": "ffn_expand",
    "cc_vit_1d_projection_wait_fp8": "projection_wait",
    "cc_vit_1d_repack_1d_to_2d_fp8": "repack_1d_to_2d",
}

# The logical tensor on each selected family boundary.  The reference is the
# next launch's captured input, so asynchronous RTX writes have settled.
BOUNDARIES = {
    0: (0, 110_592, "sync_u32"),
    1: (248, 1_966_080, "e4m3"),
    2: (8, 1_966_080, "e4m3"),
    5: (64, 983_040, "e4m3"),
    9: (72, 491_520, "e4m3"),
    15: (72, 245_760, "e4m3"),
    23: (72, 122_880, "e4m3"),
    56: (8, 98_304, "e4m3"),
    98: (8, 98_304, "e4m3"),
    154: (16, 1_843_200, "rgba16f"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def asset(path: Path, base: Path) -> dict:
    return {
        "path": str(path.relative_to(base)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ptx_for_slot(repo: Path, slot: int, function: str) -> Path:
    results = repo / "results"
    if slot == 0:
        return results / "20260831_175500_full_graph_ptx_access" / "cc_cb_clear.ptx"
    if slot == 1:
        return results / "20260831_140000_n0_full_numeric_corrected" / "neural_full_numeric_corrected.ptx"
    if slot == 2:
        return results / "20260831_142500_n1_slot2_lowering" / "n1_full_lowered.ptx"
    if 3 <= slot <= 5:
        tag = "ds_wait" if "ds_wait" in function else "chained"
        return results / "20260831_144500_swin1h_slots3_5_lowering" / f"{tag}_full.ptx"
    if slot == 6:
        return results / "20260831_155000_swin2h_slot6_lowering" / "slot6_compact_full.ptx"
    if 7 <= slot <= 9:
        tag = "ds_wait" if "ds_wait" in function else "chained"
        return results / "20260831_163500_swin2h_slots7_9_lowering" / f"{tag}_full.ptx"
    if 10 <= slot <= 15:
        tag = "inpview" if "inpview" in function else ("ds_wait" if "ds_wait" in function else "chained")
        return results / "20260831_172500_swin4h_slots10_15_lowering" / f"{tag}_full.ptx"
    if 16 <= slot <= 23:
        tag = "inpview" if "inpview" in function else ("ds_wait" if "ds_wait" in function else "chained")
        return results / "20260831_192000_swin8h_slots16_23_lowering" / f"{tag}_full.ptx"
    if 24 <= slot <= 56:
        return results / "20260831_202000_split16h_slots24_56_lowering_v4" / f"{SPLIT_TAGS[function]}_full.ptx"
    if 57 <= slot <= 98:
        return results / "20260831_215000_vit1d_slots57_98_lowering_v2" / f"{VIT_TAGS[function]}_full.ptx"
    if 99 <= slot <= 154:
        return results / "20260831_232000_decoder_slots99_154_lowering_v4" / f"{DECODER_TAGS[function]}_full.ptx"
    if slot == 155:
        return repo / "tools" / "integrated_graph_replay" / "cg2r_copy_linear.ptx"
    raise ValueError(f"unsupported slot {slot}")


def extract(archive_path: Path, inventory_path: Path, model_path: Path,
            repo: Path, output_dir: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    static_slots = {int(item["slot"]): item for item in inventory["slots"]}
    if sorted(static_slots) != list(range(156)):
        raise ValueError("inventory is not a contiguous 156-slot graph")
    output_dir.mkdir(parents=True, exist_ok=False)
    params_dir = output_dir / "params"
    refs_dir = output_dir / "references"
    params_dir.mkdir()
    refs_dir.mkdir()

    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(archive.read(members["full_graph_capture.json"]).decode("utf-8-sig"))
        windows = capture["windows"]
        by_key = {(int(item["slot"]), int(item["param_offset"])): item for item in windows}

        def checked_blob(window: dict, phase: str) -> bytes:
            name = window[f"{phase}_blob"].replace("\\", "/")
            data = archive.read(members[name])
            if len(data) != int(window["capture_bytes"]) or sha256(data) != window[f"{phase}_sha256"].upper():
                raise ValueError(f"slot {window['slot']} +{window['param_offset']} {phase} mismatch")
            return data

        params_by_slot: dict[int, bytes] = {}
        with archive.open(members["module_trace.jsonl"]) as stream:
            for raw in stream:
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                slot = event.get("slot")
                if (event.get("ev") == "nvapi_launch_cu_kernel" and event.get("frame") == 1
                        and event.get("index") == 0 and isinstance(slot, int) and 0 <= slot <= 155):
                    params_by_slot[slot] = bytes.fromhex(event["param_hex"])

        activation_resource = by_key[(0, 0)]["resource"]
        model_resource = by_key[(1, 224)]["resource"]
        activation_windows = [item for item in windows if item["resource"] == activation_resource]
        activation_end = max(int(item["resource_offset"]) + int(item["capture_bytes"])
                             for item in activation_windows)
        surface_offset = (activation_end + 255) & ~255
        surface_padded_bytes = 640 * 384 * 8
        arena_bytes = surface_offset + surface_padded_bytes

        # Only the actual frame-start snapshot is seeded.  No future RTX
        # intermediate is copied into the integrated arena.
        slot0_window = by_key[(0, 0)]
        slot0_before = checked_blob(slot0_window, "before")
        arena_initial = bytearray(arena_bytes)
        start0 = int(slot0_window["resource_offset"])
        arena_initial[start0:start0 + len(slot0_before)] = slot0_before
        arena_path = output_dir / "activation_arena_frame_start.raw"
        arena_path.write_bytes(arena_initial)

        model_copy = output_dir / "model_arena.raw"
        shutil.copyfile(model_path, model_copy)

        final_reference = archive.read(members["copy_input.raw"])
        final_copy_reference = archive.read(members["copy_output.raw"])
        if len(final_reference) != 1_843_200 or final_reference != final_copy_reference:
            raise ValueError("captured slot-155 copy is not a lossless 640x360 RGBA16F copy")
        (refs_dir / "slot154_rgba16f.raw").write_bytes(final_reference)
        (refs_dir / "slot155_rgba16f.raw").write_bytes(final_copy_reference)

        records = []
        checkpoints = {}
        for slot in range(156):
            static = static_slots[slot]
            params = params_by_slot.get(slot)
            if params is None or len(params) != int(static["param_size"]):
                raise ValueError(f"slot {slot} parameter block is missing or truncated")
            param_path = params_dir / f"slot{slot}.raw"
            param_path.write_bytes(params)
            slot_windows = sorted((item for item in windows if int(item["slot"]) == slot),
                                  key=lambda item: int(item["param_offset"]))
            activation_views = [
                {"param_offset": int(item["param_offset"]), "arena_offset": int(item["resource_offset"])}
                for item in slot_windows if item["resource"] == activation_resource
            ]
            weight_views = [
                {"param_offset": int(item["param_offset"]), "weight_offset": int(item["resource_offset"])}
                for item in slot_windows if item["resource"] == model_resource
            ]
            special = "normal"
            if slot == 1:
                special = "zero_rgba16f_texture"
            elif slot == 154:
                special = "post_linear_surface"
                activation_views.append({"param_offset": 16, "arena_offset": surface_offset})
            elif slot == 155:
                special = "linear_final_copy"
                activation_views.append({"param_offset": 0, "arena_offset": surface_offset})

            ptx = ptx_for_slot(repo, slot, static["function"]).resolve()
            if not ptx.is_file():
                raise FileNotFoundError(ptx)
            text = ptx.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"\.entry\s+" + re.escape(static["function"]) + r"\s*\(", text):
                raise ValueError(f"slot {slot} entry absent from {ptx}")

            checkpoint = None
            if slot in BOUNDARIES:
                param_offset, logical_bytes, tensor_type = BOUNDARIES[slot]
                if slot == 154:
                    arena_offset = surface_offset
                    reference = final_reference
                    source = "slot155_copy_input"
                elif slot == 0:
                    window = by_key[(slot, param_offset)]
                    arena_offset = int(window["resource_offset"])
                    reference = checked_blob(window, "after")[:logical_bytes]
                    source = "slot0_immediate_after"
                else:
                    window = by_key[(slot, param_offset)]
                    arena_offset = int(window["resource_offset"])
                    consumers = [item for item in windows if int(item["slot"]) == slot + 1
                                 and item["resource"] == activation_resource
                                 and int(item["resource_offset"]) == arena_offset]
                    if not consumers:
                        raise ValueError(f"slot {slot} boundary has no adjacent consumer")
                    reference = checked_blob(consumers[0], "before")[:logical_bytes]
                    source = f"slot{slot + 1}_input_before"
                ref_path = refs_dir / f"slot{slot}.raw"
                ref_path.write_bytes(reference)
                checkpoint = {
                    "arena_offset": arena_offset,
                    "logical_bytes": logical_bytes,
                    "tensor_type": tensor_type,
                    "reference": asset(ref_path, output_dir),
                    "oracle_source": source,
                }
                checkpoints[str(slot)] = checkpoint

            records.append({
                "slot": slot,
                "function": static["function"],
                "ptx": str(ptx),
                "ptx_sha256": sha256_file(ptx),
                "grid": static["grid"],
                "block": static["block"],
                "dynamic_shared": int(static.get("dynamic_shared", 0)),
                "param_size": int(static["param_size"]),
                "params": asset(param_path, output_dir),
                "activation_param_views": activation_views,
                "weight_param_views": weight_views,
                "special": special,
                "checkpoint": checkpoint,
            })

    plan = {
        "schema": 1,
        "experiment": "rx9070xt_full_graph_single_context_156_launch_plan",
        "status": "PASS",
        "classification": "RTX_CAPTURE_DERIVED_LITERAL_INTEGRATED_REPLAY_PLAN",
        "source_archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size,
                           "sha256": sha256_file(archive_path)},
        "activation_resource": activation_resource,
        "model_resource": model_resource,
        "activation_arena_bytes": arena_bytes,
        "surface_arena_offset": surface_offset,
        "surface_padded_bytes": surface_padded_bytes,
        "activation_arena_initial": asset(arena_path, output_dir),
        "model_arena": asset(model_copy, output_dir),
        "final_output_bytes": 1_843_200,
        "checkpoints": checkpoints,
        "slots": records,
        "integrity_gates": {
            "slot_count_156": len(records) == 156,
            "no_future_rtx_activations_seeded": True,
            "slot1_captured_null_srv_modeled_as_zero_texture": True,
            "slot154_formatted_surface_is_explicit_linear_rgba16f": True,
            "slot155_is_real_gpu_linear_rgba16f_copy": True,
        },
        "limitations": [
            "This is a replay plan; success requires the separate RX execution and numerical manifests.",
            "The captured D3D12 null SRV is modeled by a valid CUDA texture object containing zeros.",
        ],
    }
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--model-arena", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = extract(args.archive.resolve(), args.inventory.resolve(), args.model_arena.resolve(),
                     args.repo.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "slots": len(report["slots"]),
                      "activation_arena_bytes": report["activation_arena_bytes"],
                      "checkpoints": len(report["checkpoints"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
