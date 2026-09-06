#!/usr/bin/env python3
"""Build a checked, machine-readable contract for the captured DLSS-NR output head.

This does not execute NVIDIA code.  It joins the accepted launch plan, the
locally decoded model resource, the same-frame RTX oracle case, and static PTX
instruction evidence so that an AMD-native implementation has one stable ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any


FUNCTION = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
HEAD_LAYER = "block70.layer0.layer"
BLEND_SCALE = "block70.layer0.blend_scale"
FP8_MMA = "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"
FP16_MMA = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_slot(document: dict[str, Any], slot: int) -> dict[str, Any]:
    matches = [entry for entry in document["slots"] if entry.get("slot") == slot]
    if len(matches) != 1:
        raise ValueError(f"expected one slot {slot}, found {len(matches)}")
    return matches[0]


def find_tensor(document: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [entry for entry in document["tensors"] if entry.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor {name}, found {len(matches)}")
    return matches[0]


def decode_params(raw: bytes) -> dict[str, Any]:
    if len(raw) != 184:
        raise ValueError(f"expected 184 parameter bytes, got {len(raw)}")
    return {
        "padded_feature_extent": list(struct.unpack_from("<2i", raw, 32)),
        "feature_origin": list(struct.unpack_from("<2i", raw, 40)),
        "output_scale": struct.unpack_from("<f", raw, 48)[0],
        "output_mode": struct.unpack_from("<i", raw, 52)[0],
        "input_extent_f32": list(struct.unpack_from("<2f", raw, 72)),
        "inverse_input_extent_f32": list(struct.unpack_from("<2f", raw, 80)),
        "texture_handle_at_88": struct.unpack_from("<Q", raw, 88)[0],
        "texture_handle_at_96": struct.unpack_from("<Q", raw, 96)[0],
        "output_extent": list(struct.unpack_from("<2i", raw, 172)),
    }


def analyze(
    plan_path: Path,
    tensor_manifest_path: Path,
    model_arena_path: Path,
    case_manifest_path: Path,
    ptx_path: Path,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    tensor_manifest = load_json(tensor_manifest_path)
    case_manifest = load_json(case_manifest_path)
    launch = find_slot(plan, 154)
    case = find_slot(case_manifest, 154)
    head = find_tensor(tensor_manifest, HEAD_LAYER)
    blend = find_tensor(tensor_manifest, BLEND_SCALE)

    case_dir = case_manifest_path.parent / "slot154"
    params_path = case_dir / case["params"]["path"]
    activation_path = case_dir / case["activation_arena"]["path"]
    output_path = case_dir / case["outputs"][0]["asset"]["path"]
    params_raw = params_path.read_bytes()
    decoded = decode_params(params_raw)
    ptx = ptx_path.read_text(encoding="utf-8")

    arena_size = model_arena_path.stat().st_size
    with model_arena_path.open("rb") as stream:
        stream.seek(head["arena_offset"])
        head_raw = stream.read(head["data_bytes"])
        stream.seek(blend["arena_offset"])
        blend_raw = stream.read(blend["data_bytes"])
    if len(head_raw) != head["data_bytes"] or len(blend_raw) != blend["data_bytes"]:
        raise ValueError("model arena does not contain the complete output-head tensors")

    fp8_count = ptx.count(FP8_MMA)
    fp16_count = ptx.count(FP16_MMA)
    surface_store_count = len(re.findall(r"\bsust\.p\.", ptx))
    texture_instruction_count = len(re.findall(r"\btex\.", ptx))

    activation_views = {v["param_offset"]: v["arena_offset"] for v in launch["activation_param_views"]}
    weight_views = {v["param_offset"]: v["weight_offset"] for v in launch["weight_param_views"]}
    output = case["outputs"][0]
    output_width, output_height = decoded["output_extent"]

    validations = {
        "function": launch["function"] == case["function"] == FUNCTION,
        "launch_shape": launch["grid"] == [81, 49, 1] and launch["block"] == [32, 1, 1],
        "parameter_size": launch["param_size"] == case["param_size"] == len(params_raw) == 184,
        "parameter_hash": (
            sha256_bytes(params_raw) == launch["params"]["sha256"] == case["params"]["sha256"]
        ),
        "activation_bindings": activation_views.get(0) == 13873152 and activation_views.get(8) == 110592,
        "output_binding": output["arena_offset"] == activation_views.get(16) == 27807744,
        "weight_bindings": weight_views.get(24) == head["arena_offset"] and weight_views.get(104) == blend["arena_offset"],
        "head_tensor_hash": sha256_bytes(head_raw) == head["sha256"],
        "blend_tensor_hash": sha256_bytes(blend_raw) == blend["sha256"],
        "model_arena": arena_size == tensor_manifest["model_arena"]["bytes"] and sha256_file(model_arena_path) == tensor_manifest["model_arena"]["sha256"],
        "captured_inputs": activation_path.stat().st_size == case["activation_arena"]["bytes"] and sha256_file(activation_path) == case["activation_arena"]["sha256"],
        "captured_output": output_path.stat().st_size == output["logical_bytes"] and sha256_file(output_path) == output["asset"]["sha256"],
        "output_shape": [output_width, output_height] == [640, 360] and output["logical_bytes"] == output_width * output_height * 4 * 2,
        "mma_inventory": fp8_count == 256 and fp16_count == 16,
        "surface_store_inventory": surface_store_count == 2,
    }
    passed = all(validations.values())
    blend_value = struct.unpack("<e", blend_raw)[0]

    return {
        "schema": 1,
        "experiment": "output_head_operator_contract",
        "status": "PASS" if passed else "FAIL",
        "classification": "STATIC_AND_CAPTURED_OPERATOR_CONTRACT",
        "native_execution_status": "PENDING",
        "function": FUNCTION,
        "launch": {
            "grid": launch["grid"],
            "block": launch["block"],
            "parameter_bytes": launch["param_size"],
        },
        "bindings": {
            "main_feature": {"param_offset": 0, "arena_offset": activation_views.get(0)},
            "skip_feature": {"param_offset": 8, "arena_offset": activation_views.get(8)},
            "output": {"param_offset": 16, "arena_offset": output["arena_offset"]},
            "head_layer": {"param_offset": 24, "arena_offset": head["arena_offset"], "tensor": HEAD_LAYER},
            "blend_scale": {"param_offset": 104, "arena_offset": blend["arena_offset"], "tensor": BLEND_SCALE},
        },
        "tensor_contracts": {
            "main_feature": {
                "shape": [2, 192, 320, 16],
                "storage": "e4m3fn_bytes",
                "logical_bytes": 2 * 192 * 320 * 16,
                "confidence": "rtx-trace-and-native-fusion-verified",
            },
            "skip_feature": {
                "shape": [96, 160, 512],
                "storage": "e4m3fn_mma_fragment_tiles",
                "logical_bytes": 96 * 160 * 512,
                "confidence": "captured-address-and-access-pattern",
            },
            "output": {
                "shape": [360, 640, 4],
                "storage": "rgba16f",
                "logical_bytes": output["logical_bytes"],
            },
        },
        "parameters": decoded,
        "weights": {
            "blend_scale_fp16": blend_value,
            "head_layer_bytes": head["data_bytes"],
            "head_layer_sha256": head["sha256"],
            "segments": [
                {"offset": 0, "bytes": 8192, "candidate": "skip_projection_512_to_16_e4m3"},
                {"offset": 8192, "bytes": 16, "candidate": "zero_or_alignment"},
                {"offset": 8208, "bytes": 192, "candidate": "three_fp16_channel_parameter_banks"},
                {"offset": 8400, "bytes": 3072, "candidate": "e4m3_projection_stage_a"},
                {"offset": 11472, "bytes": 4096, "candidate": "fp16_accumulator_bank_a"},
                {"offset": 15568, "bytes": 4096, "candidate": "fp16_accumulator_bank_b"},
                {"offset": 19664, "bytes": 16, "candidate": "constants_or_alignment"},
                {"offset": 19680, "bytes": 1024, "candidate": "e4m3_projection_stage_b"},
                {"offset": 20704, "bytes": 64, "candidate": "fp16_channel_parameters"},
                {"offset": 20768, "bytes": 16, "candidate": "zero_or_alignment"},
                {"offset": 20784, "bytes": 1024, "candidate": "fp16_output_projection"},
            ],
        },
        "instruction_inventory": {
            "fp8_m16n8k32_mma": fp8_count,
            "fp16_m16n8k16_mma": fp16_count,
            "texture_instructions": texture_instruction_count,
            "surface_stores": surface_store_count,
        },
        "validations": validations,
        "sources": {
            "plan": str(plan_path.resolve()),
            "tensor_manifest": str(tensor_manifest_path.resolve()),
            "model_arena": str(model_arena_path.resolve()),
            "case_manifest": str(case_manifest_path.resolve()),
            "ptx": str(ptx_path.resolve()),
        },
        "next_gate": "Execute the first 512-to-16 projection from the same captured input with AMD-native code.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("tensor_manifest", type=Path)
    parser.add_argument("model_arena", type=Path)
    parser.add_argument("case_manifest", type=Path)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.plan, args.tensor_manifest, args.model_arena, args.case_manifest, args.ptx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
