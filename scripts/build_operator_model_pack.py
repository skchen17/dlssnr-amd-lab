#!/usr/bin/env python3
"""Build a local operator-level model pack from an accepted captured graph.

This intentionally discards instruction-level/PTX identity.  It records stable
operator families, resource bindings, and weight anchors for AMD-native kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


STAGES = (
    ("clear", 0, 0),
    ("encoder_1h_32", 1, 5),
    ("encoder_2h_64", 6, 9),
    ("encoder_4h_128", 10, 15),
    ("encoder_8h_256", 16, 23),
    ("bottleneck_16h_512", 24, 56),
    ("vit_1d", 57, 98),
    ("decoder_16h_512", 99, 131),
    ("decoder_8h_256", 132, 139),
    ("decoder_4h_128", 140, 145),
    ("decoder_2h_64", 146, 149),
    ("decoder_1h_32", 150, 153),
    ("output_head", 154, 154),
    ("final_copy", 155, 155),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def classify(function: str) -> str:
    name = function.casefold()
    checks = (
        ("clear", "clear"),
        ("post_block", "post_block"),
        ("final_copy", "copy_kernel"),
        ("repack_2d_to_1d", "repack_2d_to_1d"),
        ("repack_1d_to_2d", "repack_1d_to_2d"),
        ("decoder_input_upsample", "dec_input_upsample"),
        ("ffn_expand", "ffn_expand"),
        ("ffn_contract", "ffn_contract"),
        ("feed_forward_projection", "ffwd_proj"),
        ("feed_forward", "ffwd"),
        ("qkv", "qkv"),
        ("attention", "attention"),
        ("projection_pool", "proj_pool"),
        ("projection", "projection"),
        ("projection", "_proj_"),
        ("final_head", "final_head"),
        ("swin_upsample", "upsample"),
        ("swin_downsample", "ds_wait"),
        ("swin_output", "outview_wait"),
        ("swin_input", "inpview"),
        ("swin_block", "chained"),
        ("pre_block", "pre_block"),
    )
    for operator, needle in checks:
        if needle in name:
            return operator
    return "unknown"


def resolve_asset(plan_path: Path, spec: dict) -> Path:
    path = Path(spec["path"])
    return path if path.is_absolute() else plan_path.parent / path


def load_decoded_tensors(model_path: Path, model_bytes: int, model_hash: str) -> tuple[dict, str | None]:
    manifest_path = model_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return {}, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("experiment") != "local_dlssnr_weight_resource_decode":
        return {}, None
    arena = manifest.get("model_arena", {})
    if int(arena.get("bytes", -1)) != model_bytes or arena.get("sha256", "").upper() != model_hash:
        raise ValueError("decoded tensor manifest does not describe the selected model arena")
    tensors = {int(tensor["arena_offset"]): tensor for tensor in manifest.get("tensors", [])}
    if len(tensors) != len(manifest.get("tensors", [])):
        raise ValueError("decoded tensor manifest contains duplicate arena offsets")
    return tensors, manifest_path.name


def build_pack(
    plan_path: Path,
    output_dir: Path,
    copy_weights: bool,
    model_arena_override: Path | None = None,
) -> dict:
    plan_path = plan_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if any(part.casefold() == "deliverables" for part in output_dir.resolve().parts):
        raise ValueError("operator model packs contain private metadata and cannot be written to deliverables")

    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8-sig"))
    slots = plan.get("slots", [])
    if len(slots) != 156 or [int(slot["slot"]) for slot in slots] != list(range(156)):
        raise ValueError("source plan must contain contiguous slots 0..155")

    model_path = (
        model_arena_override.resolve(strict=True)
        if model_arena_override is not None
        else resolve_asset(plan_path, plan["model_arena"]).resolve(strict=True)
    )
    model_bytes = model_path.stat().st_size
    expected_bytes = int(plan["model_arena"].get("bytes", model_bytes))
    expected_hash = plan["model_arena"]["sha256"].upper()
    actual_hash = sha256_file(model_path)
    if model_bytes != expected_bytes or actual_hash != expected_hash:
        raise ValueError("model arena size or hash differs from the accepted plan")
    decoded_tensors, decoded_manifest_name = load_decoded_tensors(
        model_path, model_bytes, actual_hash
    )

    anchors: dict[int, set[int]] = {}
    for slot in slots:
        for view in slot.get("weight_param_views", []):
            offset = int(view["weight_offset"])
            if not 0 <= offset < model_bytes:
                raise ValueError(f"slot {slot['slot']} has invalid weight offset {offset}")
            anchors.setdefault(offset, set()).add(int(slot["slot"]))
    ordered_offsets = sorted(anchors)
    if decoded_tensors and set(ordered_offsets) != set(decoded_tensors):
        raise ValueError("captured weight anchors do not match decoded tensor offsets")
    spans = []
    with model_path.open("rb") as stream:
        for index, offset in enumerate(ordered_offsets):
            end = ordered_offsets[index + 1] if index + 1 < len(ordered_offsets) else model_bytes
            decoded = decoded_tensors.get(offset)
            payload_bytes = int(decoded["data_bytes"]) if decoded else end - offset
            if payload_bytes > end - offset:
                raise ValueError(f"decoded tensor at {offset} overlaps the next tensor")
            stream.seek(offset)
            payload = stream.read(payload_bytes)
            span = {
                "index": index,
                "offset": offset,
                "bytes": payload_bytes,
                "aligned_bytes": end - offset,
                "sha256": sha256_bytes(payload),
                "referenced_by_slots": sorted(anchors[offset]),
                "boundary_confidence": (
                    "resource_record_exact" if decoded else "anchor_to_next_anchor_not_yet_tensor_exact"
                ),
            }
            if decoded:
                if span["sha256"] != decoded["sha256"].upper():
                    raise ValueError(f"decoded tensor hash differs at arena offset {offset}")
                span.update(
                    {
                        "name": decoded["name"],
                        "dtype": decoded["dtype"],
                        "element_count": int(decoded["element_count"]),
                    }
                )
            spans.append(span)

    graph_stages = []
    unknown = []
    for name, first, last in STAGES:
        nodes = []
        for slot in slots[first : last + 1]:
            operator = classify(slot["function"])
            if operator == "unknown":
                unknown.append(int(slot["slot"]))
            nodes.append(
                {
                    "slot": int(slot["slot"]),
                    "operator": operator,
                    "source_function": slot["function"],
                    "activation_bindings": slot.get("activation_param_views", []),
                    "weight_bindings": slot.get("weight_param_views", []),
                    "special": slot.get("special", "normal"),
                }
            )
        graph_stages.append(
            {
                "name": name,
                "source_slot_range": [first, last],
                "amd_fusion_boundary": True,
                "nodes": nodes,
            }
        )

    output_dir.mkdir(parents=True)
    if copy_weights:
        shutil.copyfile(model_path, output_dir / "weights.raw")
    manifest = {
        "schema": 1,
        "experiment": "amd_native_operator_reconstruction_model_pack",
        "status": "SEMANTIC_GRAPH_READY" if not unknown else "SEMANTIC_GRAPH_NEEDS_CLASSIFICATION",
        "redistributable": False,
        "execution_policy": {
            "runtime_vendor": "AMD",
            "runtime_must_not_load_nvidia": True,
            "instruction_parity_required": False,
            "intermediate_bitwise_parity_required": False,
            "final_output_contract": "complete_rgba16f_frame",
            "internal_output_parameterization": "residual_allowed",
        },
        "source": {
            "plan_filename": plan_path.name,
            "plan_sha256": sha256_bytes(plan_bytes),
            "captured_slot_count": len(slots),
        },
        "model": {
            "bytes": model_bytes,
            "sha256": actual_hash,
            "source_filename": model_path.name,
            "decoded_tensor_manifest": decoded_manifest_name,
            "local_file": "weights.raw" if copy_weights else None,
            "weight_anchor_count": len(spans),
            "tensor_count": len(decoded_tensors) if decoded_tensors else None,
        },
        "activation_arena_bytes": int(plan["activation_arena_bytes"]),
        "stages": graph_stages,
        "weight_spans": spans,
        "unknown_operator_slots": unknown,
        "next_gate": "replace one complete stage with AMD-native tensor operators and compare final image impact",
        "notice": "Contains provenance for user-derived model data. Never redistribute the local weights file.",
    }
    (output_dir / "operator_model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--copy-weights", action="store_true")
    parser.add_argument(
        "--model-arena",
        type=Path,
        help="use a locally decoded model arena instead of the plan-relative captured file",
    )
    args = parser.parse_args()
    try:
        manifest = build_pack(args.plan, args.output_dir, args.copy_weights, args.model_arena)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "stages": len(manifest["stages"]),
                "slots": manifest["source"]["captured_slot_count"],
                "weight_anchors": manifest["model"]["weight_anchor_count"],
                "model_bytes": manifest["model"]["bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
