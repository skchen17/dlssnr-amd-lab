#!/usr/bin/env python3
"""Join Feature-18 NVAPI objects/parameters to traced D3D12 resources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path


COPY_NAME = "cg2r_copy_kernel"
PREBLOCK_NAME = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"


def integer(value: object) -> int:
    return int(str(value), 16) if isinstance(value, str) else int(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h64(value: int) -> str:
    return f"0x{value:016X}"


def load_u64(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", raw, offset)[0]


def resource_for_va(resources: dict[int, dict[str, object]], value: int):
    for resource in resources.values():
        begin = integer(resource.get("gpu_va", 0))
        size = int(resource.get("width", 0))
        if begin and begin <= value < begin + size:
            return {
                "resource": resource.get("resource"),
                "gpu_va_begin": h64(begin),
                "byte_offset": value - begin,
                "byte_size": size,
                "dimension": resource.get("dimension"),
                "format": resource.get("format"),
            }
    return None


def resource_from_view(event: dict[str, object],
                       resources: dict[int, dict[str, object]]):
    pointer = integer(event.get("resource", 0))
    if not pointer:
        return None
    created = resources.get(pointer)
    if created is not None:
        return copy.deepcopy(created)
    # A view-creation call itself carries the real ID3D12Resource pointer and the
    # GetDesc-derived fields recorded by the hook.  Some ReShade resources are
    # allocated through an underlying device whose creation call bypasses the
    # registered wrapper vtable, so this is an independent direct observation,
    # not an inferred resource.
    return {
        "source": "view_creation",
        "resource": event.get("resource"),
        "dimension": event.get("resource_dimension"),
        "width": event.get("resource_width"),
        "height": event.get("resource_height"),
        "format": event.get("resource_format"),
        "gpu_va": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    functions: dict[str, str] = {}
    resources: dict[int, dict[str, object]] = {}
    descriptors: dict[int, dict[str, object]] = {}
    descriptor_histories: dict[int, list[dict[str, object]]] = {}
    objects: dict[int, dict[str, object]] = {}
    launches: list[dict[str, object]] = []
    event_counts: dict[str, int] = {}
    invalid: list[dict[str, object]] = []
    unresolved_copies: list[dict[str, object]] = []

    def remember_descriptor(handle: int, binding: dict[str, object]) -> None:
        descriptors[handle] = binding
        descriptor_histories.setdefault(handle, []).append(copy.deepcopy(binding))

    def select_binding(handle: int, require_resource: bool = False):
        history = descriptor_histories.get(handle, [])
        if not history:
            return None, "unresolved"
        current = history[-1]
        if not require_resource or current.get("resource") is not None:
            return copy.deepcopy(current), "current_at_object_return"
        for candidate in reversed(history[:-1]):
            if candidate.get("resource") is not None:
                return copy.deepcopy(candidate), "last_non_null_before_internal_clobber"
        return copy.deepcopy(current), "current_null_no_prior_resource"

    with args.trace.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                invalid.append({"line": line_number, "error": str(error)})
                continue
            kind = str(event.get("ev", ""))
            event_counts[kind] = event_counts.get(kind, 0) + 1
            if kind == "nvapi_create_cu_function" and event.get("status") == 0:
                functions[str(event["function"]).lower()] = str(event["name"])
            elif kind == "d3d12_resource_create" and event.get("resource") != "0x0":
                resources[integer(event["resource"])] = event
            elif kind in {"d3d12_create_srv", "d3d12_create_uav"}:
                resource = resource_from_view(event, resources)
                remember_descriptor(integer(event["descriptor"]), {
                    "kind": "srv" if kind.endswith("srv") else "uav",
                    "created_line": line_number,
                    "resource": copy.deepcopy(resource),
                    "view": event,
                    "copy_chain": [],
                })
            elif kind == "d3d12_create_sampler":
                remember_descriptor(integer(event["descriptor"]), {
                    "kind": "sampler",
                    "created_line": line_number,
                    "sampler": event,
                    "copy_chain": [],
                })
            elif kind == "d3d12_copy_descriptor":
                source = integer(event["source"])
                destination = integer(event["destination"])
                binding = descriptors.get(source)
                if binding is None:
                    unresolved_copies.append(
                        {"line": line_number, "source": h64(source), "destination": h64(destination)}
                    )
                else:
                    cloned = copy.deepcopy(binding)
                    cloned["copy_chain"].append(
                        {"line": line_number, "source": h64(source), "destination": h64(destination)}
                    )
                    remember_descriptor(destination, cloned)
            elif kind == "nvapi_get_cuda_merged_texture_sampler" and event.get("status") == 0:
                handle = integer(event["texture_handle"])
                if handle:
                    texture_descriptor = integer(event["texture_descriptor"])
                    sampler_descriptor = integer(event["sampler_descriptor"])
                    texture_binding, texture_resolution = select_binding(
                        texture_descriptor, require_resource=True
                    )
                    sampler_binding, sampler_resolution = select_binding(
                        sampler_descriptor
                    )
                    objects[handle] = {
                        "kind": "merged_texture_sampler",
                        "created_line": line_number,
                        "texture_descriptor": h64(texture_descriptor),
                        "texture_binding": texture_binding,
                        "texture_binding_resolution": texture_resolution,
                        "sampler_descriptor": h64(sampler_descriptor),
                        "sampler_binding": sampler_binding,
                        "sampler_binding_resolution": sampler_resolution,
                    }
            elif kind == "nvapi_get_cuda_independent_descriptor" and event.get("status") == 0:
                handle = integer(event["handle"])
                if handle:
                    descriptor = integer(event["descriptor"])
                    binding, resolution = select_binding(
                        descriptor, require_resource=True
                    )
                    objects[handle] = {
                        "kind": "independent_descriptor",
                        "created_line": line_number,
                        "type": event.get("type"),
                        "descriptor": h64(descriptor),
                        "binding": binding,
                        "binding_resolution": resolution,
                    }
            elif kind == "nvapi_launch_cu_kernel":
                name = functions.get(str(event.get("function", "")).lower())
                if name in {COPY_NAME, PREBLOCK_NAME}:
                    launches.append({"line": line_number, "name": name, "event": event})

    copy_joins: list[dict[str, object]] = []
    preblock_joins: list[dict[str, object]] = []
    for launch in launches:
        event = launch["event"]
        raw = bytes.fromhex(str(event["param_hex"]))
        common = {
            "trace_line": launch["line"],
            "frame": event.get("frame"),
            "slot": event.get("slot"),
            "param_sha256": hashlib.sha256(raw).hexdigest(),
        }
        if launch["name"] == COPY_NAME:
            if len(raw) != 72:
                raise ValueError("copy parameter block is not 72 bytes")
            texture = load_u64(raw, 0)
            surface = load_u64(raw, 8)
            common.update(
                {
                    "texture_handle": h64(texture),
                    "texture_object": copy.deepcopy(objects.get(texture)),
                    "surface_handle": h64(surface),
                    "surface_object": copy.deepcopy(objects.get(surface)),
                    "dimensions": list(struct.unpack_from("<II", raw, 64)),
                }
            )
            copy_joins.append(common)
        else:
            if len(raw) != 264:
                raise ValueError("pre-block parameter block is not 264 bytes")
            fields = []
            for offset in [0, 8, 16, 24, 32, 168, 184, 192, 216, 224, 248]:
                value = load_u64(raw, offset)
                fields.append(
                    {
                        "offset": offset,
                        "value": h64(value),
                        "descriptor_object": copy.deepcopy(objects.get(value)),
                        "buffer_interval": resource_for_va(resources, value),
                    }
                )
            common["direct_b64_fields"] = fields
            preblock_joins.append(common)

    def copy_resolved(item: dict[str, object]) -> bool:
        texture = item.get("texture_object") or {}
        surface = item.get("surface_object") or {}
        return bool(
            (texture.get("texture_binding") or {}).get("resource")
            and (surface.get("binding") or {}).get("resource")
        )

    unresolved_copy_launches = sum(not copy_resolved(item) for item in copy_joins)
    preblock_primary_resolved = sum(
        bool((item["direct_b64_fields"][0].get("descriptor_object") or {}).get("texture_binding"))
        for item in preblock_joins
    )
    run_summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    output = {
        "schema": 1,
        "experiment": "d3d12_resource_trace_join",
        "classification": "RTX_RESOURCE_IDENTITY_EVIDENCE",
        "counts_as_s6": False,
        "trace_sha256": sha256(args.trace),
        "summary_sha256": sha256(args.summary),
        "reference_positive": {
            "host_exit": run_summary.get("host_exit"),
            "evaluates_ok": run_summary.get("evaluates_ok"),
            "feature18_created": run_summary.get("feature18_created"),
            "feature18_evaluation_succeeded": run_summary.get("feature18_evaluation_succeeded"),
        },
        "invalid_json_line_count": len(invalid),
        "invalid_json_lines": invalid,
        "event_counts": event_counts,
        "resource_count": len(resources),
        "descriptor_binding_count": len(descriptors),
        "nvapi_object_count": len(objects),
        "unresolved_descriptor_copy_count": len(unresolved_copies),
        "unresolved_descriptor_copies": unresolved_copies,
        "copy_launch_count": len(copy_joins),
        "copy_unresolved_resource_pairs": unresolved_copy_launches,
        "copy_joins": copy_joins,
        "preblock_launch_count": len(preblock_joins),
        "preblock_primary_resource_resolved": preblock_primary_resolved,
        "preblock_joins": preblock_joins,
        "interpretation_limit": (
            "Resource identity, format, dimensions and GPU-VA containment are proven. "
            "Resource contents and tensor dtype/stride require bounded snapshots."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    passed = bool(
        not invalid
        and copy_joins
        and unresolved_copy_launches == 0
        and preblock_joins
        and preblock_primary_resolved == len(preblock_joins)
    )
    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "resources": len(resources),
                "copy_launches": len(copy_joins),
                "copy_unresolved_resource_pairs": unresolved_copy_launches,
                "preblock_primary_resource_resolved": preblock_primary_resolved,
                "invalid_json_lines": len(invalid),
            }
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
