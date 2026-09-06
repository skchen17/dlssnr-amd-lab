#!/usr/bin/env python3
"""Join NVAPI descriptor-object creation events to captured kernel parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


COPY_NAME = "cg2r_copy_kernel"
PREBLOCK_NAME = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def u64(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", raw, offset)[0]


def u32(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<I", raw, offset)[0]


def f32(raw: bytes, offset: int) -> float:
    return struct.unpack_from("<f", raw, offset)[0]


def hex64(value: int) -> str:
    return f"0x{value:016X}"


def descriptor_summary(event: dict[str, object] | None) -> dict[str, object] | None:
    if event is None:
        return None
    if event["ev"] == "nvapi_get_cuda_merged_texture_sampler":
        return {
            "kind": "merged_texture_sampler",
            "texture_descriptor": event.get("texture_descriptor"),
            "sampler_descriptor": event.get("sampler_descriptor"),
            "status": event.get("status"),
        }
    return {
        "kind": "independent_descriptor",
        "type": event.get("type"),
        "descriptor": event.get("descriptor"),
        "status": event.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    functions: dict[str, str] = {}
    descriptor_events: dict[int, dict[str, object]] = {}
    selected_launches: list[dict[str, object]] = []
    counts = {"merged": 0, "independent": 0, "launch": 0}
    invalid_json_lines: list[dict[str, object]] = []

    with args.trace.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                invalid_json_lines.append(
                    {
                        "line": line_number,
                        "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                        "error": str(error),
                    }
                )
                continue
            kind = event.get("ev")
            if kind == "nvapi_create_cu_function" and event.get("status") == 0:
                functions[str(event["function"]).lower()] = str(event["name"])
            elif kind == "nvapi_get_cuda_merged_texture_sampler":
                counts["merged"] += 1
                if event.get("status") == 0:
                    handle = int(str(event["texture_handle"]), 16)
                    if handle:
                        descriptor_events[handle] = event
            elif kind == "nvapi_get_cuda_independent_descriptor":
                counts["independent"] += 1
                if event.get("status") == 0:
                    handle = int(str(event["handle"]), 16)
                    if handle:
                        descriptor_events[handle] = event
            elif kind == "nvapi_launch_cu_kernel":
                counts["launch"] += 1
                name = functions.get(str(event.get("function", "")).lower())
                if name in {COPY_NAME, PREBLOCK_NAME}:
                    selected_launches.append(
                        {"line": line_number, "name": name, "event": event}
                    )

    run_summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    copy_launches: list[dict[str, object]] = []
    preblock_launches: list[dict[str, object]] = []
    for selected in selected_launches:
        event = selected["event"]
        raw = bytes.fromhex(str(event["param_hex"]))
        common = {
            "trace_line": selected["line"],
            "frame": event.get("frame"),
            "slot": event.get("slot"),
            "grid": event.get("grid"),
            "block": event.get("block"),
            "param_size": len(raw),
            "param_sha256": hashlib.sha256(raw).hexdigest(),
        }
        if selected["name"] == COPY_NAME:
            if len(raw) != 72:
                raise ValueError(f"copy parameter size is {len(raw)}, expected 72")
            texture = u64(raw, 0)
            surface = u64(raw, 8)
            common.update(
                {
                    "texture_handle": hex64(texture),
                    "texture_object": descriptor_summary(descriptor_events.get(texture)),
                    "surface_handle": hex64(surface),
                    "surface_object": descriptor_summary(descriptor_events.get(surface)),
                    "source_offset": [f32(raw, 16), f32(raw, 20)],
                    "source_extent": [f32(raw, 24), f32(raw, 28)],
                    "source_scale": [f32(raw, 32), f32(raw, 36)],
                    "destination_offset": [f32(raw, 40), f32(raw, 44)],
                    "unused_extent": [f32(raw, 48), f32(raw, 52)],
                    "unused_scale": [f32(raw, 56), f32(raw, 60)],
                    "dimensions": [u32(raw, 64), u32(raw, 68)],
                }
            )
            copy_launches.append(common)
        else:
            if len(raw) != 264:
                raise ValueError(f"pre-block parameter size is {len(raw)}, expected 264")
            # These are the direct b64 load offsets proven by R-28.  A value is
            # named as a descriptor object only when it exactly matches a traced
            # successful descriptor API result.
            b64_offsets = [0, 8, 16, 24, 32, 168, 184, 192, 216, 224, 248]
            fields = []
            for offset in b64_offsets:
                value = u64(raw, offset)
                fields.append(
                    {
                        "offset": offset,
                        "value": hex64(value),
                        "descriptor_object": descriptor_summary(
                            descriptor_events.get(value)
                        ),
                    }
                )
            common["direct_b64_fields"] = fields
            common["integer_fields"] = {
                "offset_200": u32(raw, 200),
                "offset_208_212": [u32(raw, 208), u32(raw, 212)],
                "offset_240_244": [u32(raw, 240), u32(raw, 244)],
                "offset_256_260": [u32(raw, 256), u32(raw, 260)],
            }
            preblock_launches.append(common)

    unresolved_copy = sum(
        launch["texture_object"] is None or launch["surface_object"] is None
        for launch in copy_launches
    )
    output = {
        "schema": 1,
        "experiment": "descriptor_trace_join",
        "classification": "RTX_OBSERVATIONAL_ABI_EVIDENCE",
        "counts_as_s6": False,
        "trace_sha256": sha256(args.trace),
        "summary_sha256": sha256(args.summary),
        "reference_positive": {
            "host_exit": run_summary.get("host_exit"),
            "evaluates_ok": run_summary.get("evaluates_ok"),
            "feature18_created": run_summary.get("feature18_created"),
            "feature18_evaluation_succeeded": run_summary.get(
                "feature18_evaluation_succeeded"
            ),
        },
        "event_counts": counts,
        "invalid_json_line_count": len(invalid_json_lines),
        "invalid_json_lines": invalid_json_lines,
        "successful_nonzero_descriptor_objects": len(descriptor_events),
        "copy_launch_count": len(copy_launches),
        "copy_unresolved_object_pairs": unresolved_copy,
        "copy_launches": copy_launches,
        "preblock_launch_count": len(preblock_launches),
        "preblock_launches": preblock_launches,
        "interpretation_limit": (
            "Descriptor handle identity and parameter offsets are proven. D3D12 "
            "resource identity/format/content require device resource tracing and "
            "bounded before/after snapshots."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "copy_launches": len(copy_launches),
                "copy_unresolved_object_pairs": unresolved_copy,
                "preblock_launches": len(preblock_launches),
                "descriptor_objects": len(descriptor_events),
            }
        )
    )
    return 0 if copy_launches and not unresolved_copy and preblock_launches else 2


if __name__ == "__main__":
    raise SystemExit(main())
