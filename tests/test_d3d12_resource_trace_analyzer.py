#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def event(kind: str, **values):
    return {"ev": kind, **values}


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    analyzer = repo / "scripts" / "analyze_d3d12_resource_trace.py"
    copy_params = bytearray(72)
    struct.pack_into("<QQ", copy_params, 0, 0xABC, 0xDEF)
    struct.pack_into("<II", copy_params, 64, 640, 360)
    pre_params = bytearray(264)
    struct.pack_into("<Q", pre_params, 0, 0xAAA)
    struct.pack_into("<Q", pre_params, 216, 0x100020)
    records = [
        event("d3d12_resource_create", resource="0x1", gpu_va="0x0", dimension=3, width=640, height=360, format=2),
        event("d3d12_resource_create", resource="0x2", gpu_va="0x0", dimension=3, width=640, height=360, format=2),
        event("d3d12_resource_create", resource="0x3", gpu_va="0x100000", dimension=1, width=65536, height=1, format=0),
        event("d3d12_create_srv", resource="0x1", descriptor="0x100", view_format=2, view_dimension=4),
        event("d3d12_create_uav", resource="0x2", descriptor="0x120", view_format=2, view_dimension=4),
        event("d3d12_create_sampler", descriptor="0x200", filter=0),
        event("d3d12_copy_descriptor", source="0x100", destination="0x700", heap_type=0),
        event("d3d12_copy_descriptor", source="0x120", destination="0x720", heap_type=0),
        event("d3d12_copy_descriptor", source="0x200", destination="0x800", heap_type=1),
        event("nvapi_get_cuda_merged_texture_sampler", status=0, texture_handle="0x0000000000000ABC", texture_descriptor="0x700", sampler_descriptor="0x800"),
        event("nvapi_get_cuda_independent_descriptor", status=0, handle="0x0000000000000DEF", descriptor="0x720", type=0),
        event("nvapi_get_cuda_merged_texture_sampler", status=0, texture_handle="0x0000000000000AAA", texture_descriptor="0x700", sampler_descriptor="0x800"),
        event("nvapi_create_cu_function", status=0, function="0x10", name="cg2r_copy_kernel"),
        event("nvapi_create_cu_function", status=0, function="0x11", name="cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"),
        event("nvapi_launch_cu_kernel", function="0x11", frame=1, slot=1, param_hex=pre_params.hex()),
        event("nvapi_launch_cu_kernel", function="0x10", frame=1, slot=155, param_hex=copy_params.hex()),
    ]
    summary = {"host_exit": 0, "evaluates_ok": 300, "feature18_created": True, "feature18_evaluation_succeeded": True}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace = root / "trace.jsonl"
        summary_path = root / "summary.json"
        output = root / "output.json"
        trace.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(analyzer), "--trace", str(trace), "--summary", str(summary_path), "--output", str(output)],
            check=False,
        )
        parsed = json.loads(output.read_text(encoding="utf-8"))
        assert result.returncode == 0
        assert parsed["copy_unresolved_resource_pairs"] == 0
        assert parsed["preblock_primary_resource_resolved"] == 1
        field216 = next(field for field in parsed["preblock_joins"][0]["direct_b64_fields"] if field["offset"] == 216)
        assert field216["buffer_interval"]["byte_offset"] == 32
    print("[PASS] synthetic descriptor/resource/GPU-VA join")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
