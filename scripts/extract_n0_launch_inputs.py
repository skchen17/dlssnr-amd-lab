#!/usr/bin/env python3
"""Extract the captured N0 264-byte launch block without retaining GPU handles as inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ENTRY = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"


def extract(trace: Path) -> tuple[bytes, dict]:
    function_handles = set()
    launch = None
    invalid = 0
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if event.get("ev") == "nvapi_create_cu_function" and event.get("name") == ENTRY:
            function_handles.add(str(event.get("function")))
        if (
            event.get("ev") == "nvapi_launch_cu_kernel"
            and str(event.get("function")) in function_handles
            and int(event.get("param_size", 0)) == 264
        ):
            launch = event
            break
    if launch is None:
        raise ValueError("captured N0 launch not found")
    raw = bytes.fromhex(str(launch["param_hex"]))
    if len(raw) != 264:
        raise ValueError("captured N0 launch block has wrong size")
    report = {
        "schema": 1,
        "experiment": "n0_launch_block_extraction",
        "status": "PASS",
        "classification": "CAPTURED_PARAMETER_ABI",
        "counts_as_s6": False,
        "function": ENTRY,
        "frame": launch.get("frame"),
        "slot": launch.get("slot"),
        "grid": launch.get("grid"),
        "block": launch.get("block"),
        "param_bytes": len(raw),
        "param_sha256": hashlib.sha256(raw).hexdigest(),
        "invalid_json_lines": invalid,
        "runtime_replacements": [0, 8, 16, 24, 32, 216, 224, 248],
        "warning": "captured GPU handles are provenance only and must be replaced before local execution",
    }
    return raw, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("params", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    raw, report = extract(args.trace)
    args.params.parent.mkdir(parents=True, exist_ok=True)
    args.params.write_bytes(raw)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
