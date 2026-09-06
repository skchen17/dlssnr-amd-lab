#!/usr/bin/env python3
"""Recover direct PTX parameter loads for one entry and decode captured bytes.

This is a static ABI aid, not a semantic decompiler.  It records only offsets,
load widths/types and the corresponding bytes from an optional captured launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

from extract_ptx_entry import extract_entry


PARAM_RE = re.compile(
    r"\.param\s+\.align\s+(\d+)\s+\.b8\s+([\w$]+)\[(\d+)\]"
)
LOAD_RE = re.compile(
    r"ld\.param(?:\.v([248]))?\.([busf])(8|16|32|64)\s+(.+?),\s*"
    r"\[([%\w$]+)(?:\+(\d+))?\]\s*;"
)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for begin, end in sorted(intervals):
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(begin, end) for begin, end in merged]


def decode_lanes(raw: bytes, lane_bytes: int) -> list[dict[str, object]]:
    lanes: list[dict[str, object]] = []
    for index in range(0, len(raw), lane_bytes):
        lane = raw[index : index + lane_bytes]
        item: dict[str, object] = {
            "index": index // lane_bytes,
            "hex_le": lane.hex(),
            "unsigned": int.from_bytes(lane, "little", signed=False),
            "signed": int.from_bytes(lane, "little", signed=True),
        }
        if lane_bytes == 4:
            value = struct.unpack("<f", lane)[0]
            item["float32"] = value if math.isfinite(value) else str(value)
        elif lane_bytes == 8:
            value = struct.unpack("<d", lane)[0]
            item["float64"] = value if math.isfinite(value) else str(value)
        lanes.append(item)
    return lanes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptx", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--param-hex")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_raw = args.ptx.read_bytes()
    source_text = source_raw.decode("utf-8").rstrip("\x00")
    isolated, entry_meta = extract_entry(source_text, args.entry)
    param_match = PARAM_RE.search(isolated)
    if not param_match:
        raise ValueError("entry parameter byte array was not found")
    param_align = int(param_match.group(1))
    param_name = param_match.group(2)
    param_size = int(param_match.group(3))

    aliases = {param_name}
    alias_pattern = re.compile(rf"mov\.b64\s+(%rd\d+),\s*{re.escape(param_name)}\s*;")
    aliases.update(match.group(1) for match in alias_pattern.finditer(isolated))

    captured = bytes.fromhex(args.param_hex) if args.param_hex else None
    if captured is not None and len(captured) != param_size:
        raise ValueError(
            f"captured parameter size {len(captured)} does not match PTX {param_size}"
        )

    accesses: list[dict[str, object]] = []
    intervals: list[tuple[int, int]] = []
    for line_number, line in enumerate(isolated.splitlines(), start=1):
        match = LOAD_RE.search(line)
        if not match or match.group(5) not in aliases:
            continue
        vector_width = int(match.group(1) or "1")
        scalar_bits = int(match.group(3))
        lane_bytes = scalar_bits // 8
        width = vector_width * lane_bytes
        offset = int(match.group(6) or "0")
        end = offset + width
        item: dict[str, object] = {
            "offset": offset,
            "end": end,
            "width": width,
            "vector_width": vector_width,
            "scalar_kind": match.group(2),
            "scalar_bits": scalar_bits,
            "destination": " ".join(match.group(4).split()),
            "line": line_number,
        }
        if captured is not None:
            raw = captured[offset:end]
            item["captured_hex"] = raw.hex()
            item["lanes"] = decode_lanes(raw, lane_bytes)
        accesses.append(item)
        intervals.append((offset, end))

    merged = merge_intervals(intervals)
    covered = sum(end - begin for begin, end in merged)
    output = {
        "schema": 1,
        "experiment": "ptx_param_abi",
        "entry": args.entry,
        "source_ptx_sha256": hashlib.sha256(source_raw).hexdigest(),
        "ptx_version": entry_meta["version"],
        "ptx_target": entry_meta["target"],
        "param_name": param_name,
        "param_align": param_align,
        "param_size": param_size,
        "captured_param_sha256": (
            hashlib.sha256(captured).hexdigest() if captured is not None else None
        ),
        "direct_load_count": len(accesses),
        "direct_covered_bytes": covered,
        "direct_coverage_intervals": [
            {"begin": begin, "end": end} for begin, end in merged
        ],
        "accesses": accesses,
        "interpretation_limit": (
            "Offsets and bit patterns are proven; pointer/scalar/tensor semantics "
            "require dynamic resource and before/after evidence."
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
                "entry": args.entry,
                "param_size": param_size,
                "direct_load_count": len(accesses),
                "direct_covered_bytes": covered,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
