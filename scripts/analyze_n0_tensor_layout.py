#!/usr/bin/env python3
"""Derive bounded N0 FP8 tensor-layout facts from PTX dimensions and raw sizes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


CHANNELS = 32


def zero_row_ranges(data: bytes, row_bytes: int) -> list[list[int]]:
    rows = [index for index in range(len(data) // row_bytes)
            if not any(data[index * row_bytes : (index + 1) * row_bytes])]
    ranges: list[list[int]] = []
    for row in rows:
        if ranges and row == ranges[-1][1] + 1:
            ranges[-1][1] = row
        else:
            ranges.append([row, row])
    return ranges


def channel_stats(data: bytes, width: int, height: int) -> list[dict]:
    stats = []
    for channel in range(CHANNELS):
        values = data[channel::CHANNELS]
        histogram = collections.Counter(values)
        stats.append({
            "channel": channel,
            "elements": width * height,
            "nonzero": sum(value != 0 for value in values),
            "unique_byte_values": len(histogram),
            "most_common": [[value, count] for value, count in histogram.most_common(5)],
        })
    return stats


def tensor_record(name: str, data: bytes, height: int, width: int, storage: str) -> dict:
    expected = height * width * CHANNELS
    if len(data) != expected:
        raise ValueError(f"{name}: expected {expected} bytes, got {len(data)}")
    histogram = collections.Counter(data)
    if storage == "scratch_mma_tiles":
        storage_layout = "[H/4,W/4,lane32,byte16]"
        storage_shape = [height // 4, width // 4, 32, 16]
        storage_strides = [(width // 4) * 512, 512, 16, 1]
        row_bytes = (width // 4) * 512
    elif storage == "output_channel_planes":
        storage_layout = "[channel_plane2,H,W,channel16]"
        storage_shape = [2, height, width, 16]
        storage_strides = [height * width * 16, width * 16, 16, 1]
        row_bytes = width * 16
    else:
        raise ValueError(f"unknown storage layout: {storage}")
    return {
        "name": name,
        "logical_shape": [height, width, CHANNELS],
        "shape": [height, width, CHANNELS],
        "storage_layout": storage_layout,
        "storage_shape": storage_shape,
        "storage_strides_bytes": storage_strides,
        "candidate_dtype": "e4m3_fp8_byte",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "nonzero_bytes": sum(value != 0 for value in data),
        "unique_byte_values": len(histogram),
        "all_zero_storage_row_ranges": zero_row_ranges(data, row_bytes),
        "most_common_byte_values": [[value, count] for value, count in histogram.most_common(16)],
    }


def analyze(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    windows = {window["name"]: window for window in metadata["windows"]}
    root = metadata_path.parent
    padded_height = int(metadata["padded_height"])
    padded_width = int(metadata["padded_width"])
    scratch = (root / windows["scratch"]["after_file"]).read_bytes()
    output = (root / windows["output"]["after_file"]).read_bytes()
    weights_before = (root / windows["weights"]["before_file"]).read_bytes()
    weights_after = (root / windows["weights"]["after_file"]).read_bytes()
    scratch_tensor = tensor_record(
        "scratch_after", scratch, padded_height, padded_width, "scratch_mma_tiles"
    )
    output_tensor = tensor_record(
        "output_after", output, padded_height // 2, padded_width // 2,
        "output_channel_planes"
    )
    weights_unchanged = weights_before == weights_after
    pass_condition = (
        metadata.get("status") == "PASS"
        and weights_unchanged
        and scratch_tensor["nonzero_bytes"] > 0
        and output_tensor["nonzero_bytes"] > 0
    )
    return {
        "schema": 1,
        "experiment": "n0_tensor_layout_analysis",
        "status": "PASS" if pass_condition else "FAIL",
        "function": metadata.get("function"),
        "layout_derivation": {
            "channels": CHANNELS,
            "scratch_equation": f"{padded_height}*{padded_width}*32={len(scratch)}",
            "output_equation": (
                f"({padded_height}/2)*({padded_width}/2)*32={len(output)}"
            ),
            "storage_basis": (
                "Original PTX converts f16x2 with cvt.rn.satfinite.e4m3x2 and "
                "stores each 4x4x32 MMA tile as 32 lanes x 16 bytes; output addressing "
                "uses two HxWx16 channel planes."
            ),
        },
        "scratch": scratch_tensor,
        "output": output_tensor,
        "weights": {
            "capture_bytes": len(weights_before),
            "unchanged": weights_unchanged,
            "sha256": hashlib.sha256(weights_before).hexdigest(),
        },
        "counts_as_s6": False,
        "interpretation_limit": (
            "Storage layouts are derived from PTX address equations and the m16n8 D-fragment "
            "mapping; semantic channel meanings are not assigned."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.metadata)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "scratch_shape": result["scratch"]["shape"],
        "scratch_nonzero": result["scratch"]["nonzero_bytes"],
        "scratch_zero_storage_rows": result["scratch"]["all_zero_storage_row_ranges"],
        "output_shape": result["output"]["shape"],
        "output_nonzero": result["output"]["nonzero_bytes"],
        "output_zero_storage_rows": result["output"]["all_zero_storage_row_ranges"],
        "weights_unchanged": result["weights"]["unchanged"],
    }, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
