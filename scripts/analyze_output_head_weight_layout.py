#!/usr/bin/env python3
"""Validate output-head weight packing against captured RTX MMA B operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


HEAD_OFFSET = 147429888
HEAD_PROJECTION_BYTES = 8192
LANES = 32
RECORD_BYTES = 40


def packed_weight_index(input_channel: int, output_channel: int) -> int:
    if not 0 <= input_channel < 512 or not 0 <= output_channel < 16:
        raise ValueError("projection coordinate outside [512,16]")
    k_block, row = divmod(input_channel, 32)
    column = output_channel % 8
    lane = column * 4 + (row % 16) // 4
    element = row % 4 + (4 if row >= 16 else 0)
    n_half = output_channel // 8
    return k_block * 512 + lane * 16 + n_half * 8 + element


def analyze(trace_path: Path, model_arena_path: Path) -> dict:
    trace = trace_path.read_bytes()
    if len(trace) < 16 * LANES * RECORD_BYTES:
        raise ValueError("trace does not contain the first 16 FP8 MMA records")
    with model_arena_path.open("rb") as stream:
        stream.seek(HEAD_OFFSET)
        weights = stream.read(HEAD_PROJECTION_BYTES)
    if len(weights) != HEAD_PROJECTION_BYTES:
        raise ValueError("model arena does not contain the first head projection")

    # The first group uses K tiles 0 and 1.  Four A fragments reuse the same
    # two K tiles, hence the 0,1,2,3 B-fragment sequence repeats four times.
    byte_mismatches = 0
    first_mismatch = None
    for mma in range(16):
        k_block = (mma % 4) // 2
        n_half = mma % 2
        for lane in range(LANES):
            record = (mma * LANES + lane) * RECORD_BYTES
            actual = trace[record + 16:record + 24]
            expected_start = k_block * 512 + lane * 16 + n_half * 8
            expected = weights[expected_start:expected_start + 8]
            for byte_index, (got, want) in enumerate(zip(actual, expected)):
                if got == want:
                    continue
                byte_mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "mma": mma,
                        "lane": lane,
                        "byte": byte_index,
                        "actual": got,
                        "expected": want,
                    }

    indices = {
        packed_weight_index(input_channel, output_channel)
        for input_channel in range(512)
        for output_channel in range(16)
    }
    bijective = indices == set(range(HEAD_PROJECTION_BYTES))
    passed = byte_mismatches == 0 and bijective
    return {
        "schema": 1,
        "experiment": "output_head_first_projection_weight_layout",
        "status": "PASS" if passed else "FAIL",
        "classification": "RTX_TRACE_VERIFIED_WEIGHT_UNPACK_CONTRACT",
        "logical_weight_shape": [512, 16],
        "storage": "16_contiguous_k32_n16_ptx_b_fragment_tiles",
        "storage_bytes": HEAD_PROJECTION_BYTES,
        "mapping_bijective": bijective,
        "rtx_mma_records_checked": 16 * LANES,
        "rtx_b_operand_bytes_checked": 16 * LANES * 8,
        "unique_weight_bytes_covered": 1024,
        "byte_mismatches": byte_mismatches,
        "first_mismatch": first_mismatch,
        "trace_sha256": hashlib.sha256(trace).hexdigest().upper(),
        "model_arena_sha256": hashlib.sha256(model_arena_path.read_bytes()).hexdigest().upper(),
        "remaining_unknown": "pre-MMA activation normalization and spatial fragment assignment",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("model_arena", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.trace, args.model_arena)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
