#!/usr/bin/env python3
"""Compare the pre-E4 snapshot of the r816/r817/r944 dependency chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_postblock_r944_snapshot import LANES, RECORD_BYTES, TRACE_BYTES
except ModuleNotFoundError:
    from instrument_postblock_r944_snapshot import LANES, RECORD_BYTES, TRACE_BYTES


STAGES = (
    ("mul_r816", (0, 1), 4),
    ("mul_r817", (2, 3), 5),
    ("add_r944", (4, 5), 6),
    ("e4m3_rs53", (6,), 7),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES}: RTX={len(rtx)}, AMD={len(amd)}")
    records = []
    for lane in range(LANES):
        offset = lane * RECORD_BYTES
        records.append((struct.unpack_from("<7IH", rtx, offset),
                        struct.unpack_from("<7IH", amd, offset)))
    stages = []
    first_input_mismatch = None
    first_causal_mismatch = None
    for stage_name, input_indices, output_index in STAGES:
        input_mismatches = output_mismatches = exact_input_output_diff = 0
        for lane, (r, a) in enumerate(records):
            differing = [index for index in input_indices if r[index] != a[index]]
            input_mismatches += len(differing)
            if differing and first_input_mismatch is None:
                index = differing[0]
                first_input_mismatch = {
                    "stage": stage_name, "lane": lane, "word_index": index,
                    "rtx_input": f"0x{r[index]:08X}",
                    "amd_input": f"0x{a[index]:08X}",
                }
            if r[output_index] != a[output_index]:
                output_mismatches += 1
                if not differing:
                    exact_input_output_diff += 1
                    if first_causal_mismatch is None:
                        width = 4 if output_index < 7 else 2
                        first_causal_mismatch = {
                            "stage": stage_name, "lane": lane,
                            "inputs": [f"0x{r[index]:08X}" for index in input_indices],
                            "rtx_output": f"0x{r[output_index]:0{width * 2}X}",
                            "amd_output": f"0x{a[output_index]:0{width * 2}X}",
                        }
        stages.append({
            "stage": stage_name, "input_word_mismatches": input_mismatches,
            "output_word_mismatches": output_mismatches,
            "input_exact_output_diff_records": exact_input_output_diff,
        })
    return {
        "schema": 1,
        "experiment": "postblock_selected_cta_r944_pre_e4_snapshot_rtx_vs_rx9070xt",
        "status": "PASS",
        "classification": "NON_PRODUCER_SITE_R944_CAUSAL_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "bitwise_exact": rtx == amd, "stages": stages,
        "first_input_mismatch": first_input_mismatch,
        "first_causal_mismatch": first_causal_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_bytes(args.rtx.read_bytes(), args.amd.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
