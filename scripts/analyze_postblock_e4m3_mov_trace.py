#!/usr/bin/env python3
"""Compare post-block E4M3/movmatrix traces and identify the first producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_postblock_e4m3_mov_trace import (
        E4M3_COUNT, LANES, MOV_COUNT, RECORD_BYTES, TRACE_BYTES,
    )
except ModuleNotFoundError:
    from instrument_postblock_e4m3_mov_trace import (
        E4M3_COUNT, LANES, MOV_COUNT, RECORD_BYTES, TRACE_BYTES,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES} bytes")
    families = []
    first_causal = None
    for name, start, count in (("e4m3", 0, E4M3_COUNT),
                               ("movmatrix", E4M3_COUNT, MOV_COUNT)):
        input_mismatches = output_mismatches = 0
        input_exact_output_diff = 0
        warp_input_exact_output_diff = 0
        first = None
        first_input = None
        for operation in range(count):
            records = []
            for lane in range(LANES):
                offset = (start + operation) * LANES * RECORD_BYTES + lane * RECORD_BYTES
                ri, ro = struct.unpack_from("<II", rtx, offset)
                ai, ao = struct.unpack_from("<II", amd, offset)
                if name == "e4m3":
                    ro &= 0xFFFF
                    ao &= 0xFFFF
                records.append((lane, ri, ro, ai, ao))
                input_mismatches += ri != ai
                output_mismatches += ro != ao
                if ri != ai and first_input is None:
                    first_input = {
                        "family": name, "operation": operation, "lane": lane,
                        "rtx_input": f"0x{ri:08X}", "amd_input": f"0x{ai:08X}",
                    }
                if ri == ai and ro != ao:
                    input_exact_output_diff += 1
                    item = {"family": name, "operation": operation, "lane": lane,
                            "rtx_input": f"0x{ri:08X}", "amd_input": f"0x{ai:08X}",
                            "rtx_output": f"0x{ro:08X}", "amd_output": f"0x{ao:08X}"}
                    if name == "e4m3" and first is None:
                        first = item
                    if name == "e4m3" and first_causal is None:
                        first_causal = item
            if name == "movmatrix" and all(ri == ai for _, ri, _, ai, _ in records):
                differences = [(lane, ri, ro, ai, ao) for lane, ri, ro, ai, ao in records
                               if ro != ao]
                if differences:
                    warp_input_exact_output_diff += 1
                    lane, ri, ro, ai, ao = differences[0]
                    item = {"family": name, "operation": operation, "lane": lane,
                            "rtx_input": f"0x{ri:08X}", "amd_input": f"0x{ai:08X}",
                            "rtx_output": f"0x{ro:08X}", "amd_output": f"0x{ao:08X}"}
                    if first is None:
                        first = item
                    if first_causal is None:
                        first_causal = item
        families.append({
            "name": name, "operation_count": count,
            "input_word_mismatches": input_mismatches,
            "output_word_mismatches": output_mismatches,
            "input_exact_output_diff_records": input_exact_output_diff,
            "warp_input_exact_output_diff_operations": warp_input_exact_output_diff,
            "first_input_mismatch": first_input,
            "first_causal_mismatch": first,
        })
    return {
        "schema": 1,
        "experiment": "postblock_selected_cta_e4m3_mov_rtx_vs_rx9070xt",
        "status": "PASS",
        "classification": "POSTBLOCK_PRE_FP8_FIRST_DIVERGENCE_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "bitwise_exact": rtx == amd, "families": families,
        "first_causal_mismatch": first_causal,
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
