#!/usr/bin/env python3
"""Compare selected-CTA post-block MMA register traces bitwise."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_postblock_mma_trace import (
        F16_COUNT, FP8_COUNT, LANES, RECORD_BYTES, TRACE_BYTES,
    )
except ModuleNotFoundError:
    from instrument_postblock_mma_trace import (
        F16_COUNT, FP8_COUNT, LANES, RECORD_BYTES, TRACE_BYTES,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES} bytes")
    families = []
    first_mismatch = None
    first_output_mismatch_with_local_record_inputs_exact = None
    first_output_mismatch_with_warp_inputs_exact = None
    for name, start, count in (("fp8", 0, FP8_COUNT),
                               ("f16", FP8_COUNT, F16_COUNT)):
        input_words = output_words = input_exact = output_exact = 0
        active_records = 0
        family_first = None
        warp_input_exact_mma_count = 0
        warp_input_exact_output_diff_mma_count = 0
        for mma in range(count):
            mma_records = []
            for lane in range(LANES):
                offset = (start + mma) * LANES * RECORD_BYTES + lane * RECORD_BYTES
                rw = struct.unpack_from("<10I", rtx, offset)
                aw = struct.unpack_from("<10I", amd, offset)
                mma_records.append((lane, rw, aw))
                if any(rw) or any(aw):
                    active_records += 1
                inputs_equal = rw[:8] == aw[:8]
                input_words += 8
                output_words += 2
                input_exact += sum(a == b for a, b in zip(rw[:8], aw[:8]))
                output_exact += sum(a == b for a, b in zip(rw[8:], aw[8:]))
                if rw != aw:
                    mismatch = {
                        "family": name,
                        "mma": mma,
                        "lane": lane,
                        "input_words_exact": inputs_equal,
                        "rtx": [f"0x{x:08X}" for x in rw],
                        "amd": [f"0x{x:08X}" for x in aw],
                    }
                    if family_first is None:
                        family_first = mismatch
                    if first_mismatch is None:
                        first_mismatch = mismatch
                    if inputs_equal and rw[8:] != aw[8:] and \
                            first_output_mismatch_with_local_record_inputs_exact is None:
                        first_output_mismatch_with_local_record_inputs_exact = mismatch
            warp_inputs_exact = all(rw[:8] == aw[:8] for _, rw, aw in mma_records)
            if warp_inputs_exact:
                warp_input_exact_mma_count += 1
                output_mismatch = next(
                    ((lane, rw, aw) for lane, rw, aw in mma_records if rw[8:] != aw[8:]),
                    None)
                if output_mismatch is not None:
                    warp_input_exact_output_diff_mma_count += 1
                    if first_output_mismatch_with_warp_inputs_exact is None:
                        lane, rw, aw = output_mismatch
                        first_output_mismatch_with_warp_inputs_exact = {
                            "family": name, "mma": mma, "lane": lane,
                            "rtx_output": [f"0x{x:08X}" for x in rw[8:]],
                            "amd_output": [f"0x{x:08X}" for x in aw[8:]],
                        }
        families.append({
            "name": name,
            "mma_count": count,
            "active_records": active_records,
            "input_word_exact_fraction": input_exact / input_words,
            "output_word_exact_fraction": output_exact / output_words,
            "warp_input_exact_mma_count": warp_input_exact_mma_count,
            "warp_input_exact_output_diff_mma_count":
                warp_input_exact_output_diff_mma_count,
            "bitwise_exact": family_first is None,
            "first_mismatch": family_first,
        })
    return {
        "schema": 1,
        "experiment": "postblock_selected_cta_mma_trace_rtx_vs_rx9070xt",
        "status": "PASS",
        "classification": "POSTBLOCK_FP8_F16_MMA_FIRST_DIVERGENCE_LOCALIZATION",
        "trace_bytes": TRACE_BYTES,
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
        "bitwise_exact": rtx == amd,
        "families": families,
        "first_mismatch": first_mismatch,
        "first_output_mismatch_with_local_record_inputs_exact":
            first_output_mismatch_with_local_record_inputs_exact,
        "first_output_mismatch_with_warp_inputs_exact":
            first_output_mismatch_with_warp_inputs_exact,
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
