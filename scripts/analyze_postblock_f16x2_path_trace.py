#!/usr/bin/env python3
"""Compare the low-perturbation r816/r817/r944 packed-f16 path trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_postblock_f16x2_path_trace import EXPECTED, TRACE_BYTES
except ModuleNotFoundError:
    from instrument_postblock_f16x2_path_trace import EXPECTED, TRACE_BYTES


LANES = 32
RECORD_BYTES = 20


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES}: RTX={len(rtx)}, AMD={len(amd)}")
    operations = []
    first_input_mismatch = None
    first_causal_mismatch = None
    for operation, (kind, dst, sources) in enumerate(EXPECTED):
        input_mismatches = output_mismatches = exact_input_output_diff = 0
        for lane in range(LANES):
            offset = (operation * LANES + lane) * RECORD_BYTES
            r = struct.unpack_from("<IIIII", rtx, offset)
            a = struct.unpack_from("<IIIII", amd, offset)
            expected_marker = 1 if kind == "mul" else 2
            if r[4] != expected_marker or a[4] != expected_marker:
                raise ValueError(f"invalid kind marker at operation {operation}, lane {lane}")
            differing = [source for source in range(len(sources)) if r[source] != a[source]]
            input_mismatches += len(differing)
            if differing and first_input_mismatch is None:
                source = differing[0]
                first_input_mismatch = {
                    "operation": operation, "kind": kind, "dst": dst,
                    "lane": lane, "source": source,
                    "source_register": sources[source],
                    "rtx_input": f"0x{r[source]:08X}",
                    "amd_input": f"0x{a[source]:08X}",
                }
            if r[3] != a[3]:
                output_mismatches += 1
                if not differing:
                    exact_input_output_diff += 1
                    if first_causal_mismatch is None:
                        first_causal_mismatch = {
                            "operation": operation, "kind": kind, "dst": dst,
                            "lane": lane,
                            "inputs": [f"0x{value:08X}" for value in r[:len(sources)]],
                            "rtx_output": f"0x{r[3]:08X}",
                            "amd_output": f"0x{a[3]:08X}",
                        }
        operations.append({
            "operation": operation, "kind": kind, "dst": dst,
            "sources": list(sources), "input_word_mismatches": input_mismatches,
            "output_word_mismatches": output_mismatches,
            "input_exact_output_diff_records": exact_input_output_diff,
        })
    return {
        "schema": 1,
        "experiment": "postblock_selected_cta_f16x2_r944_path_rtx_vs_rx9070xt",
        "status": "PASS",
        "classification": "LOW_PERTURBATION_POSTBLOCK_R944_CAUSAL_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "bitwise_exact": rtx == amd, "operations": operations,
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
