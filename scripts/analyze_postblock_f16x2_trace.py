#!/usr/bin/env python3
"""Compare RTX and AMD packed-f16 post-block traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_postblock_f16x2_trace import (
        KIND_CODES, LANES, OPERATION_COUNT, RECORD_BYTES, TRACE_BYTES,
    )
except ModuleNotFoundError:
    from instrument_postblock_f16x2_trace import (
        KIND_CODES, LANES, OPERATION_COUNT, RECORD_BYTES, TRACE_BYTES,
    )


CODE_KINDS = {value: key for key, value in KIND_CODES.items()}
ARITY = {"mul": 2, "add": 2, "min": 2, "max": 2, "fma": 3, "abs": 1}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES}: RTX={len(rtx)}, AMD={len(amd)}")
    input_word_mismatches = 0
    output_word_mismatches = 0
    exact_input_output_diff_records = 0
    first_input_mismatch = None
    first_causal_mismatch = None
    operation_kinds: list[str] = []
    per_kind = {kind: {"operation_count": 0, "input_word_mismatches": 0,
                       "output_word_mismatches": 0,
                       "input_exact_output_diff_records": 0}
                for kind in KIND_CODES}
    for operation in range(OPERATION_COUNT):
        operation_kind = None
        for lane in range(LANES):
            offset = (operation * LANES + lane) * RECORD_BYTES
            r_record = struct.unpack_from("<IIIII", rtx, offset)
            a_record = struct.unpack_from("<IIIII", amd, offset)
            if r_record[4] != a_record[4]:
                raise ValueError(f"kind marker mismatch at operation {operation}, lane {lane}")
            kind = CODE_KINDS.get(r_record[4])
            if kind is None:
                raise ValueError(f"invalid kind marker {r_record[4]} at operation {operation}, lane {lane}")
            if operation_kind is None:
                operation_kind = kind
                per_kind[kind]["operation_count"] += 1
            elif operation_kind != kind:
                raise ValueError(f"kind marker varies by lane at operation {operation}")
            arity = ARITY[kind]
            differences = [index for index in range(arity) if r_record[index] != a_record[index]]
            input_word_mismatches += len(differences)
            per_kind[kind]["input_word_mismatches"] += len(differences)
            if differences and first_input_mismatch is None:
                source = differences[0]
                first_input_mismatch = {
                    "operation": operation, "kind": kind, "lane": lane,
                    "source": source,
                    "rtx_input": f"0x{r_record[source]:08X}",
                    "amd_input": f"0x{a_record[source]:08X}",
                }
            if r_record[3] != a_record[3]:
                output_word_mismatches += 1
                per_kind[kind]["output_word_mismatches"] += 1
                if not differences:
                    exact_input_output_diff_records += 1
                    per_kind[kind]["input_exact_output_diff_records"] += 1
                    if first_causal_mismatch is None:
                        first_causal_mismatch = {
                            "operation": operation, "kind": kind, "lane": lane,
                            "inputs": [f"0x{value:08X}" for value in r_record[:arity]],
                            "rtx_output": f"0x{r_record[3]:08X}",
                            "amd_output": f"0x{a_record[3]:08X}",
                        }
        operation_kinds.append(operation_kind or "unknown")
    return {
        "schema": 1,
        "experiment": "postblock_selected_cta_f16x2_rtx_vs_rx9070xt",
        "status": "PASS",
        "classification": "POSTBLOCK_PRE_E4M3_FIRST_DIVERGENCE_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "bitwise_exact": rtx == amd,
        "operation_count": OPERATION_COUNT,
        "input_word_mismatches": input_word_mismatches,
        "output_word_mismatches": output_word_mismatches,
        "input_exact_output_diff_records": exact_input_output_diff_records,
        "first_input_mismatch": first_input_mismatch,
        "first_causal_mismatch": first_causal_mismatch,
        "per_kind": per_kind,
        "operation_kinds": operation_kinds,
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
