#!/usr/bin/env python3
"""Compare RTX and RX checkpoints along N0's first divergent activation path."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_n0_norm_path_trace import (
        BYTES_PER_STAGE_LANE,
        LANES,
        STAGES,
        TRACE_BYTES,
    )
except ModuleNotFoundError:
    from instrument_n0_norm_path_trace import BYTES_PER_STAGE_LANE, LANES, STAGES, TRACE_BYTES


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze(rtx_path: Path, amd_path: Path, *, stages_layout=STAGES,
            trace_bytes: int = TRACE_BYTES) -> dict:
    rtx = rtx_path.read_bytes()
    amd = amd_path.read_bytes()
    if len(rtx) != trace_bytes or len(amd) != trace_bytes:
        raise ValueError(f"trace size must be {trace_bytes} bytes")
    stages = []
    first_divergent_stage = None
    for stage, (name, _, register) in enumerate(stages_layout):
        base = stage * LANES * BYTES_PER_STAGE_LANE
        byte_mismatches = word_mismatches = half_mismatches = 0
        mismatches = []
        for lane in range(LANES):
            offset = base + lane * BYTES_PER_STAGE_LANE
            r_bytes = rtx[offset:offset + 4]
            a_bytes = amd[offset:offset + 4]
            byte_mismatches += sum(left != right for left, right in zip(r_bytes, a_bytes))
            r_word = struct.unpack("<I", r_bytes)[0]
            a_word = struct.unpack("<I", a_bytes)[0]
            if r_word != a_word:
                word_mismatches += 1
                r_halves = struct.unpack("<2H", r_bytes)
                a_halves = struct.unpack("<2H", a_bytes)
                half_mismatches += sum(left != right for left, right in zip(r_halves, a_halves))
                mismatches.append({
                    "lane": lane,
                    "rtx_u32_hex": f"{r_word:08X}",
                    "amd_u32_hex": f"{a_word:08X}",
                    "rtx_half_hex": [f"{value:04X}" for value in r_halves],
                    "amd_half_hex": [f"{value:04X}" for value in a_halves],
                })
        if word_mismatches and first_divergent_stage is None:
            first_divergent_stage = stage
        stages.append({
            "index": stage,
            "name": name,
            "register": register,
            "bitwise_gate": word_mismatches == 0,
            "byte_mismatches": byte_mismatches,
            "word_mismatches": word_mismatches,
            "half_mismatches": half_mismatches,
            "mismatches": mismatches,
        })
    return {
        "schema": 1,
        "experiment": "n0_normalization_path_rtx_vs_rx9070xt",
        "status": "PASS" if first_divergent_stage is None else "LOCALIZED",
        "classification": "FIRST_DIVERGENT_ACTIVATION_PATH_LOCALIZATION",
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
        "trace_bytes": trace_bytes,
        "stage_count": len(stages_layout),
        "first_divergent_stage": first_divergent_stage,
        "first_divergent_stage_name": (
            stages_layout[first_divergent_stage][0] if first_divergent_stage is not None else None
        ),
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", required=True, type=Path)
    parser.add_argument("--amd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.rtx, args.amd)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "first_divergent_stage": report["first_divergent_stage"],
        "first_divergent_stage_name": report["first_divergent_stage_name"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
