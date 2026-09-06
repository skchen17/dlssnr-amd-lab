#!/usr/bin/env python3
"""Compare RTX and RX checkpoints feeding zero-input N0 MMA 178 B1."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    from scripts.instrument_n0_mma178_b_path_trace import BYTES_PER_STAGE_LANE, LANES, STAGES, TRACE_BYTES
except ModuleNotFoundError:
    from instrument_n0_mma178_b_path_trace import BYTES_PER_STAGE_LANE, LANES, STAGES, TRACE_BYTES


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze(rtx_path: Path, amd_path: Path) -> dict:
    rtx, amd = rtx_path.read_bytes(), amd_path.read_bytes()
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES} bytes")
    stages, first = [], None
    for stage, (name, _, register) in enumerate(STAGES):
        base = stage * LANES * BYTES_PER_STAGE_LANE
        mismatches = []
        byte_mismatches = half_mismatches = 0
        for lane in range(LANES):
            offset = base + lane * 4
            rb, ab = rtx[offset:offset + 4], amd[offset:offset + 4]
            byte_mismatches += sum(x != y for x, y in zip(rb, ab))
            rw, aw = struct.unpack("<I", rb)[0], struct.unpack("<I", ab)[0]
            if rw != aw:
                rh, ah = struct.unpack("<2H", rb), struct.unpack("<2H", ab)
                half_mismatches += sum(x != y for x, y in zip(rh, ah))
                mismatches.append({
                    "lane": lane,
                    "rtx_u32_hex": f"{rw:08X}",
                    "amd_u32_hex": f"{aw:08X}",
                    "rtx_half_hex": [f"{x:04X}" for x in rh],
                    "amd_half_hex": [f"{x:04X}" for x in ah],
                })
        if mismatches and first is None:
            first = stage
        stages.append({
            "index": stage, "name": name, "register": register,
            "bitwise_gate": not mismatches, "byte_mismatches": byte_mismatches,
            "word_mismatches": len(mismatches), "half_mismatches": half_mismatches,
            "mismatches": mismatches,
        })
    return {
        "schema": 1,
        "experiment": "n0_mma178_b_path_rtx_vs_rx9070xt",
        "status": "PASS" if first is None else "LOCALIZED",
        "classification": "N0_ZERO_INPUT_MMA178_B1_PATH_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "trace_bytes": TRACE_BYTES, "stage_count": len(STAGES),
        "first_divergent_stage": first,
        "first_divergent_stage_name": STAGES[first][0] if first is not None else None,
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
    print(json.dumps({"status": report["status"], "first_divergent_stage_name": report["first_divergent_stage_name"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
