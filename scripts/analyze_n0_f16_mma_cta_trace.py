#!/usr/bin/env python3
"""Compare selected-CTA f16 MMA A/B/C/D fragment traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MMA_COUNT = 16
LANES = 32
LANE_BYTES = 40
PRE_BYTES = 32
TRACE_BYTES = MMA_COUNT * LANES * LANE_BYTES


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes) -> dict:
    if len(rtx) != TRACE_BYTES or len(amd) != TRACE_BYTES:
        raise ValueError(f"trace size must be {TRACE_BYTES} bytes")
    per_mma = []
    details = []
    first_pre = first_d = None
    total_pre = total_d_bytes = total_d_halves = 0
    for mma in range(MMA_COUNT):
        pre = d_bytes = d_halves = 0
        for lane in range(LANES):
            start = (mma * LANES + lane) * LANE_BYTES
            for offset in range(PRE_BYTES):
                if rtx[start + offset] != amd[start + offset]:
                    pre += 1
                    if len(details) < 128:
                        details.append({
                            "mma": mma, "lane": lane, "region": "ABC",
                            "register": ["A0", "A1", "A2", "A3", "B0", "B1", "C0", "C1"][offset // 4],
                            "byte_in_register": offset % 4,
                            "rtx": rtx[start + offset], "amd": amd[start + offset],
                        })
            for offset in range(8):
                if rtx[start + PRE_BYTES + offset] != amd[start + PRE_BYTES + offset]:
                    d_bytes += 1
                    if len(details) < 128:
                        details.append({
                            "mma": mma, "lane": lane, "region": "D",
                            "register": f"D{offset // 4}", "byte_in_register": offset % 4,
                            "rtx": rtx[start + PRE_BYTES + offset], "amd": amd[start + PRE_BYTES + offset],
                        })
            for offset in range(0, 8, 2):
                if rtx[start + PRE_BYTES + offset:start + PRE_BYTES + offset + 2] != amd[start + PRE_BYTES + offset:start + PRE_BYTES + offset + 2]:
                    d_halves += 1
        if pre and first_pre is None:
            first_pre = mma
        if d_halves and first_d is None:
            first_d = mma
        total_pre += pre
        total_d_bytes += d_bytes
        total_d_halves += d_halves
        per_mma.append({"mma": mma, "pre_fragment_byte_mismatches": pre,
                        "d_byte_mismatches": d_bytes, "d_half_mismatches": d_halves})
    return {
        "schema": 1,
        "experiment": "n0_selected_cta_f16_mma_rtx_vs_rx9070xt",
        "status": "PASS" if total_pre == 0 and total_d_halves == 0 else "FAIL",
        "classification": "FIRST_UPSTREAM_F16_MMA_DIVERGENCE_LOCALIZATION",
        "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
        "mma_count": MMA_COUNT,
        "pre_fragment_byte_mismatches": total_pre,
        "d_byte_mismatches": total_d_bytes,
        "d_half_mismatches": total_d_halves,
        "first_pre_fragment_mismatch_mma": first_pre,
        "first_d_mismatch_mma": first_d,
        "first_mismatch_details": details,
        "per_mma": per_mma,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_bytes(args.rtx.read_bytes(), args.amd.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
