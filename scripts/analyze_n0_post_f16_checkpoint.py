#!/usr/bin/env python3
"""Compare the first 16 N0 f16 MMA result fragments across RTX and RX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FRAGMENT_BYTES = 4096
LANES = 32
MMA_OPERATIONS = 16
BYTES_PER_MMA_PER_LANE = 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def half_mismatches(left: bytes, right: bytes) -> int:
    return sum(left[i:i + 2] != right[i:i + 2] for i in range(0, len(left), 2))


def per_mma_half_mismatches(left: bytes, right: bytes) -> list[int]:
    counts = []
    lane_stride = MMA_OPERATIONS * BYTES_PER_MMA_PER_LANE
    for mma in range(MMA_OPERATIONS):
        count = 0
        for lane in range(LANES):
            start = lane * lane_stride + mma * BYTES_PER_MMA_PER_LANE
            count += half_mismatches(
                left[start:start + BYTES_PER_MMA_PER_LANE],
                right[start:start + BYTES_PER_MMA_PER_LANE],
            )
        counts.append(count)
    return counts


def analyze(rtx_dir: Path, amd_dir: Path) -> dict:
    rtx = (rtx_dir / "post_f16_fragments.raw").read_bytes()
    amd = (amd_dir / "scratch.raw").read_bytes()[:FRAGMENT_BYTES]
    if len(rtx) != FRAGMENT_BYTES or len(amd) != FRAGMENT_BYTES:
        raise ValueError(f"expected two {FRAGMENT_BYTES}-byte fragments")
    manifest = json.loads((rtx_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    integrity = (
        manifest.get("status") == "PASS"
        and manifest.get("device_name") == "NVIDIA GeForce RTX 5070"
        and manifest.get("fragments_sha256") == sha256(rtx)
    )
    byte_count = sum(a != b for a, b in zip(rtx, amd))
    half_count = half_mismatches(rtx, amd)
    by_mma = per_mma_half_mismatches(rtx, amd)
    status = "PASS" if integrity and byte_count == 0 and half_count == 0 else "FAIL"
    return {
        "schema": 1,
        "experiment": "n0_post_first_16_f16_mma_rtx5070_vs_rx9070xt",
        "status": status,
        "classification": "BITWISE_F16_MMA_FRAGMENT_PARITY",
        "counts_as_s6": False,
        "rtx_integrity": integrity,
        "bytes": FRAGMENT_BYTES,
        "half_values": FRAGMENT_BYTES // 2,
        "lanes": LANES,
        "mma_operations": MMA_OPERATIONS,
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
        "byte_mismatches": byte_count,
        "half_mismatches": half_count,
        "per_mma_half_mismatches": by_mma,
        "closed_boundary": "first 16 native f16 MMA operations after texture/weight fragment parity",
        "next_gate": "regenerate and compare the complete 740-operation N0 output",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rtx_dir", type=Path)
    parser.add_argument("amd_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.rtx_dir.resolve(), args.amd_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
