#!/usr/bin/env python3
"""Verify RTX/RX parity at the first N0 pre-MMA A/B fragment checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def mismatches(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("fragment lengths differ")
    return sum(a != b for a, b in zip(left, right))


def split_fragments(raw: bytes) -> tuple[bytes, bytes]:
    if len(raw) != 3072:
        raise ValueError(f"expected 3072 bytes, got {len(raw)}")
    a = bytearray()
    b = bytearray()
    for lane in range(32):
        base = lane * 96
        a.extend(raw[base:base + 64])
        b.extend(raw[base + 64:base + 96])
    return bytes(a), bytes(b)


def analyze(root: Path) -> dict:
    rtx_dir = root / "results/20260831_132553_rtx5070_n0_pre_mma_checkpoint"
    amd_dir = root / "results/20260831_133700_amd_n0_pre_mma_checkpoint"
    rtx = (rtx_dir / "pre_mma_fragments.raw").read_bytes()
    amd = (amd_dir / "scratch.raw").read_bytes()[:3072]
    rtx_a, rtx_b = split_fragments(rtx)
    amd_a, amd_b = split_fragments(amd)
    total = mismatches(rtx, amd)
    a_count = mismatches(rtx_a, amd_a)
    b_count = mismatches(rtx_b, amd_b)
    manifest = json.loads((rtx_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    integrity = (
        manifest.get("status") == "PASS"
        and manifest.get("device_name") == "NVIDIA GeForce RTX 5070"
        and manifest.get("fragments_sha256") == sha256(rtx)
    )
    status = "PASS" if integrity and total == 0 and a_count == 0 and b_count == 0 else "FAIL"
    return {
        "schema": 1,
        "experiment": "n0_pre_mma_rtx5070_vs_rx9070xt",
        "status": status,
        "classification": "BITWISE_PRE_MMA_INPUT_AND_WEIGHT_PARITY",
        "counts_as_s6": False,
        "rtx_integrity": integrity,
        "bytes": len(rtx),
        "lanes": 32,
        "bytes_per_lane": 96,
        "rtx_sha256": sha256(rtx),
        "amd_sha256": sha256(amd),
        "total_byte_mismatches": total,
        "input_a": {"bytes": len(rtx_a), "byte_mismatches": a_count},
        "weights_b": {"bytes": len(rtx_b), "byte_mismatches": b_count},
        "closed_boundaries": [
            "texture sampling and input preprocessing through first shared-memory fragments",
            "weight pointer relocation and first weight-fragment loads",
            "lane/register layout immediately before first f16 MMA",
        ],
        "next_gate": "export the 4,096-byte D fragments immediately after the first 16 f16 MMA operations",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
