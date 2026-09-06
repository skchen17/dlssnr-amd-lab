#!/usr/bin/env python3
"""Apply a narrow, diagnostic segment correction to the first N0 cosine."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


TARGET = re.compile(r"cos\.approx\.ftz\.f32\s+(?P<dst>%r229)\s*,\s*(?P<src>%r226)\s*;")
TARGETS = {
    12: (3252, 2530),
    14: (13010, 10122),
    16: (52041, 40491),
    18: (208167, 161967),
}
POSITIVE_CORRECTION_U32 = 0x33C00000  # +8.940696716308594e-08
NEGATIVE_CORRECTION_U32 = 0xB3800000  # -5.960464477539063e-08


def f32_u32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def lower(text: str, segment_bits: int) -> tuple[str, int, dict]:
    if segment_bits not in TARGETS:
        raise ValueError(f"unsupported segment width: {segment_bits}")
    segments = 1 << segment_bits
    positive_segment, negative_segment = TARGETS[segment_bits]
    scale_u32 = f32_u32(segments / (2.0 * 3.141592653589793))
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        index = count
        count += 1
        prefix = f"__n0_cos_target_{index}"
        dst = match.group("dst")
        src = match.group("src")
        return "\n".join(
            (
                match.group(0),
                "{",
                f".reg .pred %{prefix}_positive, %{prefix}_negative;",
                f".reg .u32 %{prefix}_segment;",
                f".reg .f32 %{prefix}_position;",
                f"mul.rn.f32 %{prefix}_position, {src}, 0f{scale_u32:08X};",
                f"cvt.rzi.u32.f32 %{prefix}_segment, %{prefix}_position;",
                f"setp.eq.u32 %{prefix}_positive, %{prefix}_segment, {positive_segment};",
                f"setp.eq.u32 %{prefix}_negative, %{prefix}_segment, {negative_segment};",
                f"@%{prefix}_positive add.rn.f32 {dst}, {dst}, 0f{POSITIVE_CORRECTION_U32:08X};",
                f"@%{prefix}_negative add.rn.f32 {dst}, {dst}, 0f{NEGATIVE_CORRECTION_U32:08X};",
                "}",
            )
        )

    lowered = TARGET.sub(replace, text)
    metadata = {
        "segment_bits": segment_bits,
        "segments": segments,
        "scale_u32": f"0x{scale_u32:08X}",
        "positive_segment": positive_segment,
        "positive_correction_u32": f"0x{POSITIVE_CORRECTION_U32:08X}",
        "negative_segment": negative_segment,
        "negative_correction_u32": f"0x{NEGATIVE_CORRECTION_U32:08X}",
    }
    return lowered, count, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--segment-bits", type=int, choices=sorted(TARGETS), required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    text, count, metadata = lower(source.decode("utf-8"), args.segment_bits)
    output = text.encode("utf-8")
    passed = count == 1
    report = {
        "schema": 1,
        "experiment": "ptx_n0_cos_target_segment_ablation",
        "status": "PASS" if passed else "FAIL",
        "classification": "DIAGNOSTIC_OVERFIT_NOT_GENERAL_COS_MODEL",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "cos_sites_corrected": count,
        **metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
