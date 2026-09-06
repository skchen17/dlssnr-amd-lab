#!/usr/bin/env python3
"""Apply phase-segmented joint raw-cos0 and post-correction sqrt0 ULP shifts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGET = re.compile(r"cos\.approx\.ftz\.f32\s+(?P<cos>%r229)\s*,\s*(?P<phase>%r226)\s*;")


def lower(text: str, model: dict) -> tuple[str, int]:
    segments, scale = model.get("segments"), model.get("scale_u32")
    cos_shifts = model.get("cos0_raw_ulp_shifts_i32")
    sqrt_shifts = model.get("sqrt0_post_ulp_shifts_i32")
    if not isinstance(segments, int) or segments <= 0 or segments & (segments - 1):
        raise ValueError("segments must be a power of two")
    if (not isinstance(cos_shifts, list) or len(cos_shifts) != segments or
            not isinstance(sqrt_shifts, list) or len(sqrt_shifts) != segments):
        raise ValueError("invalid joint shift tables")
    words = [value for pair in zip(cos_shifts, sqrt_shifts) for value in pair]
    symbol = "__n0_cos0_sqrt0_joint_ulp"
    declaration = (f".const .align 4 .u32 {symbol}[{len(words)}] = {{\n  " +
        ",\n  ".join(", ".join(f"0x{(int(v) & 0xFFFFFFFF):08X}" for v in words[i:i + 8])
                       for i in range(0, len(words), 8)) + "\n};\n\n")
    position = text.find(".visible .entry ")
    if position < 0:
        raise ValueError("PTX entry declaration not found")
    text = text[:position] + declaration + text[position:]
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        prefix = f"__n0_joint_ulp_{count}"
        count += 1
        cos, phase = match.group("cos"), match.group("phase")
        return "\n".join((match.group(0), "{",
            f".reg .pred %{prefix}_cos_nonzero, %{prefix}_sqrt_nonzero;",
            f".reg .b32 %{prefix}_seg, %{prefix}_cos_shift, %{prefix}_sqrt_shift, %{prefix}_bits;",
            f".reg .f32 %{prefix}_position;",
            f".reg .b64 %{prefix}_base, %{prefix}_offset, %{prefix}_address;",
            f"mul.rn.f32 %{prefix}_position, {phase}, 0f{int(scale):08X};",
            f"cvt.rzi.u32.f32 %{prefix}_seg, %{prefix}_position;",
            f"min.u32 %{prefix}_seg, %{prefix}_seg, {segments - 1};",
            f"mov.u64 %{prefix}_base, {symbol};",
            f"mul.wide.u32 %{prefix}_offset, %{prefix}_seg, 8;",
            f"add.u64 %{prefix}_address, %{prefix}_base, %{prefix}_offset;",
            f"ld.const.b32 %{prefix}_cos_shift, [%{prefix}_address];",
            f"ld.const.b32 %{prefix}_sqrt_shift, [%{prefix}_address+4];",
            f"setp.ne.s32 %{prefix}_cos_nonzero, %{prefix}_cos_shift, 0;",
            f"@%{prefix}_cos_nonzero mov.b32 %{prefix}_bits, {cos};",
            f"@%{prefix}_cos_nonzero add.s32 %{prefix}_bits, %{prefix}_bits, %{prefix}_cos_shift;",
            f"@%{prefix}_cos_nonzero mov.b32 {cos}, %{prefix}_bits;",
            f"setp.ne.s32 %{prefix}_sqrt_nonzero, %{prefix}_sqrt_shift, 0;",
            f"@%{prefix}_sqrt_nonzero mov.b32 %{prefix}_bits, %r221;",
            f"@%{prefix}_sqrt_nonzero add.s32 %{prefix}_bits, %{prefix}_bits, %{prefix}_sqrt_shift;",
            f"@%{prefix}_sqrt_nonzero mov.b32 %r221, %{prefix}_bits;", "}"))

    return TARGET.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    text, count = lower(args.input.read_text(encoding="utf-8"),
                        json.loads(args.model.read_text(encoding="utf-8")))
    args.output.write_text(text, encoding="utf-8", newline="\n")
    args.report.write_text(json.dumps({
        "schema": 1, "status": "PASS" if count == 1 else "FAIL",
        "joint_sites_corrected": count,
    }, indent=2) + "\n", encoding="utf-8")
    return 0 if count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
