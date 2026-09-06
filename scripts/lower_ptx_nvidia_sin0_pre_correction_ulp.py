#!/usr/bin/env python3
"""Insert a phase-segmented ULP shift immediately after N0's first sine."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGET = re.compile(r"sin\.approx\.ftz\.f32\s+(?P<dst>%r228)\s*,\s*(?P<src>%r226)\s*;")


def lower(text: str, model: dict) -> tuple[str, int]:
    shifts, segments, scale = model.get("ulp_shifts_i32"), model.get("segments"), model.get("scale_u32")
    if not isinstance(segments, int) or segments <= 0 or segments & (segments - 1):
        raise ValueError("segments must be a power of two")
    if not isinstance(shifts, list) or len(shifts) != segments:
        raise ValueError("invalid shift table")
    words = [int(value) & 0xFFFFFFFF for value in shifts]
    symbol = "__n0_nvidia_sin0_pre_ulp_shift"
    declaration = (f".const .align 4 .u32 {symbol}[{segments}] = {{\n  " +
        ",\n  ".join(", ".join(f"0x{v:08X}" for v in words[i:i+8])
                       for i in range(0, segments, 8)) + "\n};\n\n")
    position = text.find(".visible .entry ")
    if position < 0:
        raise ValueError("PTX entry declaration not found")
    text = text[:position] + declaration + text[position:]
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        prefix = f"__n0_sin_pre_ulp_{count}"
        count += 1
        dst, src = match.group("dst"), match.group("src")
        return "\n".join((match.group(0), "{", f".reg .pred %{prefix}_nonzero;",
            f".reg .b32 %{prefix}_seg, %{prefix}_shift, %{prefix}_bits;",
            f".reg .f32 %{prefix}_position;", f".reg .b64 %{prefix}_base, %{prefix}_offset, %{prefix}_address;",
            f"mul.rn.f32 %{prefix}_position, {src}, 0f{scale:08X};",
            f"cvt.rzi.u32.f32 %{prefix}_seg, %{prefix}_position;",
            f"min.u32 %{prefix}_seg, %{prefix}_seg, {segments - 1};",
            f"mov.u64 %{prefix}_base, {symbol};", f"mul.wide.u32 %{prefix}_offset, %{prefix}_seg, 4;",
            f"add.u64 %{prefix}_address, %{prefix}_base, %{prefix}_offset;",
            f"ld.const.b32 %{prefix}_shift, [%{prefix}_address];",
            f"setp.ne.s32 %{prefix}_nonzero, %{prefix}_shift, 0;",
            f"@%{prefix}_nonzero mov.b32 %{prefix}_bits, {dst};",
            f"@%{prefix}_nonzero add.s32 %{prefix}_bits, %{prefix}_bits, %{prefix}_shift;",
            f"@%{prefix}_nonzero mov.b32 {dst}, %{prefix}_bits;", "}"))
    return TARGET.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path); parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    text, count = lower(args.input.read_text(encoding="utf-8"),
                        json.loads(args.model.read_text(encoding="utf-8")))
    args.output.write_text(text, encoding="utf-8", newline="\n")
    args.report.write_text(json.dumps({"schema": 1, "status": "PASS" if count == 1 else "FAIL",
                                      "sine_sites_corrected": count}, indent=2) + "\n", encoding="utf-8")
    return 0 if count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
