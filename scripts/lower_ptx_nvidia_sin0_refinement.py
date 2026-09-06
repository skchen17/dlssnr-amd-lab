#!/usr/bin/env python3
"""Append a sparse phase refinement after the accepted N0 sine correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = (
    "@%__n0_sin_fix_0_nonzero add.rn.f32 %r228, %r228, %__n0_sin_fix_0_q;\n}"
)


def lower(text: str, model: dict) -> tuple[str, int]:
    rows = model.get("coefficients_u32")
    segments, scale = model.get("segments"), model.get("scale_u32")
    if not isinstance(segments, int) or segments <= 0 or segments & (segments - 1):
        raise ValueError("segments must be a power of two")
    if not isinstance(rows, list) or len(rows) != segments or any(
        not isinstance(row, list) or len(row) != 1 for row in rows
    ):
        raise ValueError("invalid degree-zero coefficient table")
    words = [int(row[0]) & 0xFFFFFFFF for row in rows]
    symbol = "__n0_nvidia_sin0_refinement"
    declaration = (f".const .align 4 .u32 {symbol}[{segments}] = {{\n  " +
        ",\n  ".join(", ".join(f"0x{value:08X}" for value in words[i:i + 8])
                       for i in range(0, segments, 8)) + "\n};\n\n")
    position = text.find(".visible .entry ")
    if position < 0:
        raise ValueError("PTX entry declaration not found")
    text = text[:position] + declaration + text[position:]
    if text.count(MARKER) != 1:
        return text, 0
    prefix = "__n0_sin_refine_0"
    block = "\n".join(("{", f".reg .pred %{prefix}_nonzero;",
        f".reg .b32 %{prefix}_seg, %{prefix}_cbits;",
        f".reg .f32 %{prefix}_position, %{prefix}_correction;",
        f".reg .b64 %{prefix}_base, %{prefix}_offset, %{prefix}_address;",
        f"mul.rn.f32 %{prefix}_position, %r226, 0f{scale:08X};",
        f"cvt.rzi.u32.f32 %{prefix}_seg, %{prefix}_position;",
        f"min.u32 %{prefix}_seg, %{prefix}_seg, {segments - 1};",
        f"mov.u64 %{prefix}_base, {symbol};",
        f"mul.wide.u32 %{prefix}_offset, %{prefix}_seg, 4;",
        f"add.u64 %{prefix}_address, %{prefix}_base, %{prefix}_offset;",
        f"ld.const.b32 %{prefix}_cbits, [%{prefix}_address];",
        f"mov.b32 %{prefix}_correction, %{prefix}_cbits;",
        f"setp.ne.f32 %{prefix}_nonzero, %{prefix}_correction, 0f00000000;",
        f"@%{prefix}_nonzero add.rn.f32 %r228, %r228, %{prefix}_correction;", "}"))
    return text.replace(MARKER, MARKER + "\n" + block), 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path); parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    text, count = lower(args.input.read_text(encoding="utf-8"),
                        json.loads(args.model.read_text(encoding="utf-8")))
    args.output.write_text(text, encoding="utf-8", newline="\n")
    args.report.write_text(json.dumps({"schema": 1, "status": "PASS" if count == 1 else "FAIL",
                                      "refinement_sites": count}, indent=2) + "\n", encoding="utf-8")
    return 0 if count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
