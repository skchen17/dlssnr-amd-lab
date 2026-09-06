#!/usr/bin/env python3
"""Apply a sparse per-segment ULP adjustment after N0's first sqrt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGETS = {
    "first": re.compile(r"sqrt\.approx\.ftz\.f32\s+(?P<dst>%r221)\s*,\s*(?P<src>%r220)\s*;"),
    "second": re.compile(r"sqrt\.approx\.ftz\.f32\s+(?P<dst>%r225)\s*,\s*(?P<src>%r224)\s*;"),
}


def lower(text: str, model: dict, site: str = "first") -> tuple[str, int]:
    shifts = model.get("ulp_shifts_i32")
    segments, scale = model.get("segments"), model.get("scale_u32")
    if not isinstance(segments, int) or segments <= 0 or segments & (segments - 1):
        raise ValueError("model segments must be a power of two")
    if not isinstance(shifts, list) or len(shifts) != segments:
        raise ValueError("invalid shift table")
    words = [int(value) & 0xFFFFFFFF for value in shifts]
    if site not in TARGETS:
        raise ValueError(f"unknown sqrt site: {site}")
    symbol = "__n0_nvidia_sqrt0_ulp_shift" if site == "first" else "__n0_nvidia_sqrt1_ulp_shift"
    declaration = (f".const .align 4 .u32 {symbol}[{len(words)}] = {{\n  "
                   + ",\n  ".join(", ".join(f"0x{value:08X}" for value in words[i:i+8])
                                      for i in range(0, len(words), 8)) + "\n};\n\n")
    position = text.find(".visible .entry ")
    if position < 0:
        raise ValueError("PTX entry declaration not found")
    text = text[:position] + declaration + text[position:]
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        prefix = f"__n0_sqrt_ulp_{count}"
        count += 1
        dst = match.group("dst")
        return "\n".join((match.group(0), "{", f".reg .pred %{prefix}_nonzero;",
            f".reg .b32 %{prefix}_seg, %{prefix}_shift, %{prefix}_bits;",
            f".reg .f32 %{prefix}_position;",
            f".reg .b64 %{prefix}_base, %{prefix}_offset, %{prefix}_address;",
            f"mul.rn.f32 %{prefix}_position, {dst}, 0f{scale:08X};",
            f"cvt.rzi.u32.f32 %{prefix}_seg, %{prefix}_position;",
            f"min.u32 %{prefix}_seg, %{prefix}_seg, {segments - 1};",
            f"mov.u64 %{prefix}_base, {symbol};",
            f"mul.wide.u32 %{prefix}_offset, %{prefix}_seg, 4;",
            f"add.u64 %{prefix}_address, %{prefix}_base, %{prefix}_offset;",
            f"ld.const.b32 %{prefix}_shift, [%{prefix}_address];",
            f"setp.ne.s32 %{prefix}_nonzero, %{prefix}_shift, 0;",
            f"@%{prefix}_nonzero mov.b32 %{prefix}_bits, {dst};",
            f"@%{prefix}_nonzero add.s32 %{prefix}_bits, %{prefix}_bits, %{prefix}_shift;",
            f"@%{prefix}_nonzero mov.b32 {dst}, %{prefix}_bits;", "}"))

    return TARGETS[site].sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--site", choices=sorted(TARGETS), default="first")
    args = parser.parse_args()
    source = args.input.read_bytes()
    text, count = lower(source.decode("utf-8"), json.loads(args.model.read_text(encoding="utf-8")), args.site)
    output = text.encode("utf-8")
    report = {"schema": 1, "experiment": "ptx_n0_nvidia_sqrt0_ulp_correction_lowering",
              "status": "PASS" if count == 1 else "FAIL",
              "source_sha256": hashlib.sha256(source).hexdigest().upper(),
              "output_sha256": hashlib.sha256(output).hexdigest().upper(),
              "sqrt_site": args.site,
              "sqrt_sites_corrected": count}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
