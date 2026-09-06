#!/usr/bin/env python3
"""Apply a sparse RTX 5070 correction after N0's first sqrt approximation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGET = re.compile(r"sqrt\.approx\.ftz\.f32\s+(?P<dst>%r221)\s*,\s*(?P<src>%r220)\s*;")


def lower(text: str, model: dict) -> tuple[str, int]:
    rows = model.get("coefficients_u32")
    segments, degree, scale = model.get("segments"), model.get("degree"), model.get("scale_u32")
    width = degree + 1 if isinstance(degree, int) else 0
    if not isinstance(segments, int) or segments <= 0 or segments & (segments - 1):
        raise ValueError("model segments must be a power of two")
    if not isinstance(rows, list) or len(rows) != segments or any(
        not isinstance(row, list) or len(row) != width for row in rows
    ):
        raise ValueError("invalid coefficient layout")
    words = [value for row in rows for value in row]
    symbol = "__n0_nvidia_sqrt0_correction"
    declaration = (f".const .align 4 .u32 {symbol}[{len(words)}] = {{\n  "
                   + ",\n  ".join(", ".join(f"0x{words[i+j]:08X}" for j in range(width))
                                      for i in range(0, len(words), width)) + "\n};\n\n")
    position = text.find(".visible .entry ")
    if position < 0:
        raise ValueError("PTX entry declaration not found")
    text = text[:position] + declaration + text[position:]
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        prefix = f"__n0_sqrt_fix_{count}"
        count += 1
        dst = match.group("dst")
        cbits = [f"%{prefix}_c{i}b" for i in range(width)]
        coeffs = [f"%{prefix}_c{i}" for i in range(width)]
        lines = [match.group(0), "{", f".reg .pred %{prefix}_nonzero;",
                 f".reg .b32 %{prefix}_seg, " + ", ".join(cbits) + ";",
                 f".reg .f32 %{prefix}_position, %{prefix}_segf, %{prefix}_local, %{prefix}_q, " + ", ".join(coeffs) + ";",
                 f".reg .b64 %{prefix}_base, %{prefix}_offset, %{prefix}_address;",
                 f"mul.rn.f32 %{prefix}_position, {dst}, 0f{scale:08X};",
                 f"cvt.rzi.u32.f32 %{prefix}_seg, %{prefix}_position;",
                 f"min.u32 %{prefix}_seg, %{prefix}_seg, {segments - 1};",
                 f"cvt.rn.f32.u32 %{prefix}_segf, %{prefix}_seg;",
                 f"sub.rn.f32 %{prefix}_local, %{prefix}_position, %{prefix}_segf;",
                 f"mov.u64 %{prefix}_base, {symbol};",
                 f"mul.wide.u32 %{prefix}_offset, %{prefix}_seg, {width * 4};",
                 f"add.u64 %{prefix}_address, %{prefix}_base, %{prefix}_offset;"]
        for i in range(width):
            suffix = f"+{i * 4}" if i else ""
            lines.extend((f"ld.const.b32 {cbits[i]}, [%{prefix}_address{suffix}];",
                          f"mov.b32 {coeffs[i]}, {cbits[i]};"))
        lines.append(f"mov.f32 %{prefix}_q, {coeffs[-1]};")
        for i in range(degree - 1, -1, -1):
            lines.extend((f"mul.rn.f32 %{prefix}_q, %{prefix}_q, %{prefix}_local;",
                          f"add.rn.f32 %{prefix}_q, %{prefix}_q, {coeffs[i]};"))
        lines.extend((f"setp.ne.f32 %{prefix}_nonzero, %{prefix}_q, 0f00000000;",
                      f"@%{prefix}_nonzero add.rn.f32 {dst}, {dst}, %{prefix}_q;", "}"))
        return "\n".join(lines)

    return TARGET.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    text, count = lower(source.decode("utf-8"), json.loads(args.model.read_text(encoding="utf-8")))
    output = text.encode("utf-8")
    report = {"schema": 1, "experiment": "ptx_n0_nvidia_sqrt0_correction_lowering",
              "status": "PASS" if count == 1 else "FAIL",
              "classification": "RTX5070_EMPIRICAL_SQRT_SFU_CORRECTION_CANDIDATE",
              "source_sha256": hashlib.sha256(source).hexdigest().upper(),
              "output_sha256": hashlib.sha256(output).hexdigest().upper(),
              "sqrt_sites_corrected": count}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
