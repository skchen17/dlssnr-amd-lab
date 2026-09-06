#!/usr/bin/env python3
"""Capture sqrt0 and consumed normal0 FP16 for every N0 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GRID_X = 80
GRID_Y = 48
SAMPLES_PER_CTA = 64
SAMPLES = GRID_X * GRID_Y * SAMPLES_PER_CTA
TRACE_OFFSET = 384 * 640 * 32
BYTES_PER_SAMPLE = 8
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE
SQRT = re.compile(r"sqrt\.approx\.ftz\.f32\s+%r221\s*,\s*%r220\s*;")
NORMAL_F16 = re.compile(r"cvt\.rn\.f16\.f32\s+%rs15\s*,\s*%r149\s*;")


def store_block(source: str, field_offset: int, half: bool, tag: str) -> str:
    prefix = f"__n0_normal0_grid_{tag}"
    source_register = source
    lines = [
        "{",
        f".reg .b32 %{prefix}_ctax, %{prefix}_ctay, %{prefix}_linear, "
        f"%{prefix}_sample, %{prefix}_offset;",
        f".reg .b64 %{prefix}_scratch, %{prefix}_wide, %{prefix}_address;",
    ]
    if half:
        source_register = f"%{prefix}_half"
        lines.append(f".reg .b32 {source_register};")
    lines.extend((
        f"mov.u32 %{prefix}_ctax, %ctaid.x;",
        f"mov.u32 %{prefix}_ctay, %ctaid.y;",
        f"mad.lo.u32 %{prefix}_linear, %{prefix}_ctay, {GRID_X}, %{prefix}_ctax;",
        f"mad.lo.u32 %{prefix}_sample, %{prefix}_linear, {SAMPLES_PER_CTA}, %r5556;",
        f"mad.lo.u32 %{prefix}_offset, %{prefix}_sample, {BYTES_PER_SAMPLE}, "
        f"{TRACE_OFFSET + field_offset};",
        f"ld.param.b64 %{prefix}_scratch, [%rd16+216];",
        f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
        f"add.s64 %{prefix}_address, %{prefix}_scratch, %{prefix}_wide;",
    ))
    if half:
        lines.append(f"cvt.u32.u16 {source_register}, {source};")
    lines.extend((f"st.global.b32 [%{prefix}_address], {source_register};", "}"))
    return "\n".join(lines)


def instrument(text: str) -> tuple[str, dict[str, int]]:
    text, sqrt_count = SQRT.subn(
        lambda match: match.group(0) + "\n" + store_block("%r221", 0, False, "sqrt"), text
    )
    text, normal_count = NORMAL_F16.subn(
        lambda match: match.group(0) + "\n" + store_block("%rs15", 4, True, "normal"), text
    )
    return text, {"sqrt0": sqrt_count, "normal0_f16": normal_count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, counts = instrument(source.decode("utf-8"))
    output = output_text.encode("utf-8")
    passed = counts == {"sqrt0": 1, "normal0_f16": 1}
    report = {
        "schema": 1,
        "experiment": "n0_normal0_boundary_full_grid_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "N0_COSINE_TO_CONSUMED_FP16_FULL_GRID_DATASET",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "grid": [GRID_X, GRID_Y, 1],
        "block": [32, 1, 1],
        "sample_count": SAMPLES,
        "samples_per_cta": SAMPLES_PER_CTA,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "fields": ["sqrt0_f32", "normal0_f16_as_u32"],
        "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "scratch_extra_bytes": TRACE_BYTES,
        "insertion_counts": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
