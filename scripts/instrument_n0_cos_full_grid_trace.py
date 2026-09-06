#!/usr/bin/env python3
"""Capture the first Box-Muller cosine input/output pair for every N0 sample."""

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
COS = re.compile(r"cos\.approx\.ftz\.f32\s+%r229\s*,\s*%r226\s*;")


def instrument(text: str) -> tuple[str, int]:
    prefix = "__n0_cos_grid_trace"
    block = "\n".join([
        "{",
        f".reg .b32 %{prefix}_ctax, %{prefix}_ctay, %{prefix}_linear, %{prefix}_sample, %{prefix}_offset;",
        f".reg .b64 %{prefix}_scratch, %{prefix}_wide, %{prefix}_address;",
        f"mov.u32 %{prefix}_ctax, %ctaid.x;",
        f"mov.u32 %{prefix}_ctay, %ctaid.y;",
        f"mad.lo.u32 %{prefix}_linear, %{prefix}_ctay, {GRID_X}, %{prefix}_ctax;",
        f"mad.lo.u32 %{prefix}_sample, %{prefix}_linear, {SAMPLES_PER_CTA}, %r5556;",
        f"mad.lo.u32 %{prefix}_offset, %{prefix}_sample, {BYTES_PER_SAMPLE}, {TRACE_OFFSET};",
        f"ld.param.b64 %{prefix}_scratch, [%rd16+216];",
        f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
        f"add.s64 %{prefix}_address, %{prefix}_scratch, %{prefix}_wide;",
        f"st.global.b32 [%{prefix}_address], %r226;",
        f"st.global.b32 [%{prefix}_address+4], %r229;",
        "}",
    ])
    return COS.subn(lambda match: match.group(0) + "\n" + block, text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    text, count = instrument(source.decode("utf-8"))
    output = text.encode("utf-8")
    passed = count == 1
    report = {
        "schema": 1,
        "experiment": "n0_cos_full_grid_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "N0_NVIDIA_COS_APPROXIMATION_CURVE_DATASET",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "grid": [GRID_X, GRID_Y, 1],
        "block": [32, 1, 1],
        "sample_count": SAMPLES,
        "samples_per_cta": SAMPLES_PER_CTA,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "scratch_extra_bytes": TRACE_BYTES,
        "insertion_count": count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
