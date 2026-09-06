#!/usr/bin/env python3
"""Build full-grid sine0 and normal1-boundary N0 trace variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GRID_X, GRID_Y = 80, 48
SAMPLES_PER_CTA = 64
SAMPLES = GRID_X * GRID_Y * SAMPLES_PER_CTA
TRACE_OFFSET = 384 * 640 * 32
BYTES_PER_SAMPLE = 8
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE
SIN0 = re.compile(r"sin\.approx\.ftz\.f32\s+%r228\s*,\s*%r226\s*;")
SQRT0 = re.compile(r"sqrt\.approx\.ftz\.f32\s+%r221\s*,\s*%r220\s*;")
NORMAL1 = re.compile(r"cvt\.rn\.f16\.f32\s+%rs16\s*,\s*%r150\s*;")


def address_lines(prefix: str, field_offset: int) -> list[str]:
    return [
        f".reg .b32 %{prefix}_ctax, %{prefix}_ctay, %{prefix}_linear, "
        f"%{prefix}_sample, %{prefix}_offset;",
        f".reg .b64 %{prefix}_scratch, %{prefix}_wide, %{prefix}_address;",
        f"mov.u32 %{prefix}_ctax, %ctaid.x;",
        f"mov.u32 %{prefix}_ctay, %ctaid.y;",
        f"mad.lo.u32 %{prefix}_linear, %{prefix}_ctay, {GRID_X}, %{prefix}_ctax;",
        f"mad.lo.u32 %{prefix}_sample, %{prefix}_linear, {SAMPLES_PER_CTA}, %r5556;",
        f"mad.lo.u32 %{prefix}_offset, %{prefix}_sample, {BYTES_PER_SAMPLE}, "
        f"{TRACE_OFFSET + field_offset};",
        f"ld.param.b64 %{prefix}_scratch, [%rd16+216];",
        f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
        f"add.s64 %{prefix}_address, %{prefix}_scratch, %{prefix}_wide;",
    ]


def instrument_sin(text: str) -> tuple[str, int]:
    prefix = "__n0_sin0_grid"
    block = "\n".join([
        "{", *address_lines(prefix, 0),
        f"st.global.b32 [%{prefix}_address], %r226;",
        f"st.global.b32 [%{prefix}_address+4], %r228;", "}",
    ])
    return SIN0.subn(lambda match: match.group(0) + "\n" + block, text)


def boundary_store(source: str, field_offset: int, half: bool, tag: str) -> str:
    prefix = f"__n0_normal1_grid_{tag}"
    lines = ["{", *address_lines(prefix, field_offset)]
    value = source
    if half:
        value = f"%{prefix}_half"
        lines.insert(1, f".reg .b32 {value};")
        lines.append(f"cvt.u32.u16 {value}, {source};")
    lines.extend((f"st.global.b32 [%{prefix}_address], {value};", "}"))
    return "\n".join(lines)


def instrument_boundary(text: str) -> tuple[str, dict[str, int]]:
    text, sqrt_count = SQRT0.subn(
        lambda match: match.group(0) + "\n" + boundary_store("%r221", 0, False, "sqrt"), text
    )
    text, normal_count = NORMAL1.subn(
        lambda match: match.group(0) + "\n" + boundary_store("%rs16", 4, True, "normal"), text
    )
    return text, {"sqrt0": sqrt_count, "normal1_f16": normal_count}


def generate(text: str, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sin_text, sin_count = instrument_sin(text)
    boundary_text, boundary_counts = instrument_boundary(text)
    if sin_count != 1 or boundary_counts != {"sqrt0": 1, "normal1_f16": 1}:
        raise ValueError(
            f"instrumentation counts invalid: sin={sin_count}, boundary={boundary_counts}"
        )
    variants = []
    for index, (name, fields, output_text) in enumerate((
        ("sin0", ["phase0_f32", "sin0_f32"], sin_text),
        ("normal1_boundary", ["sqrt0_f32", "normal1_f16_as_u32"], boundary_text),
    )):
        data = output_text.encode("utf-8")
        filename = f"variant_{index}_{name}.ptx"
        (output_dir / filename).write_bytes(data)
        variants.append({
            "index": index,
            "name": name,
            "filename": filename,
            "fields": fields,
            "sha256": hashlib.sha256(data).hexdigest().upper(),
        })
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    variants = generate(source.decode("utf-8"), args.output_dir)
    report = {
        "schema": 1,
        "experiment": "n0_normal1_full_grid_trace_variants",
        "status": "PASS",
        "classification": "N0_FIRST_BOX_MULLER_SINE_PATH_FULL_GRID_DATASET",
        "grid": [GRID_X, GRID_Y, 1],
        "block": [32, 1, 1],
        "sample_count": SAMPLES,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "trace_offset": TRACE_OFFSET,
        "trace_bytes_per_variant": TRACE_BYTES,
        "scratch_extra_bytes": TRACE_BYTES,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "variants": variants,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
