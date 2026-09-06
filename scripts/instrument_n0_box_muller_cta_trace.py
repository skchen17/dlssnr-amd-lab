#!/usr/bin/env python3
"""Trace the N0 Box-Muller positional-feature path for all 64 samples of one CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


BASE_SCRATCH_BYTES = 384 * 640 * 32
SAMPLES = 64
STAGES = [
    ("hash", "xor.b32 %r169, %r168, %r167;", "%r169", "b32"),
    ("uniform0", "mul.ftz.f32 %r181, %r180, 0f33800000;", "%r181", "f32"),
    ("lg2_u0", "lg2.approx.ftz.f32 %r218, %r181;", "%r218", "f32"),
    ("sqrt0", "sqrt.approx.ftz.f32 %r221, %r220;", "%r221", "f32"),
    ("phase1", "mul.ftz.f32 %r226, %r193, 0f40C90FDB;", "%r226", "f32"),
    ("cos1", "cos.approx.ftz.f32 %r229, %r226;", "%r229", "f32"),
    ("normal0", "mul.ftz.f32 %r149, %r221, %r229;", "%r149", "f32"),
    ("normal0_f16", "cvt.rn.f16.f32 %rs15, %r149;", "%rs15", "u16"),
]
BYTES_PER_SAMPLE = len(STAGES) * 4
TRACE_OFFSET = BASE_SCRATCH_BYTES
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE
def instrument_stages(text: str, target_x: int, target_y: int,
                      stages: list[tuple[str, str, str, str]]) -> tuple[str, int]:
    count = 0
    for index, (_, statement, register, kind) in enumerate(stages):
        prefix = f"__n0_box_trace_{index}"
        base = TRACE_OFFSET + index * SAMPLES * 4
        source = register
        half_decl = half_convert = ""
        if kind == "u16":
            source = f"%{prefix}_half"
            half_decl = f".reg .b32 {source};"
            half_convert = f"cvt.u32.u16 {source}, {register};"
        block = "\n".join([
            "{", f".reg .pred %{prefix}_px, %{prefix}_py, %{prefix}_selected;",
            f".reg .b32 %{prefix}_ctax, %{prefix}_ctay, %{prefix}_offset;", half_decl,
            f".reg .b64 %{prefix}_scratch, %{prefix}_wide, %{prefix}_address;",
            f"mov.u32 %{prefix}_ctax, %ctaid.x;", f"mov.u32 %{prefix}_ctay, %ctaid.y;",
            f"setp.eq.u32 %{prefix}_px, %{prefix}_ctax, {target_x};",
            f"setp.eq.u32 %{prefix}_py, %{prefix}_ctay, {target_y};",
            f"and.pred %{prefix}_selected, %{prefix}_px, %{prefix}_py;",
            f"ld.param.b64 %{prefix}_scratch, [%rd16+216];",
            f"mad.lo.u32 %{prefix}_offset, %r5556, 4, {base};",
            f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
            f"add.s64 %{prefix}_address, %{prefix}_scratch, %{prefix}_wide;", half_convert,
            f"@%{prefix}_selected st.global.b32 [%{prefix}_address], {source};", "}",
        ])
        pattern = re.compile(re.escape(statement))
        text, changed = pattern.subn(lambda match: match.group(0) + "\n" + block, text)
        if changed != 1:
            raise ValueError(f"expected one {stages[index][0]} statement, found {changed}")
        count += changed
    return text, count


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, int]:
    return instrument_stages(text, target_x, target_y, STAGES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path); parser.add_argument("--target-x", type=int, required=True)
    parser.add_argument("--target-y", type=int, required=True)
    args = parser.parse_args()
    if not (0 <= args.target_x < 80 and 0 <= args.target_y < 48):
        parser.error("target CTA must be inside the 80x48 N0 grid")
    source = args.input.read_bytes(); output_text, count = instrument(source.decode("utf-8"), args.target_x, args.target_y)
    output = output_text.encode("utf-8")
    report = {
        "schema": 1, "experiment": "n0_box_muller_selected_cta_trace_instrumentation", "status": "PASS",
        "classification": "N0_FIRST_OBSERVABLE_INPUT_APPROX_MATH_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(), "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0], "sample_count": SAMPLES,
        "stage_count": len(STAGES), "stages": [name for name, _, _, _ in STAGES],
        "bytes_per_sample": BYTES_PER_SAMPLE, "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES, "scratch_extra_bytes": TRACE_BYTES,
        "required_grid": [80, 48, 1], "required_block": [32, 1, 1], "insertion_count": count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":"))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
