#!/usr/bin/env python3
"""Trace all three N0 Box-Muller outputs for 64 samples of one CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.instrument_n0_box_muller_cta_trace import (
        BASE_SCRATCH_BYTES, SAMPLES, instrument_stages,
    )
except ModuleNotFoundError:
    from instrument_n0_box_muller_cta_trace import (
        BASE_SCRATCH_BYTES, SAMPLES, instrument_stages,
    )


STAGES = [
    ("hash", "xor.b32 %r169, %r168, %r167;", "%r169", "b32"),
    ("uniform0", "mul.ftz.f32 %r181, %r180, 0f33800000;", "%r181", "f32"),
    ("lg2_u0", "lg2.approx.ftz.f32 %r218, %r181;", "%r218", "f32"),
    ("sqrt0_input", "mul.ftz.f32 %r220, %r219, 0fC0000000;", "%r220", "f32"),
    ("sqrt0", "sqrt.approx.ftz.f32 %r221, %r220;", "%r221", "f32"),
    ("phase0_input", "mul.ftz.f32 %r193, %r192, 0f33800000;", "%r193", "f32"),
    ("phase0", "mul.ftz.f32 %r226, %r193, 0f40C90FDB;", "%r226", "f32"),
    ("sin0", "sin.approx.ftz.f32 %r228, %r226;", "%r228", "f32"),
    ("cos0", "cos.approx.ftz.f32 %r229, %r226;", "%r229", "f32"),
    ("normal0", "mul.ftz.f32 %r149, %r221, %r229;", "%r149", "f32"),
    ("normal1", "mul.ftz.f32 %r150, %r221, %r228;", "%r150", "f32"),
    ("normal0_f16", "cvt.rn.f16.f32 %rs15, %r149;", "%rs15", "u16"),
    ("normal1_f16", "cvt.rn.f16.f32 %rs16, %r150;", "%rs16", "u16"),
    ("uniform1", "mul.ftz.f32 %r205, %r204, 0f33800000;", "%r205", "f32"),
    ("lg2_u1", "lg2.approx.ftz.f32 %r222, %r205;", "%r222", "f32"),
    ("sqrt1_input", "mul.ftz.f32 %r224, %r223, 0fC0000000;", "%r224", "f32"),
    ("sqrt1", "sqrt.approx.ftz.f32 %r225, %r224;", "%r225", "f32"),
    ("phase1_input", "mul.ftz.f32 %r217, %r216, 0f33800000;", "%r217", "f32"),
    ("phase1", "mul.ftz.f32 %r227, %r217, 0f40C90FDB;", "%r227", "f32"),
    ("cos1", "cos.approx.ftz.f32 %r230, %r227;", "%r230", "f32"),
    ("normal2", "mul.ftz.f32 %r151, %r225, %r230;", "%r151", "f32"),
    ("normal2_f16", "cvt.rn.f16.f32 %rs17, %r151;", "%rs17", "u16"),
]
BYTES_PER_SAMPLE = len(STAGES) * 4
TRACE_OFFSET = BASE_SCRATCH_BYTES
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, int]:
    return instrument_stages(text, target_x, target_y, STAGES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--target-x", type=int, required=True)
    parser.add_argument("--target-y", type=int, required=True)
    args = parser.parse_args()
    if not (0 <= args.target_x < 80 and 0 <= args.target_y < 48):
        parser.error("target CTA must be inside the 80x48 N0 grid")
    source = args.input.read_bytes()
    output_text, count = instrument(source.decode("utf-8"), args.target_x, args.target_y)
    output = output_text.encode("utf-8")
    report = {
        "schema": 1,
        "experiment": "n0_box_muller_full_selected_cta_trace_instrumentation",
        "status": "PASS",
        "classification": "N0_ALL_BOX_MULLER_OUTPUTS_APPROX_MATH_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0],
        "sample_count": SAMPLES,
        "stage_count": len(STAGES),
        "stages": [name for name, _, _, _ in STAGES],
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "scratch_extra_bytes": TRACE_BYTES,
        "required_grid": [80, 48, 1],
        "required_block": [32, 1, 1],
        "insertion_count": count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
