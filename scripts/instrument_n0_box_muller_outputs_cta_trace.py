#!/usr/bin/env python3
"""Trace only the three FP16 Box-Muller values consumed by one selected N0 CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.instrument_n0_box_muller_cta_trace import SAMPLES, instrument_stages
except ModuleNotFoundError:
    from instrument_n0_box_muller_cta_trace import SAMPLES, instrument_stages


STAGES = [
    ("normal0_f16", "cvt.rn.f16.f32 %rs15, %r149;", "%rs15", "u16"),
    ("normal1_f16", "cvt.rn.f16.f32 %rs16, %r150;", "%rs16", "u16"),
    ("normal2_f16", "cvt.rn.f16.f32 %rs17, %r151;", "%rs17", "u16"),
]
BYTES_PER_SAMPLE = len(STAGES) * 4
TRACE_OFFSET = 384 * 640 * 32
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
        "experiment": "n0_box_muller_outputs_selected_cta_trace_instrumentation",
        "status": "PASS" if count == len(STAGES) else "FAIL",
        "classification": "N0_CONSUMED_BOX_MULLER_FP16_BOUNDARY_TRACE",
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
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
