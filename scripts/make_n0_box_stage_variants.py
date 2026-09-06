#!/usr/bin/env python3
"""Build independent selected-CTA Box-Muller stage trace variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.instrument_n0_box_muller_cta_trace import SAMPLES, instrument_stages
except ModuleNotFoundError:
    from instrument_n0_box_muller_cta_trace import SAMPLES, instrument_stages


STAGES = (
    ("sqrt0", "sqrt.approx.ftz.f32 %r221, %r220;", "%r221", "f32"),
    ("cos0", "cos.approx.ftz.f32 %r229, %r226;", "%r229", "f32"),
    ("normal0", "mul.ftz.f32 %r149, %r221, %r229;", "%r149", "f32"),
)
TRACE_BYTES = SAMPLES * 4


def generate(text: str, output_dir: Path, target_x: int, target_y: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for index, stage in enumerate(STAGES):
        output, count = instrument_stages(text, target_x, target_y, [stage])
        if count != 1:
            raise ValueError(f"failed to instrument {stage[0]}")
        data = output.encode("utf-8")
        filename = f"stage_{index}_{stage[0]}.ptx"
        (output_dir / filename).write_bytes(data)
        variants.append({"index": index, "name": stage[0], "kind": stage[3],
                         "filename": filename,
                         "sha256": hashlib.sha256(data).hexdigest().upper()})
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--target-x", type=int, required=True)
    parser.add_argument("--target-y", type=int, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    variants = generate(source.decode("utf-8"), args.output_dir, args.target_x, args.target_y)
    report = {"schema": 1, "experiment": "n0_box_stage_single_observation_variants",
              "status": "PASS", "target_cta": [args.target_x, args.target_y, 0],
              "trace_offset": 384 * 640 * 32, "trace_bytes_per_variant": TRACE_BYTES,
              "scratch_extra_bytes": TRACE_BYTES, "variant_count": len(variants),
              "source_sha256": hashlib.sha256(source).hexdigest().upper(), "variants": variants}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
