#!/usr/bin/env python3
"""Build one selected-CTA trace-store PTX variant per MMA-176 A-path stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.instrument_n0_mma176_a_path_trace import STAGES, store_block
except ModuleNotFoundError:
    from instrument_n0_mma176_a_path_trace import STAGES, store_block


TRACE_BYTES_PER_VARIANT = 32 * 4


def generate(text: str, output_dir: Path, target_x: int, target_y: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for index, (name, pattern, register) in enumerate(STAGES):
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))
        if len(matches) != 1: raise ValueError(f"expected one source match for {name}, found {len(matches)}")
        output = re.sub(pattern, lambda match: match.group(0) + "\n" + store_block(0, register, target_x, target_y),
                        text, count=1, flags=re.DOTALL)
        filename = f"stage_{index:02d}.ptx"; data = output.encode("utf-8"); (output_dir / filename).write_bytes(data)
        variants.append({"index":index,"name":name,"register":register,"filename":filename,
                         "sha256":hashlib.sha256(data).hexdigest().upper()})
    return variants


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("input",type=Path);parser.add_argument("output_dir",type=Path);parser.add_argument("report",type=Path)
    parser.add_argument("--target-x",type=int,required=True);parser.add_argument("--target-y",type=int,required=True);args=parser.parse_args()
    if not(0<=args.target_x<80 and 0<=args.target_y<48):parser.error("target CTA must be inside the 80x48 N0 grid")
    source=args.input.read_bytes();variants=generate(source.decode("utf-8"),args.output_dir,args.target_x,args.target_y)
    report={"schema":1,"experiment":"n0_mma176_a_single_stage_variant_generation","status":"PASS",
        "classification":"RTX_HIDDEN_FUSION_SENSITIVITY_SWEEP","source_sha256":hashlib.sha256(source).hexdigest().upper(),
        "target_cta":[args.target_x,args.target_y,0],"variant_count":len(variants),"trace_offset":384*640*32,
        "trace_bytes_per_variant":TRACE_BYTES_PER_VARIANT,"scratch_extra_bytes":TRACE_BYTES_PER_VARIANT,"variants":variants}
    args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,separators=(",",":")));return 0


if __name__=="__main__":raise SystemExit(main())
