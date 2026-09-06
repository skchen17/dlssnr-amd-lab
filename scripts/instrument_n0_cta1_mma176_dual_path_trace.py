#!/usr/bin/env python3
"""Trace both activation paths feeding the first divergent CTA(1,0) MMA 176."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
BYTES_PER_STAGE_LANE = 4
TRACE_OFFSET = 384 * 640 * 32
TARGET_CTA = (1, 0)

STAGES = (
    ("a_mma_d_r2150", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2148, %r2150\},.*?;", "%r2150"),
    ("a_reduction_r2321", r"\{add\.f16x2 %r2321,%r2266,%r2267;\s*\}", "%r2321"),
    ("a_max_floor_r2319", r"\{max\.f16x2 %r2361,%r2321,%r2319;\s*\}", "%r2319"),
    ("a_max_r2361", r"\{max\.f16x2 %r2361,%r2321,%r2319;\s*\}", "%r2361"),
    ("a_rsqrt_r2401", r"\{\.reg\.b16 hl, hu;\s*\.reg\.b32 fl, fu;\s*mov\.b32 \{hl, hu\}, %r2361;.*?mov\.b32 %r2401, \{hl, hu\};\s*\}", "%r2401"),
    ("a_scale1_r2455", r"\{mul\.f16x2 %r2455,%r2150,%r2401;\s*\}", "%r2455"),
    ("a_scale2_param_r2441", r"\{\.reg \.f16 low;\s*cvt\.rn\.f16\.f32 low, %r2438;\s*mov\.b32 %r2441, \{low,low\};\s*\}", "%r2441"),
    ("a_scale2_r2511", r"\{mul\.f16x2 %r2511,%r2455,%r2441;\s*\}", "%r2511"),
    ("a_e4m3_pack_r2941", r"mov\.b32 %r2941, \{%rs230, %rs231\};", "%r2941"),
    ("b_mma_d_r2537", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2537, %r2539\},.*?;", "%r2537"),
    ("b_reduction_r2713", r"\{add\.f16x2 %r2713,%r2656,%r2657;\s*\}", "%r2713"),
    ("b_max_floor_r2319", r"\{max\.f16x2 %r2753,%r2713,%r2319;\s*\}", "%r2319"),
    ("b_max_r2753", r"\{max\.f16x2 %r2753,%r2713,%r2319;\s*\}", "%r2753"),
    ("b_rsqrt_r2793", r"\{\.reg\.b16 hl, hu;\s*\.reg\.b32 fl, fu;\s*mov\.b32 \{hl, hu\}, %r2753;.*?mov\.b32 %r2793, \{hl, hu\};\s*\}", "%r2793"),
    ("b_scale_r2832", r"\{mul\.f16x2 %r2832,%r2537,%r2793;\s*\}", "%r2832"),
    ("b_e4m3_pack_r4337", r"mov\.b32 %r4337, \{%rs256, %rs257\};", "%r4337"),
)
TRACE_BYTES = len(STAGES) * LANES * BYTES_PER_STAGE_LANE


def store_block(stage: int, register: str) -> str:
    base = TRACE_OFFSET + stage * LANES * BYTES_PER_STAGE_LANE
    p = f"__n0_cta1_dual_{stage}"
    return "\n".join(("{", f".reg .pred %{p}_px, %{p}_py, %{p}_selected;",
        f".reg .b32 %{p}_lane, %{p}_offset, %{p}_ctax, %{p}_ctay;", f".reg .b64 %{p}_wide, %{p}_address;",
        f"mov.u32 %{p}_ctax, %ctaid.x;", f"mov.u32 %{p}_ctay, %ctaid.y;",
        f"setp.eq.u32 %{p}_px, %{p}_ctax, {TARGET_CTA[0]};", f"setp.eq.u32 %{p}_py, %{p}_ctay, {TARGET_CTA[1]};",
        f"and.pred %{p}_selected, %{p}_px, %{p}_py;", f"mov.u32 %{p}_lane, %laneid;",
        f"mad.lo.u32 %{p}_offset, %{p}_lane, {BYTES_PER_STAGE_LANE}, {base};", f"cvt.u64.u32 %{p}_wide, %{p}_offset;",
        f"add.s64 %{p}_address, %rd6, %{p}_wide;", f"@%{p}_selected st.global.b32 [%{p}_address], {register};", "}"))


def instrument(text: str) -> tuple[str, list[dict]]:
    output = text; grouped: dict[str, list[tuple[int, str, str]]] = {}
    for index, (name, pattern, register) in enumerate(STAGES): grouped.setdefault(pattern, []).append((index, name, register))
    report = []
    for pattern, stages in grouped.items():
        matches = list(re.finditer(pattern, output, flags=re.DOTALL))
        if len(matches) != 1: raise ValueError(f"expected one source match for {', '.join(x[1] for x in stages)}, found {len(matches)}")
        suffix = "\n" + "\n".join(store_block(i, reg) for i, _, reg in stages)
        output = re.sub(pattern, lambda m: m.group(0) + suffix, output, count=1, flags=re.DOTALL)
        report.extend({"index": i, "name": name, "register": reg} for i, name, reg in stages)
    report.sort(key=lambda x: x["index"]); return output, report


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("input",type=Path);parser.add_argument("output",type=Path);parser.add_argument("report",type=Path);args=parser.parse_args();source=args.input.read_bytes()
    try: output_text,stages=instrument(source.decode("utf-8"));status,error="PASS",None
    except ValueError as exc: output_text,stages=source.decode("utf-8"),[];status,error="FAIL",str(exc)
    output=output_text.encode("utf-8");result={"schema":1,"experiment":"n0_cta1_mma176_dual_path_trace_instrumentation","status":status,"classification":"N0_FIRST_OBSERVABLE_MISMATCH_DUAL_ACTIVATION_PATH_TRACE","source_sha256":hashlib.sha256(source).hexdigest().upper(),"output_sha256":hashlib.sha256(output).hexdigest().upper(),"lanes":LANES,"bytes_per_stage_lane":4,"stage_count":len(STAGES),"trace_offset":TRACE_OFFSET,"trace_bytes":TRACE_BYTES,"scratch_extra_bytes":TRACE_BYTES,"target_cta":[1,0,0],"required_grid":[80,48,1],"required_block":[32,1,1],"stages":stages,"error":error}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(output);args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,separators=(",",":")));return 0 if status=="PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
