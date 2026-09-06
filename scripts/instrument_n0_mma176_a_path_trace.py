#!/usr/bin/env python3
"""Trace both normalization reductions feeding N0 selected-CTA MMA 176 A."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
BYTES_PER_STAGE_LANE = 4
TRACE_OFFSET = 384 * 640 * 32

STAGES = (
    ("a0_mma_r2136", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2136, %r2138\},.*?;", "%r2136"),
    ("a0_mma_r2140", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2140, %r2142\},.*?;", "%r2140"),
    ("a0_mma_r2144", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2144, %r2146\},.*?;", "%r2144"),
    ("a0_mma_r2148", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2148, %r2150\},.*?;", "%r2148"),
    ("a0_square_r2200", r"\{mul\.f16x2 %r2200,%r2136,%r2136;\s*\}", "%r2200"),
    ("a0_square_r2201", r"\{mul\.f16x2 %r2201,%r2144,%r2144;\s*\}", "%r2201"),
    ("a0_square_r2206", r"\{mul\.f16x2 %r2206,%r2140,%r2140;\s*\}", "%r2206"),
    ("a0_square_r2207", r"\{mul\.f16x2 %r2207,%r2148,%r2148;\s*\}", "%r2207"),
    ("a0_pair_r2248", r"\{add\.f16x2 %r2248,%r2200,%r2201;\s*\}", "%r2248"),
    ("a0_pair_r2247", r"\{add\.f16x2 %r2247,%r2206,%r2207;\s*\}", "%r2247"),
    ("a0_lane_r2252", r"\{add\.f16x2 %r2252,%r2247,%r2248;\s*\}", "%r2252"),
    ("a0_shuffle2_r2256", r"\{shfl\.sync\.bfly\.b32 %r2256,%r2252,%r2253,%r2254,%r2255;\s*\}", "%r2256"),
    ("a0_sum2_r2257", r"\{add\.f16x2 %r2257,%r2252,%r2256;\s*\}", "%r2257"),
    ("a0_shuffle1_r2259", r"\{shfl\.sync\.bfly\.b32 %r2259,%r2257,%r2258,%r2254,%r2255;\s*\}", "%r2259"),
    ("a0_sum1_r2260", r"\{add\.f16x2 %r2260,%r2257,%r2259;\s*\}", "%r2260"),
    ("a0_swap_r2261", r"mov\.b32 %r2261, \{%rs386, %rs385\};", "%r2261"),
    ("a0_reduction_r2318", r"\{add\.f16x2 %r2318,%r2260,%r2261;\s*\}", "%r2318"),
    ("a0_max_r2359", r"\{max\.f16x2 %r2359,%r2318,%r2319;\s*\}", "%r2359"),
    ("a0_rsqrt_r2399", r"\{\.reg\.b16 hl, hu;\s*\.reg\.b32 fl, fu;\s*mov\.b32 \{hl, hu\}, %r2359;.*?mov\.b32 %r2399, \{hl, hu\};\s*\}", "%r2399"),
    ("a0_scale1_r2453", r"\{mul\.f16x2 %r2453,%r2148,%r2399;\s*\}", "%r2453"),
    ("a0_scale2_r2509", r"\{mul\.f16x2 %r2509,%r2453,%r2441;\s*\}", "%r2509"),
    ("a0_pack_r2940", r"mov\.b32 %r2940, \{%rs228, %rs229\};", "%r2940"),
    ("a1_mma_r2138", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2136, %r2138\},.*?;", "%r2138"),
    ("a1_mma_r2142", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2140, %r2142\},.*?;", "%r2142"),
    ("a1_mma_r2146", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2144, %r2146\},.*?;", "%r2146"),
    ("a1_mma_r2150", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2148, %r2150\},.*?;", "%r2150"),
    ("a1_square_r2203", r"\{mul\.f16x2 %r2203,%r2138,%r2138;\s*\}", "%r2203"),
    ("a1_square_r2204", r"\{mul\.f16x2 %r2204,%r2146,%r2146;\s*\}", "%r2204"),
    ("a1_square_r2209", r"\{mul\.f16x2 %r2209,%r2142,%r2142;\s*\}", "%r2209"),
    ("a1_square_r2210", r"\{mul\.f16x2 %r2210,%r2150,%r2150;\s*\}", "%r2210"),
    ("a1_pair_r2250", r"\{add\.f16x2 %r2250,%r2203,%r2204;\s*\}", "%r2250"),
    ("a1_pair_r2249", r"\{add\.f16x2 %r2249,%r2209,%r2210;\s*\}", "%r2249"),
    ("a1_lane_r2262", r"\{add\.f16x2 %r2262,%r2249,%r2250;\s*\}", "%r2262"),
    ("a1_shuffle2_r2263", r"\{shfl\.sync\.bfly\.b32 %r2263,%r2262,%r2253,%r2254,%r2255;\s*\}", "%r2263"),
    ("a1_sum2_r2264", r"\{add\.f16x2 %r2264,%r2262,%r2263;\s*\}", "%r2264"),
    ("a1_shuffle1_r2265", r"\{shfl\.sync\.bfly\.b32 %r2265,%r2264,%r2258,%r2254,%r2255;\s*\}", "%r2265"),
    ("a1_sum1_r2266", r"\{add\.f16x2 %r2266,%r2264,%r2265;\s*\}", "%r2266"),
    ("a1_swap_r2267", r"mov\.b32 %r2267, \{%rs388, %rs387\};", "%r2267"),
    ("a1_reduction_r2321", r"\{add\.f16x2 %r2321,%r2266,%r2267;\s*\}", "%r2321"),
    ("a1_max_r2361", r"\{max\.f16x2 %r2361,%r2321,%r2319;\s*\}", "%r2361"),
    ("a1_rsqrt_r2401", r"\{\.reg\.b16 hl, hu;\s*\.reg\.b32 fl, fu;\s*mov\.b32 \{hl, hu\}, %r2361;.*?mov\.b32 %r2401, \{hl, hu\};\s*\}", "%r2401"),
    ("a1_scale1_r2443", r"\{mul\.f16x2 %r2443,%r2138,%r2401;\s*\}", "%r2443"),
    ("scale2_param_r2441", r"\{\.reg \.f16 low;\s*cvt\.rn\.f16\.f32 low, %r2438;\s*mov\.b32 %r2441, \{low,low\};\s*\}", "%r2441"),
    ("a1_scale2_r2506", r"\{mul\.f16x2 %r2506,%r2443,%r2441;\s*\}", "%r2506"),
    ("a1_pack_r2939", r"mov\.b32 %r2939, \{%rs226, %rs227\};", "%r2939"),
)
TRACE_BYTES = len(STAGES) * LANES * BYTES_PER_STAGE_LANE


def store_block(stage: int, register: str, target_x: int, target_y: int) -> str:
    base = TRACE_OFFSET + stage * LANES * BYTES_PER_STAGE_LANE
    p = f"__n0_a176_{stage}"
    return "\n".join(("{", f".reg .pred %{p}_px, %{p}_py, %{p}_selected;",
        f".reg .b32 %{p}_lane, %{p}_offset, %{p}_ctax, %{p}_ctay;", f".reg .b64 %{p}_wide, %{p}_address;",
        f"mov.u32 %{p}_ctax, %ctaid.x;", f"mov.u32 %{p}_ctay, %ctaid.y;",
        f"setp.eq.u32 %{p}_px, %{p}_ctax, {target_x};", f"setp.eq.u32 %{p}_py, %{p}_ctay, {target_y};",
        f"and.pred %{p}_selected, %{p}_px, %{p}_py;", f"mov.u32 %{p}_lane, %laneid;",
        f"mad.lo.u32 %{p}_offset, %{p}_lane, 4, {base};", f"cvt.u64.u32 %{p}_wide, %{p}_offset;",
        f"add.s64 %{p}_address, %rd6, %{p}_wide;", f"@%{p}_selected st.global.b32 [%{p}_address], {register};", "}"))


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, list[dict]]:
    output = text
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for index, (name, pattern, register) in enumerate(STAGES):
        grouped.setdefault(pattern, []).append((index, name, register))
    report = []
    for pattern, stages in grouped.items():
        matches = list(re.finditer(pattern, output, flags=re.DOTALL))
        if len(matches) != 1:
            raise ValueError(f"expected one source match for {', '.join(x[1] for x in stages)}, found {len(matches)}")
        suffix = "\n" + "\n".join(store_block(i, register, target_x, target_y) for i, _, register in stages)
        output = re.sub(pattern, lambda match: match.group(0) + suffix, output, count=1, flags=re.DOTALL)
        report.extend({"index": i, "name": name, "register": register} for i, name, register in stages)
    report.sort(key=lambda item: item["index"])
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path); parser.add_argument("report", type=Path)
    parser.add_argument("--target-x", type=int, required=True); parser.add_argument("--target-y", type=int, required=True)
    args = parser.parse_args()
    if not (0 <= args.target_x < 80 and 0 <= args.target_y < 48): parser.error("target CTA must be inside the 80x48 N0 grid")
    source = args.input.read_bytes()
    try: output_text, stages = instrument(source.decode("utf-8"), args.target_x, args.target_y); status, error = "PASS", None
    except ValueError as exc: output_text, stages = source.decode("utf-8"), []; status, error = "FAIL", str(exc)
    output = output_text.encode("utf-8")
    result = {"schema":1,"experiment":"n0_selected_cta_mma176_a_path_trace_instrumentation","status":status,
        "classification":"N0_MMA176_A_NORMALIZATION_REDUCTION_TRACE","source_sha256":hashlib.sha256(source).hexdigest().upper(),
        "output_sha256":hashlib.sha256(output).hexdigest().upper(),"lanes":LANES,"bytes_per_stage_lane":4,
        "stage_count":len(STAGES),"trace_offset":TRACE_OFFSET,"trace_bytes":TRACE_BYTES,"scratch_extra_bytes":TRACE_BYTES,
        "target_cta":[args.target_x,args.target_y,0],"required_grid":[80,48,1],"required_block":[32,1,1],"stages":stages,"error":error}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes(output)
    args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,separators=(",",":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
