#!/usr/bin/env python3
"""Trace the N0 normalization path feeding the first divergent FP8 MMA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
BYTES_PER_STAGE_LANE = 4
TRACE_OFFSET = 4 * 1024 * 1024

# The RTX/RX all-MMA trace first differs at MMA 184, lane 7, A2.  That byte is
# produced from r2517 -> rs237 -> the upper half of r2960.  These checkpoints
# walk backwards through the exact normalization path that produces r2517.
STAGES = (
    ("mma_d_r2164", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2164, %r2166\},.*?;", "%r2164"),
    ("max_input_r2329", r"\{max\.f16x2 %r2369,%r2329,%r2319;\s*\}", "%r2329"),
    ("max_input_r2319", r"\{max\.f16x2 %r2369,%r2329,%r2319;\s*\}", "%r2319"),
    ("max_result_r2369", r"\{max\.f16x2 %r2369,%r2329,%r2319;\s*\}", "%r2369"),
    ("rsqrt_result_r2409", r"\{\.reg\.b16 hl, hu;\s*\.reg\.b32 fl, fu;\s*mov\.b32 \{hl, hu\}, %r2369;.*?mov\.b32 %r2409, \{hl, hu\};\s*\}", "%r2409"),
    ("scale1_result_r2469", r"\{mul\.f16x2 %r2469,%r2164,%r2409;\s*\}", "%r2469"),
    ("scale2_f32_r2438", r"ld\.global\.b32 %r2438, \[%rd55\+20576\];", "%r2438"),
    ("scale2_f16x2_r2441", r"\{\.reg \.f16 low;\s*cvt\.rn\.f16\.f32 low, %r2438;\s*mov\.b32 %r2441, \{low,low\};\s*\}", "%r2441"),
    ("scale2_result_r2517", r"\{mul\.f16x2 %r2517,%r2469,%r2441;\s*\}", "%r2517"),
    ("e4m3_pack_r2960", r"mov\.b32 %r2960, \{%rs236, %rs237\};", "%r2960"),
)

TRACE_BYTES = len(STAGES) * LANES * BYTES_PER_STAGE_LANE


def store_block(stage: int, register: str) -> str:
    base = TRACE_OFFSET + stage * LANES * BYTES_PER_STAGE_LANE
    prefix = f"__n0_norm_{stage}"
    return "\n".join((
        "{",
        f".reg .b32 %{prefix}_lane, %{prefix}_offset;",
        f".reg .b64 %{prefix}_wide, %{prefix}_address;",
        f"mov.u32 %{prefix}_lane, %laneid;",
        f"mad.lo.u32 %{prefix}_offset, %{prefix}_lane, {BYTES_PER_STAGE_LANE}, {base};",
        f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
        f"add.s64 %{prefix}_address, %rd6, %{prefix}_wide;",
        f"st.global.b32 [%{prefix}_address], {register};",
        "}",
    ))


def instrument(text: str) -> tuple[str, list[dict]]:
    report = []
    output = text
    # Three stages share the same max instruction. Insert their stores together
    # so each source pattern is still required to occur exactly once.
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for index, (name, pattern, register) in enumerate(STAGES):
        grouped.setdefault(pattern, []).append((index, name, register))
    for pattern, stages in grouped.items():
        matches = list(re.finditer(pattern, output, flags=re.DOTALL))
        if len(matches) != 1:
            names = ", ".join(name for _, name, _ in stages)
            raise ValueError(f"expected one source match for {names}, found {len(matches)}")
        suffix = "\n" + "\n".join(store_block(index, register) for index, _, register in stages)
        output = re.sub(pattern, lambda match: match.group(0) + suffix, output, count=1, flags=re.DOTALL)
        report.extend({"index": index, "name": name, "register": register} for index, name, register in stages)
    report.sort(key=lambda item: item["index"])
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    try:
        output_text, stages = instrument(source.decode("utf-8"))
        status = "PASS"
        error = None
    except ValueError as exc:
        output_text = source.decode("utf-8")
        stages = []
        status = "FAIL"
        error = str(exc)
    output = output_text.encode("utf-8")
    result = {
        "schema": 1,
        "experiment": "n0_normalization_path_trace_instrumentation",
        "status": status,
        "classification": "N0_FIRST_DIVERGENT_ACTIVATION_PATH_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "lanes": LANES,
        "bytes_per_stage_lane": BYTES_PER_STAGE_LANE,
        "stage_count": len(STAGES),
        "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "scratch_register": "%rd6",
        "required_grid": [1, 1, 1],
        "required_block": [32, 1, 1],
        "stages": stages,
        "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
