#!/usr/bin/env python3
"""Trace the short activation path feeding B1 of zero-input N0 MMA 178."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
BYTES_PER_STAGE_LANE = 4
TRACE_OFFSET = 4 * 1024 * 1024

# MMA 178 consumes {%r4349,%r4350}.  Its only first-divergent byte is the
# upper E4M3 byte of %rs266 inside %r4350.  %rs266 is converted from %r2842,
# which is %r2561 * %r2803.  The stages below bracket that exact path while
# avoiding materialization of every internal reduction node.
STAGES = (
    ("mma146_d_r2561", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2561, %r2563\},.*?;", "%r2561"),
    ("mma147_d_r2565", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2565, %r2567\},.*?;", "%r2565"),
    ("reduction_r2723", r"\{add\.f16x2 %r2723,%r2672,%r2673;\s*\}", "%r2723"),
    ("max_other_r2319", r"\{max\.f16x2 %r2763,%r2723,%r2319;\s*\}", "%r2319"),
    ("max_result_r2763", r"\{max\.f16x2 %r2763,%r2723,%r2319;\s*\}", "%r2763"),
    ("rsqrt_result_r2803", r"\{\.reg\.b16 hl, hu;\s*\.reg\.b32 fl, fu;\s*mov\.b32 \{hl, hu\}, %r2763;.*?mov\.b32 %r2803, \{hl, hu\};\s*\}", "%r2803"),
    ("scale_mma146_r2842", r"\{mul\.f16x2 %r2842,%r2561,%r2803;\s*\}", "%r2842"),
    ("scale_mma147_r2843", r"\{mul\.f16x2 %r2843,%r2565,%r2803;\s*\}", "%r2843"),
    ("e4m3_b1_pack_r4350", r"mov\.b32 %r4350, \{%rs266, %rs267\};", "%r4350"),
)

TRACE_BYTES = len(STAGES) * LANES * BYTES_PER_STAGE_LANE


def store_block(stage: int, register: str) -> str:
    base = TRACE_OFFSET + stage * LANES * BYTES_PER_STAGE_LANE
    prefix = f"__n0_mma178_b_{stage}"
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
    output = text
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for index, (name, pattern, register) in enumerate(STAGES):
        grouped.setdefault(pattern, []).append((index, name, register))
    report = []
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
        status, error = "PASS", None
    except ValueError as exc:
        output_text, stages = source.decode("utf-8"), []
        status, error = "FAIL", str(exc)
    output = output_text.encode("utf-8")
    result = {
        "schema": 1,
        "experiment": "n0_mma178_b_path_trace_instrumentation",
        "status": status,
        "classification": "N0_ZERO_INPUT_MMA178_B1_PATH_TRACE",
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
