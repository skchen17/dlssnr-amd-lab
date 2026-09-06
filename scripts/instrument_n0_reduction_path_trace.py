#!/usr/bin/env python3
"""Trace every visible node in the N0 r2329 packed-FP16 square reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.instrument_n0_norm_path_trace import (
        BYTES_PER_STAGE_LANE,
        LANES,
        TRACE_OFFSET,
        store_block,
    )
except ModuleNotFoundError:
    from instrument_n0_norm_path_trace import BYTES_PER_STAGE_LANE, LANES, TRACE_OFFSET, store_block


STAGES = (
    ("mma_r2152", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2152, %r2154\},.*?;", "%r2152"),
    ("mma_r2156", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2156, %r2158\},.*?;", "%r2156"),
    ("mma_r2160", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2160, %r2162\},.*?;", "%r2160"),
    ("mma_r2164", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+\{%r2164, %r2166\},.*?;", "%r2164"),
    ("square_r2212", r"\{mul\.f16x2 %r2212,%r2152,%r2152;\s*\}", "%r2212"),
    ("square_r2213", r"\{mul\.f16x2 %r2213,%r2160,%r2160;\s*\}", "%r2213"),
    ("square_r2218", r"\{mul\.f16x2 %r2218,%r2156,%r2156;\s*\}", "%r2218"),
    ("square_r2219", r"\{mul\.f16x2 %r2219,%r2164,%r2164;\s*\}", "%r2219"),
    ("pair_r2268", r"\{add\.f16x2 %r2268,%r2218,%r2219;\s*\}", "%r2268"),
    ("pair_r2269", r"\{add\.f16x2 %r2269,%r2212,%r2213;\s*\}", "%r2269"),
    ("lane_r2272", r"\{add\.f16x2 %r2272,%r2268,%r2269;\s*\}", "%r2272"),
    ("shuffle_xor2_r2273", r"\{shfl\.sync\.bfly\.b32 %r2273,%r2272,%r2253,%r2254,%r2255;\s*\}", "%r2273"),
    ("xor2_sum_r2274", r"\{add\.f16x2 %r2274,%r2272,%r2273;\s*\}", "%r2274"),
    ("shuffle_xor1_r2275", r"\{shfl\.sync\.bfly\.b32 %r2275,%r2274,%r2258,%r2254,%r2255;\s*\}", "%r2275"),
    ("xor1_sum_r2276", r"\{add\.f16x2 %r2276,%r2274,%r2275;\s*\}", "%r2276"),
    ("swap_r2277", r"mov\.b32 %r2277, \{%rs390, %rs389\};", "%r2277"),
    ("final_r2329", r"\{add\.f16x2 %r2329,%r2276,%r2277;\s*\}", "%r2329"),
)
TRACE_BYTES = len(STAGES) * LANES * BYTES_PER_STAGE_LANE


def instrument(text: str) -> tuple[str, list[dict]]:
    output = text
    report = []
    for index, (name, pattern, register) in enumerate(STAGES):
        matches = list(re.finditer(pattern, output, flags=re.DOTALL))
        if len(matches) != 1:
            raise ValueError(f"expected one source match for {name}, found {len(matches)}")
        output = re.sub(pattern, lambda match: match.group(0) + "\n" + store_block(index, register),
                        output, count=1, flags=re.DOTALL)
        report.append({"index": index, "name": name, "register": register})
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path); parser.add_argument("report", type=Path)
    args = parser.parse_args(); source = args.input.read_bytes()
    try:
        output_text, stages = instrument(source.decode("utf-8")); status = "PASS"; error = None
    except ValueError as exc:
        output_text = source.decode("utf-8"); stages = []; status = "FAIL"; error = str(exc)
    output = output_text.encode("utf-8")
    result = {
        "schema": 1, "experiment": "n0_r2329_reduction_path_trace_instrumentation",
        "status": status, "classification": "N0_HIDDEN_PRECISION_FUSION_BOUNDARY_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "lanes": LANES, "bytes_per_stage_lane": BYTES_PER_STAGE_LANE,
        "stage_count": len(STAGES), "trace_offset": TRACE_OFFSET, "trace_bytes": TRACE_BYTES,
        "scratch_register": "%rd6", "required_grid": [1, 1, 1], "required_block": [32, 1, 1],
        "stages": stages, "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(output)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
