#!/usr/bin/env python3
"""Trace all FP8 MMA fragments for one CTA and warp in the Swin 4h entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.lower_ptx_fp8_mma import FP8_MMA
except ModuleNotFoundError:
    from lower_ptx_fp8_mma import FP8_MMA


LANES = 32
BYTES_PER_LANE_MMA = 40
TRACE_PARAM_OFFSET = 72


def instrument(text: str, function: str, target_x: int, target_y: int,
               target_warp_y: int) -> tuple[str, int]:
    if min(target_x, target_y, target_warp_y) < 0:
        raise ValueError("CTA and warp coordinates must be non-negative")
    entry = re.compile(
        rf"(?s)(\.visible\s+\.entry\s+{re.escape(function)}\s*\(.*?\)\s*"
        rf"(?:\.[^\n]*\n\s*)*\{{)"
    )
    setup = r"""
.reg .b32 %__s4mma_lane, %__s4mma_ctax, %__s4mma_ctay, %__s4mma_warpy, %__s4mma_offset;
.reg .b64 %__s4mma_base, %__s4mma_address, %__s4mma_wide;
.reg .pred %__s4mma_px, %__s4mma_py, %__s4mma_pw, %__s4mma_pb, %__s4mma_guard;
ld.param.u64 %__s4mma_base, [FUNCTION_param_0+72];
mov.u32 %__s4mma_lane, %laneid;
mov.u32 %__s4mma_ctax, %ctaid.x;
mov.u32 %__s4mma_ctay, %ctaid.y;
mov.u32 %__s4mma_warpy, %tid.y;
setp.eq.u32 %__s4mma_px, %__s4mma_ctax, CTA_X;
setp.eq.u32 %__s4mma_py, %__s4mma_ctay, CTA_Y;
setp.eq.u32 %__s4mma_pw, %__s4mma_warpy, WARP_Y;
setp.ne.u64 %__s4mma_pb, %__s4mma_base, 0;
and.pred %__s4mma_guard, %__s4mma_px, %__s4mma_py;
and.pred %__s4mma_guard, %__s4mma_guard, %__s4mma_pw;
and.pred %__s4mma_guard, %__s4mma_guard, %__s4mma_pb;
""".replace("FUNCTION", function).replace("CTA_X", str(target_x)).replace(
        "CTA_Y", str(target_y)
    ).replace("WARP_Y", str(target_warp_y))
    output, entry_count = entry.subn(
        lambda match: match.group(1) + setup, text, count=1
    )
    if entry_count != 1:
        raise ValueError(f"entry {function} not found exactly once")

    mma_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal mma_index
        inputs = [match.group(f"a{i}") for i in range(4)]
        inputs += [match.group(f"b{i}") for i in range(2)]
        inputs += [match.group(f"c{i}") for i in range(2)]
        outputs = [match.group(f"d{i}") for i in range(2)]
        base = mma_index * LANES * BYTES_PER_LANE_MMA
        lines = [
            f"mad.lo.u32 %__s4mma_offset, %__s4mma_lane, {BYTES_PER_LANE_MMA}, {base};",
            "cvt.u64.u32 %__s4mma_wide, %__s4mma_offset;",
            "add.u64 %__s4mma_address, %__s4mma_base, %__s4mma_wide;",
        ]
        lines.extend(
            f"@%__s4mma_guard st.global.b32 [%__s4mma_address+{offset * 4}], {register};"
            for offset, register in enumerate(inputs)
        )
        lines.append(match.group(0))
        lines.extend(
            f"@%__s4mma_guard st.global.b32 [%__s4mma_address+{32 + offset * 4}], {register};"
            for offset, register in enumerate(outputs)
        )
        mma_index += 1
        return "\n".join(lines)

    return FP8_MMA.sub(replace, output), mma_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--target-x", required=True, type=int)
    parser.add_argument("--target-y", required=True, type=int)
    parser.add_argument("--target-warp-y", required=True, type=int)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, count = instrument(
        source.decode("utf-8"), args.function, args.target_x, args.target_y,
        args.target_warp_y,
    )
    output = output_text.encode("utf-8")
    trace_bytes = count * LANES * BYTES_PER_LANE_MMA
    passed = count == args.expected_count and len(FP8_MMA.findall(output_text)) == count
    report = {
        "schema": 1,
        "experiment": "swin4h_selected_cta_warp_fp8_mma_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "SWIN4H_EXACT_INPUT_FIRST_OUTPUT_MISMATCH_MMA_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "function": args.function,
        "target_cta": [args.target_x, args.target_y, 0],
        "target_warp_y": args.target_warp_y,
        "mma_count": count,
        "expected_count": args.expected_count,
        "lanes": LANES,
        "bytes_per_lane_mma": BYTES_PER_LANE_MMA,
        "trace_bytes": trace_bytes,
        "trace_param_offset": TRACE_PARAM_OFFSET,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
