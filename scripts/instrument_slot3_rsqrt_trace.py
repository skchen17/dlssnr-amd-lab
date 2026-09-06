#!/usr/bin/env python3
"""Export all slot-3 approximate-rsqrt inputs, f32 results and f16 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
WORDS_PER_BLOCK = 4  # packed f16 input, f32 low/high result, packed f16 output
EXPECTED_BLOCKS = 16
BYTES_PER_LANE = WORDS_PER_BLOCK * EXPECTED_BLOCKS * 4
RSQRT_BLOCK = re.compile(
    r"(?s)(\{\.reg\.b16 hl, hu;\s*"
    r"\.reg\.b32 fl, fu;\s*"
    r"mov\.b32 \{hl, hu\}, (?P<src>%r\d+);\s*"
    r"cvt\.f32\.f16 fl, hl;\s*"
    r"cvt\.f32\.f16 fu, hu;\s*"
    r"rsqrt\.approx\.ftz\.f32 fl, fl;\s*"
    r"rsqrt\.approx\.ftz\.f32 fu, fu;\s*"
    r"cvt\.rn\.f16\.f32 hl, fl;\s*"
    r"cvt\.rn\.f16\.f32 hu, fu;\s*"
    r"mov\.b32 (?P<dst>%r\d+), \{hl, hu\};\s*)(\})"
)


def instrument(text: str, function: str, cta_x: int = 2,
               cta_y: int = 0) -> tuple[str, list[dict]]:
    if cta_x < 0 or cta_y < 0:
        raise ValueError("CTA coordinates must be non-negative")
    entry = re.compile(
        rf"(?s)(\.visible\s+\.entry\s+{re.escape(function)}\s*\(.*?\)\s*"
        rf"(?:\.[^\n]*\n\s*)*\{{)"
    )
    setup = r"""
.reg .b32 %__rsq_lane, %__rsq_ctax, %__rsq_ctay;
.reg .b64 %__rsq_base, %__rsq_address, %__rsq_wide;
.reg .pred %__rsq_px, %__rsq_py, %__rsq_pb, %__rsq_guard;
ld.param.u64 %__rsq_base, [FUNCTION_param_0+64];
mov.u32 %__rsq_lane, %laneid;
mov.u32 %__rsq_ctax, %ctaid.x;
mov.u32 %__rsq_ctay, %ctaid.y;
setp.eq.u32 %__rsq_px, %__rsq_ctax, CTA_X;
setp.eq.u32 %__rsq_py, %__rsq_ctay, CTA_Y;
setp.ne.u64 %__rsq_pb, %__rsq_base, 0;
and.pred %__rsq_guard, %__rsq_px, %__rsq_py;
and.pred %__rsq_guard, %__rsq_guard, %__rsq_pb;
mul.wide.u32 %__rsq_wide, %__rsq_lane, BYTES_PER_LANE;
add.u64 %__rsq_address, %__rsq_base, %__rsq_wide;
""".replace("FUNCTION", function).replace("CTA_X", str(cta_x)).replace(
        "CTA_Y", str(cta_y)
    ).replace("BYTES_PER_LANE", str(BYTES_PER_LANE))
    output, entry_count = entry.subn(
        lambda match: match.group(1) + setup, text, count=1
    )
    if entry_count != 1:
        raise ValueError(f"entry {function} not found exactly once")
    blocks: list[dict] = []

    def replace(match: re.Match[str]) -> str:
        index = len(blocks)
        source = match.group("src")
        destination = match.group("dst")
        blocks.append({"index": index, "source": source, "destination": destination})
        base = index * WORDS_PER_BLOCK * 4
        stores = "\n".join([
            f"@%__rsq_guard st.global.b32 [%__rsq_address+{base}], {source};",
            f"@%__rsq_guard st.global.b32 [%__rsq_address+{base + 4}], fl;",
            f"@%__rsq_guard st.global.b32 [%__rsq_address+{base + 8}], fu;",
            f"@%__rsq_guard st.global.b32 [%__rsq_address+{base + 12}], {destination};",
        ])
        return match.group(1) + stores + "\n" + match.group(4)

    output = RSQRT_BLOCK.sub(replace, output)
    if len(blocks) != EXPECTED_BLOCKS:
        raise ValueError(
            f"expected {EXPECTED_BLOCKS} rsqrt blocks, found {len(blocks)}"
        )
    return output, blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--cta-x", type=int, default=2)
    parser.add_argument("--cta-y", type=int, default=0)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, blocks = instrument(
        source.decode("utf-8"), args.function, args.cta_x, args.cta_y
    )
    output = output_text.encode("utf-8")
    report = {
        "schema": 1,
        "experiment": "slot3_rsqrt_register_trace_instrumentation",
        "status": "PASS",
        "classification": "ARCHITECTURE_APPROX_RSQRT_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "function": args.function,
        "cta": [args.cta_x, args.cta_y, 0],
        "blocks": blocks,
        "lanes": LANES,
        "words_per_block": WORDS_PER_BLOCK,
        "bytes_per_lane": BYTES_PER_LANE,
        "checkpoint_bytes": LANES * BYTES_PER_LANE,
        "checkpoint_param_offset": 64,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
