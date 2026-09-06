#!/usr/bin/env python3
"""Export every FP8 MMA A/B/C/D fragment for one selected CTA through param +64."""

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


BYTES_PER_LANE_MMA = 40
LANES = 32


def instrument(text: str, function: str, cta_x: int = 0, cta_y: int = 0) -> tuple[str, int]:
    if cta_x < 0 or cta_y < 0:
        raise ValueError("CTA coordinates must be non-negative")
    entry = re.compile(
        rf"(?s)(\.visible\s+\.entry\s+{re.escape(function)}\s*\(.*?\)\s*"
        rf"(?:\.[^\n]*\n\s*)*\{{)"
    )
    setup = r"""
.reg .b32 %__mma_ck_lane, %__mma_ck_ctax, %__mma_ck_ctay, %__mma_ck_offset;
.reg .b64 %__mma_ck_base, %__mma_ck_address, %__mma_ck_wide;
.reg .pred %__mma_ck_px, %__mma_ck_py, %__mma_ck_pb, %__mma_ck_guard;
ld.param.u64 %__mma_ck_base, [FUNCTION_param_0+64];
mov.u32 %__mma_ck_lane, %laneid;
mov.u32 %__mma_ck_ctax, %ctaid.x;
mov.u32 %__mma_ck_ctay, %ctaid.y;
setp.eq.u32 %__mma_ck_px, %__mma_ck_ctax, CTA_X;
setp.eq.u32 %__mma_ck_py, %__mma_ck_ctay, CTA_Y;
setp.ne.u64 %__mma_ck_pb, %__mma_ck_base, 0;
and.pred %__mma_ck_guard, %__mma_ck_px, %__mma_ck_py;
and.pred %__mma_ck_guard, %__mma_ck_guard, %__mma_ck_pb;
""".replace("FUNCTION", function).replace("CTA_X", str(cta_x)).replace("CTA_Y", str(cta_y))
    output, entry_count = entry.subn(lambda match: match.group(1) + setup, text, count=1)
    if entry_count != 1:
        raise ValueError(f"entry {function} not found exactly once")
    mma_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal mma_index
        a = [match.group(f"a{i}") for i in range(4)]
        b = [match.group(f"b{i}") for i in range(2)]
        c = [match.group(f"c{i}") for i in range(2)]
        d = [match.group(f"d{i}") for i in range(2)]
        base = mma_index * LANES * BYTES_PER_LANE_MMA
        before = [
            f"mad.lo.u32 %__mma_ck_offset, %__mma_ck_lane, {BYTES_PER_LANE_MMA}, {base};",
            "cvt.u64.u32 %__mma_ck_wide, %__mma_ck_offset;",
            "add.u64 %__mma_ck_address, %__mma_ck_base, %__mma_ck_wide;",
        ]
        for offset, register in enumerate(a + b + c):
            before.append(
                f"@%__mma_ck_guard st.global.b32 [%__mma_ck_address+{offset * 4}], {register};")
        after = [
            f"@%__mma_ck_guard st.global.b32 [%__mma_ck_address+32], {d[0]};",
            f"@%__mma_ck_guard st.global.b32 [%__mma_ck_address+36], {d[1]};",
        ]
        mma_index += 1
        return "\n".join(before) + "\n" + match.group(0) + "\n" + "\n".join(after)

    output = FP8_MMA.sub(replace, output)
    return output, mma_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--cta-x", type=int, default=0)
    parser.add_argument("--cta-y", type=int, default=0)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, count = instrument(source.decode("utf-8"), args.function, args.cta_x, args.cta_y)
    output = output_text.encode("utf-8")
    expected_bytes = count * LANES * BYTES_PER_LANE_MMA
    passed = count == args.expected_count and len(FP8_MMA.findall(output_text)) == count
    report = {
        "schema": 1,
        "experiment": "slot_fp8_mma_register_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "INTERNAL_NUMERICAL_DIVERGENCE_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "function": args.function,
        "mma_count": count,
        "expected_count": args.expected_count,
        "cta": [args.cta_x, args.cta_y, 0],
        "lanes": LANES,
        "bytes_per_lane_mma": BYTES_PER_LANE_MMA,
        "checkpoint_bytes": expected_bytes,
        "checkpoint_param_offset": 64,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
