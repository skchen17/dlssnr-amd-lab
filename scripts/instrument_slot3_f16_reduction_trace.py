#!/usr/bin/env python3
"""Export the slot-3 packed-FP16 square/reduction tree feeding rsqrt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
REGISTERS = [
    1751, 1753, 1755, 1757, 1759, 1761, 1763, 1765,
    1807, 1810, 1813, 1816, 1808, 1811, 1814, 1817,
    1868, 1870, 1867, 1869,
    1871, 1877, 1872, 1878, 1873, 1879, 1874, 1880,
    1875, 1881, 1876, 1882, 1922, 1924,
]
BYTES_PER_LANE = len(REGISTERS) * 4
ANCHOR = "{add.f16x2 %r1924,%r1881,%r1882;\n}"


def instrument(text: str, function: str, cta_x: int = 2,
               cta_y: int = 0) -> tuple[str, int]:
    entry = re.compile(
        rf"(?s)(\.visible\s+\.entry\s+{re.escape(function)}\s*\(.*?\)\s*"
        rf"(?:\.[^\n]*\n\s*)*\{{)"
    )
    setup = r"""
.reg .b32 %__f16r_lane, %__f16r_ctax, %__f16r_ctay, %__f16r_offset;
.reg .b64 %__f16r_base, %__f16r_address, %__f16r_wide;
.reg .pred %__f16r_px, %__f16r_py, %__f16r_pb, %__f16r_guard;
ld.param.u64 %__f16r_base, [FUNCTION_param_0+64];
mov.u32 %__f16r_lane, %laneid;
mov.u32 %__f16r_ctax, %ctaid.x;
mov.u32 %__f16r_ctay, %ctaid.y;
setp.eq.u32 %__f16r_px, %__f16r_ctax, CTA_X;
setp.eq.u32 %__f16r_py, %__f16r_ctay, CTA_Y;
setp.ne.u64 %__f16r_pb, %__f16r_base, 0;
and.pred %__f16r_guard, %__f16r_px, %__f16r_py;
and.pred %__f16r_guard, %__f16r_guard, %__f16r_pb;
""".replace("FUNCTION", function).replace("CTA_X", str(cta_x)).replace(
        "CTA_Y", str(cta_y)
    )
    output, count = entry.subn(lambda match: match.group(1) + setup, text, count=1)
    if count != 1:
        raise ValueError(f"entry {function} not found exactly once")
    if output.count(ANCHOR) != 1:
        raise ValueError("reduction checkpoint anchor not found exactly once")
    stores = [
        f"mad.lo.u32 %__f16r_offset, %__f16r_lane, {BYTES_PER_LANE}, 0;",
        "cvt.u64.u32 %__f16r_wide, %__f16r_offset;",
        "add.u64 %__f16r_address, %__f16r_base, %__f16r_wide;",
    ]
    stores.extend(
        f"@%__f16r_guard st.global.b32 [%__f16r_address+{i * 4}], %r{register};"
        for i, register in enumerate(REGISTERS)
    )
    return output.replace(ANCHOR, ANCHOR + "\n" + "\n".join(stores), 1), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path); parser.add_argument("--function", required=True)
    parser.add_argument("--cta-x", type=int, default=2); parser.add_argument("--cta-y", type=int, default=0)
    args = parser.parse_args(); source = args.input.read_bytes()
    output_text, _ = instrument(source.decode("utf-8"), args.function, args.cta_x, args.cta_y)
    output = output_text.encode("utf-8")
    report = {"schema": 1, "experiment": "slot3_f16_reduction_trace_instrumentation",
              "status": "PASS", "classification": "PACKED_F16_REDUCTION_DIVERGENCE_TRACE",
              "source_sha256": hashlib.sha256(source).hexdigest(),
              "output_sha256": hashlib.sha256(output).hexdigest(), "function": args.function,
              "cta": [args.cta_x, args.cta_y, 0], "registers": REGISTERS, "lanes": LANES,
              "bytes_per_lane": BYTES_PER_LANE, "checkpoint_bytes": LANES * BYTES_PER_LANE,
              "checkpoint_param_offset": 64, "anchor": ANCHOR}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":"))); return 0


if __name__ == "__main__": raise SystemExit(main())
