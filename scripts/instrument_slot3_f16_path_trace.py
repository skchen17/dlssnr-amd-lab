#!/usr/bin/env python3
"""Export the packed-FP16 path feeding slot-3 MMA 216 for one CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
REGISTERS = [
    # Inputs to the first packed-FP16 multiplication stage.
    1751, 1753, 1755, 1757, 1759, 1761, 1763, 1765,
    # Alternating first-stage scale factors.
    2002, 2004,
    # First-stage products.
    2056, 2058, 2060, 2062, 2064, 2066, 2068, 2070,
    # FP32 scale source and duplicated packed-FP16 scale.
    2021, 2024,
    # Second-stage products converted into the MMA 216-223 A fragment.
    2103, 2105, 2104, 2106, 2107, 2109, 2108, 2110,
]
BYTES_PER_LANE = len(REGISTERS) * 4
ANCHOR = "cvt.rn.satfinite.e4m3x2.f16x2 %rs224, %r2110;"


def instrument(text: str, function: str, cta_x: int = 2,
               cta_y: int = 0) -> tuple[str, int]:
    if cta_x < 0 or cta_y < 0:
        raise ValueError("CTA coordinates must be non-negative")
    entry = re.compile(
        rf"(?s)(\.visible\s+\.entry\s+{re.escape(function)}\s*\(.*?\)\s*"
        rf"(?:\.[^\n]*\n\s*)*\{{)"
    )
    setup = r"""
.reg .b32 %__f16p_lane, %__f16p_ctax, %__f16p_ctay, %__f16p_offset;
.reg .b64 %__f16p_base, %__f16p_address, %__f16p_wide;
.reg .pred %__f16p_px, %__f16p_py, %__f16p_pb, %__f16p_guard;
ld.param.u64 %__f16p_base, [FUNCTION_param_0+64];
mov.u32 %__f16p_lane, %laneid;
mov.u32 %__f16p_ctax, %ctaid.x;
mov.u32 %__f16p_ctay, %ctaid.y;
setp.eq.u32 %__f16p_px, %__f16p_ctax, CTA_X;
setp.eq.u32 %__f16p_py, %__f16p_ctay, CTA_Y;
setp.ne.u64 %__f16p_pb, %__f16p_base, 0;
and.pred %__f16p_guard, %__f16p_px, %__f16p_py;
and.pred %__f16p_guard, %__f16p_guard, %__f16p_pb;
""".replace("FUNCTION", function).replace("CTA_X", str(cta_x)).replace(
        "CTA_Y", str(cta_y)
    )
    output, entry_count = entry.subn(
        lambda match: match.group(1) + setup, text, count=1
    )
    if entry_count != 1:
        raise ValueError(f"entry {function} not found exactly once")
    if output.count(ANCHOR) != 1:
        raise ValueError("slot-3 packed-FP16 checkpoint anchor not found exactly once")
    stores = [
        f"mad.lo.u32 %__f16p_offset, %__f16p_lane, {BYTES_PER_LANE}, 0;",
        "cvt.u64.u32 %__f16p_wide, %__f16p_offset;",
        "add.u64 %__f16p_address, %__f16p_base, %__f16p_wide;",
    ]
    stores.extend(
        f"@%__f16p_guard st.global.b32 [%__f16p_address+{index * 4}], %r{register};"
        for index, register in enumerate(REGISTERS)
    )
    checkpoint = ANCHOR + "\n" + "\n".join(stores)
    return output.replace(ANCHOR, checkpoint, 1), entry_count


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
    output_text, _ = instrument(
        source.decode("utf-8"), args.function, args.cta_x, args.cta_y
    )
    output = output_text.encode("utf-8")
    report = {
        "schema": 1,
        "experiment": "slot3_f16_path_register_trace_instrumentation",
        "status": "PASS",
        "classification": "FIRST_DIVERGENT_ACTIVATION_PATH_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "function": args.function,
        "cta": [args.cta_x, args.cta_y, 0],
        "registers": REGISTERS,
        "lanes": LANES,
        "bytes_per_lane": BYTES_PER_LANE,
        "checkpoint_bytes": LANES * BYTES_PER_LANE,
        "checkpoint_param_offset": 64,
        "anchor": ANCHOR,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
