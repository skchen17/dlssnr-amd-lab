#!/usr/bin/env python3
"""Snapshot slot-3 normalization registers at a later, output-safe E4M3 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LANES = 32
REGISTERS = [
    # Four packed-square partials for each of the two interleaved groups.
    1807, 1808, 1810, 1811, 1813, 1814, 1816, 1817,
    # First pair sums.
    1867, 1868, 1869, 1870,
    # Two warp reduction trees, including swapped final halves.
    1871, 1872, 1873, 1874, 1875, 1876,
    1877, 1878, 1879, 1880, 1881, 1882,
    # Completed sums, clamp constant/input, and packed rsqrt outputs.
    1922, 1924, 1902, 1962, 1964, 2002, 2004,
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
.reg .b32 %__dn_lane, %__dn_ctax, %__dn_ctay, %__dn_offset;
.reg .b64 %__dn_base, %__dn_address, %__dn_wide;
.reg .pred %__dn_px, %__dn_py, %__dn_pb, %__dn_guard;
ld.param.u64 %__dn_base, [FUNCTION_param_0+64];
mov.u32 %__dn_lane, %laneid;
mov.u32 %__dn_ctax, %ctaid.x;
mov.u32 %__dn_ctay, %ctaid.y;
setp.eq.u32 %__dn_px, %__dn_ctax, CTA_X;
setp.eq.u32 %__dn_py, %__dn_ctay, CTA_Y;
setp.ne.u64 %__dn_pb, %__dn_base, 0;
and.pred %__dn_guard, %__dn_px, %__dn_py;
and.pred %__dn_guard, %__dn_guard, %__dn_pb;
""".replace("FUNCTION", function).replace("CTA_X", str(cta_x)).replace(
        "CTA_Y", str(cta_y)
    )
    output, entry_count = entry.subn(
        lambda match: match.group(1) + setup, text, count=1
    )
    if entry_count != 1:
        raise ValueError(f"entry {function} not found exactly once")
    if output.count(ANCHOR) != 1:
        raise ValueError("delayed normalization snapshot anchor not found exactly once")
    stores = [
        f"mad.lo.u32 %__dn_offset, %__dn_lane, {BYTES_PER_LANE}, 0;",
        "cvt.u64.u32 %__dn_wide, %__dn_offset;",
        "add.u64 %__dn_address, %__dn_base, %__dn_wide;",
    ]
    stores.extend(
        f"@%__dn_guard st.global.b32 [%__dn_address+{index * 4}], %r{register};"
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
        "experiment": "slot3_delayed_norm_snapshot_instrumentation",
        "status": "PASS",
        "classification": "OUTPUT_SAFE_DELAYED_NORMALIZATION_SNAPSHOT",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "function": args.function,
        "cta": [args.cta_x, args.cta_y, 0],
        "registers": REGISTERS,
        "lanes": LANES,
        "bytes_per_lane": BYTES_PER_LANE,
        "checkpoint_bytes": LANES * BYTES_PER_LANE,
        "checkpoint_param_offset": 64,
        "anchor": ANCHOR,
        "probe_policy": "all stores occur at the previously output-preserving E4M3 boundary",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
