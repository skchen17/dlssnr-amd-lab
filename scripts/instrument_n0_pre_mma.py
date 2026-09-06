#!/usr/bin/env python3
"""Create a one-CTA N0 checkpoint that exports the first MMA A/B fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FIRST_MMA = re.compile(r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.f16\.f16\.f16")


CHECKPOINT = r"""
{
.reg .b32 %__pre_mma_lane;
.reg .b64 %__pre_mma_offset, %__pre_mma_base, %__pre_mma_output;
mov.u32 %__pre_mma_lane, %laneid;
mul.wide.u32 %__pre_mma_offset, %__pre_mma_lane, 96;
add.s64 %__pre_mma_base, %rd6, %__pre_mma_offset;
st.global.b32 [%__pre_mma_base], %r487;
st.global.b32 [%__pre_mma_base+4], %r488;
st.global.b32 [%__pre_mma_base+8], %r489;
st.global.b32 [%__pre_mma_base+12], %r490;
st.global.b32 [%__pre_mma_base+16], %r499;
st.global.b32 [%__pre_mma_base+20], %r500;
st.global.b32 [%__pre_mma_base+24], %r501;
st.global.b32 [%__pre_mma_base+28], %r502;
st.global.b32 [%__pre_mma_base+32], %r503;
st.global.b32 [%__pre_mma_base+36], %r504;
st.global.b32 [%__pre_mma_base+40], %r505;
st.global.b32 [%__pre_mma_base+44], %r506;
st.global.b32 [%__pre_mma_base+48], %r507;
st.global.b32 [%__pre_mma_base+52], %r508;
st.global.b32 [%__pre_mma_base+56], %r509;
st.global.b32 [%__pre_mma_base+60], %r510;
st.global.b32 [%__pre_mma_base+64], %r491;
st.global.b32 [%__pre_mma_base+68], %r492;
st.global.b32 [%__pre_mma_base+72], %r493;
st.global.b32 [%__pre_mma_base+76], %r494;
st.global.b32 [%__pre_mma_base+80], %r495;
st.global.b32 [%__pre_mma_base+84], %r496;
st.global.b32 [%__pre_mma_base+88], %r497;
st.global.b32 [%__pre_mma_base+92], %r498;
ld.param.b64 %__pre_mma_output, [%rd16+248];
st.global.b32 [%__pre_mma_output], 1;
}
ret;
}
"""


def instrument(text: str) -> str:
    matches = list(FIRST_MMA.finditer(text))
    if len(matches) != 16:
        raise ValueError(f"expected 16 initial f16 MMA statements, found {len(matches)}")
    prefix = text[: matches[0].start()]
    normalized, changed = re.subn(r"(?m)^\.version 9\.4\r?$", ".version 8.7", prefix)
    if changed != 1:
        raise ValueError(f"expected one PTX version directive, changed {changed}")
    return normalized + CHECKPOINT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output = instrument(source.decode("utf-8")).encode("utf-8")
    report = {
        "schema": 1,
        "experiment": "n0_pre_mma_fragment_checkpoint",
        "status": "PASS",
        "classification": "CONTROLLED_PTX_PREFIX_INSTRUMENTATION",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "ptx_version": "8.7",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "bytes_per_lane": 96,
        "export_order": [
            "A_r487_r490",
            "A_r499_r502",
            "A_r503_r506",
            "A_r507_r510",
            "B_r491_r494",
            "B_r495_r498",
        ],
        "boundary": "checkpoint is immediately before the first f16 MMA",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
