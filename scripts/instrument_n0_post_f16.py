#!/usr/bin/env python3
"""Create a one-CTA N0 checkpoint immediately after the first 16 f16 MMAs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


F16_MMA = re.compile(
    r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.f16\.f16\.f16.*?;",
    re.DOTALL,
)
OUTPUT_REGISTERS = [
    544, 547, 550, 553, 556, 559, 562, 565,
    568, 571, 574, 577, 580, 583, 586, 589,
    592, 595, 598, 601, 604, 607, 610, 613,
    616, 619, 622, 625, 628, 631, 634, 637,
]


def checkpoint_text() -> str:
    stores = "\n".join(
        f"st.global.b32 [%__post_f16_base+{index * 4}], %r{register};"
        for index, register in enumerate(OUTPUT_REGISTERS)
    )
    return f"""
{{
.reg .b32 %__post_f16_lane;
.reg .b64 %__post_f16_offset, %__post_f16_base, %__post_f16_output;
mov.u32 %__post_f16_lane, %laneid;
mul.wide.u32 %__post_f16_offset, %__post_f16_lane, 128;
add.s64 %__post_f16_base, %rd6, %__post_f16_offset;
{stores}
ld.param.b64 %__post_f16_output, [%rd16+248];
st.global.b32 [%__post_f16_output], 1;
}}
ret;
}}
"""


def instrument(text: str) -> str:
    matches = list(F16_MMA.finditer(text))
    if len(matches) != 16:
        raise ValueError(f"expected 16 f16 MMA statements, found {len(matches)}")
    prefix = text[:matches[-1].end()]
    normalized, changed = re.subn(r"(?m)^\.version 9\.4\r?$", ".version 8.7", prefix)
    if changed != 1:
        raise ValueError(f"expected one PTX version directive, changed {changed}")
    return normalized + checkpoint_text()


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
        "experiment": "n0_post_f16_fragment_checkpoint",
        "status": "PASS",
        "classification": "CONTROLLED_PTX_POST_MMA_INSTRUMENTATION",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "ptx_version": "8.7",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "mma_operations": 16,
        "bytes_per_lane": 128,
        "exported_bytes": 4096,
        "boundary": "immediately after the first 16 f16 MMA operations",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
