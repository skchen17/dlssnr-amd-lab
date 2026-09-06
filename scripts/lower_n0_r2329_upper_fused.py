#!/usr/bin/env python3
"""Add a scoped FP32 fused upper-envelope candidate for N0 r2329."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGET = re.compile(r"\{add\.f16x2 %r2329,%r2276,%r2277;\s*\}")


def candidate_block(policy: str) -> str:
    lines = [
        "{",
        ".reg .b16 %__r2329_a0, %__r2329_a1, %__r2329_b0, %__r2329_b1, %__r2329_h0, %__r2329_h1, %__r2329_out;",
        ".reg .b32 %__r2329_bits0, %__r2329_bits1, %__r2329_sh2_0, %__r2329_sh2_1, %__r2329_sh1_0, %__r2329_sh1_1, %__r2329_candidate;",
        ".reg .f32 %__r2329_fa0, %__r2329_fa1, %__r2329_fb0, %__r2329_fb1, %__r2329_lane0, %__r2329_lane1, %__r2329_x2_0, %__r2329_x2_1, %__r2329_x1_0, %__r2329_x1_1, %__r2329_sum;",
        "mov.b32 {%__r2329_a0, %__r2329_a1}, %r2268;",
        "mov.b32 {%__r2329_b0, %__r2329_b1}, %r2269;",
        "cvt.f32.f16 %__r2329_fa0, %__r2329_a0;",
        "cvt.f32.f16 %__r2329_fa1, %__r2329_a1;",
        "cvt.f32.f16 %__r2329_fb0, %__r2329_b0;",
        "cvt.f32.f16 %__r2329_fb1, %__r2329_b1;",
        "add.rn.f32 %__r2329_lane0, %__r2329_fa0, %__r2329_fb0;",
        "add.rn.f32 %__r2329_lane1, %__r2329_fa1, %__r2329_fb1;",
        "mov.b32 %__r2329_bits0, %__r2329_lane0;",
        "mov.b32 %__r2329_bits1, %__r2329_lane1;",
        "shfl.sync.bfly.b32 %__r2329_sh2_0, %__r2329_bits0, %r2253, %r2254, %r2255;",
        "shfl.sync.bfly.b32 %__r2329_sh2_1, %__r2329_bits1, %r2253, %r2254, %r2255;",
        "mov.b32 %__r2329_x2_0, %__r2329_sh2_0;",
        "mov.b32 %__r2329_x2_1, %__r2329_sh2_1;",
        "add.rn.f32 %__r2329_x2_0, %__r2329_lane0, %__r2329_x2_0;",
        "add.rn.f32 %__r2329_x2_1, %__r2329_lane1, %__r2329_x2_1;",
        "mov.b32 %__r2329_bits0, %__r2329_x2_0;",
        "mov.b32 %__r2329_bits1, %__r2329_x2_1;",
        "shfl.sync.bfly.b32 %__r2329_sh1_0, %__r2329_bits0, %r2258, %r2254, %r2255;",
        "shfl.sync.bfly.b32 %__r2329_sh1_1, %__r2329_bits1, %r2258, %r2254, %r2255;",
        "mov.b32 %__r2329_x1_0, %__r2329_sh1_0;",
        "mov.b32 %__r2329_x1_1, %__r2329_sh1_1;",
        "add.rn.f32 %__r2329_x1_0, %__r2329_x2_0, %__r2329_x1_0;",
        "add.rn.f32 %__r2329_x1_1, %__r2329_x2_1, %__r2329_x1_1;",
        "cvt.rn.f16.f32 %__r2329_h0, %__r2329_x1_0;",
        "cvt.rn.f16.f32 %__r2329_h1, %__r2329_x1_1;",
        "cvt.f32.f16 %__r2329_x1_0, %__r2329_h0;",
        "cvt.f32.f16 %__r2329_x1_1, %__r2329_h1;",
        "add.rn.f32 %__r2329_sum, %__r2329_x1_0, %__r2329_x1_1;",
        "cvt.rn.f16.f32 %__r2329_out, %__r2329_sum;",
        "mov.b32 %__r2329_candidate, {%__r2329_out, %__r2329_out};",
    ]
    if policy == "max":
        lines.append("max.f16x2 %r2329, %r2329, %__r2329_candidate;")
    elif policy == "min":
        lines.append("min.f16x2 %r2329, %r2329, %__r2329_candidate;")
    elif policy == "fused":
        lines.append("mov.b32 %r2329, %__r2329_candidate;")
    else:
        raise ValueError(f"unsupported policy {policy}")
    lines.append("}")
    return "\n".join(lines)


def lower(text: str, policy: str = "max") -> tuple[str, int]:
    matches = list(TARGET.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected one r2329 target, found {len(matches)}")
    return TARGET.sub(lambda match: match.group(0) + "\n" + candidate_block(policy), text, count=1), 1


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("input",type=Path); parser.add_argument("output",type=Path); parser.add_argument("report",type=Path); parser.add_argument("--policy",choices=("max","min","fused"),default="max"); args=parser.parse_args()
    source=args.input.read_bytes(); output_text,count=lower(source.decode("utf-8"),args.policy); output=output_text.encode("utf-8")
    report={"schema":1,"experiment":"n0_r2329_fusion_candidate","status":"PASS","classification":"EMPIRICAL_HIDDEN_PRECISION_CANDIDATE","counts_as_s7":False,"source_sha256":hashlib.sha256(source).hexdigest().upper(),"output_sha256":hashlib.sha256(output).hexdigest().upper(),"targets_lowered":count,"policy":args.policy,"scope":"r2329 only; square/pair rounded, FP32 lane/xor2 fusion, xor1/final rounded"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(output);args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,separators=(",",":")));return 0


if __name__ == "__main__":
    raise SystemExit(main())
