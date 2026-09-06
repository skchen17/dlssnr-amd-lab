#!/usr/bin/env python3
"""Replace architecture-dependent approximate reciprocal math with RN PTX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REG = r"%?[A-Za-z_][A-Za-z0-9_]*"
RSQRT = re.compile(rf"rsqrt\.approx\.ftz\.f32\s+(?P<dst>{REG})\s*,\s*(?P<src>{REG})\s*;")
RCP = re.compile(rf"rcp\.approx\.ftz\.f32\s+(?P<dst>{REG})\s*,\s*(?P<src>{REG})\s*;")


def lower(text: str) -> tuple[str, int, int]:
    rsqrt_count = 0
    rcp_count = 0

    def replace_rsqrt(match: re.Match[str]) -> str:
        nonlocal rsqrt_count
        temporary = f"%__accurate_rsqrt_{rsqrt_count}"
        rsqrt_count += 1
        return (
            "{\n"
            f".reg .f32 {temporary};\n"
            f"sqrt.rn.ftz.f32 {temporary}, {match.group('src')};\n"
            f"div.rn.ftz.f32 {match.group('dst')}, 0f3F800000, {temporary};\n"
            "}"
        )

    def replace_rcp(match: re.Match[str]) -> str:
        nonlocal rcp_count
        rcp_count += 1
        return f"div.rn.ftz.f32 {match.group('dst')}, 0f3F800000, {match.group('src')};"

    lowered = RSQRT.sub(replace_rsqrt, text)
    lowered = RCP.sub(replace_rcp, lowered)
    return lowered, rsqrt_count, rcp_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-rsqrt", type=int, required=True)
    parser.add_argument("--expected-rcp", type=int, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    lowered, rsqrt_count, rcp_count = lower(source.decode("utf-8"))
    output = lowered.encode("utf-8")
    remaining = len(RSQRT.findall(lowered)) + len(RCP.findall(lowered))
    passed = (rsqrt_count == args.expected_rsqrt and rcp_count == args.expected_rcp
              and remaining == 0)
    report = {
        "schema": 1,
        "experiment": "ptx_architecture_independent_reciprocal_math_lowering",
        "status": "PASS" if passed else "FAIL",
        "classification": "NUMERICAL_STABILITY_PTX_REWRITE",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "rsqrt_approx_lowered": rsqrt_count,
        "rcp_approx_lowered": rcp_count,
        "expected_rsqrt": args.expected_rsqrt,
        "expected_rcp": args.expected_rcp,
        "remaining_approx_reciprocals": remaining,
        "semantics": "RN sqrt/div is a valid higher-accuracy implementation of approximate reciprocal operations",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
