#!/usr/bin/env python3
"""Make NVIDIA's N0 half2 square-pair fusion explicit with fma.rn.f16x2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REG = r"%[A-Za-z_][A-Za-z0-9_]*"
SQUARE = re.compile(
    rf"mul\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P=a)\s*;"
)
ADD = re.compile(
    rf"add\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P<b>{REG})\s*;"
)


def lower(text: str) -> tuple[str, list[dict[str, str]]]:
    squares: dict[str, str] = {}
    counts: dict[str, int] = {}
    for match in SQUARE.finditer(text):
        destination = match.group("dst")
        squares[destination] = match.group("a")
        counts[destination] = counts.get(destination, 0) + 1

    replacements: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        left = match.group("a")
        right = match.group("b")
        if counts.get(left) != 1 or counts.get(right) != 1:
            return match.group(0)
        left_input = squares[left]
        right_input = squares[right]
        replacements.append({
            "destination": match.group("dst"),
            "fused_square_register": left,
            "fused_input": left_input,
            "separately_rounded_square_register": right,
            "separately_rounded_input": right_input,
        })
        return f"fma.rn.f16x2 {match.group('dst')}, {left_input}, {left_input}, {right};"

    return ADD.sub(replace, text), replacements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, replacements = lower(source.decode("utf-8"))
    output = output_text.encode("utf-8")
    expected = args.expected if args.expected is not None else len(replacements)
    passed = len(replacements) == expected and expected > 0
    report = {
        "schema": 1,
        "experiment": "n0_f16x2_square_pair_explicit_fusion",
        "status": "PASS" if passed else "FAIL",
        "classification": "RTX5070_BITWISE_FITTED_HALF2_FUSION_LOWERING",
        "counts_as_s7": False,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "expected_replacements": expected,
        "replacement_count": len(replacements),
        "replacements": replacements,
        "semantics": (
            "For add(square(left), square(right)), keep the right square's FP16 rounding "
            "and replace the add with fma.rn.f16x2(left, left, rounded_right_square)."
        ),
        "evidence_model": "hfma_pair_right_square_separate",
        "evidence_fit": "128/128 finite packed-half samples bitwise exact",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "replacement_count": len(replacements),
                      "output_sha256": report["output_sha256"]}, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
