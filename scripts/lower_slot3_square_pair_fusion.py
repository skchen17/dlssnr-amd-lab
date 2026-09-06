#!/usr/bin/env python3
"""Make selected slot-3 packed-half square-pair contractions explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REG = r"%[A-Za-z_][A-Za-z0-9_]*"
SQUARE = re.compile(rf"mul\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P=a)\s*;")
ADD = re.compile(rf"add\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P<b>{REG})\s*;")
TARGETS = ("%r1868", "%r1870", "%r1867", "%r1869")


def lower(text: str, direction: str,
          selected_targets: tuple[str, ...] = TARGETS) -> tuple[str, list[dict[str, str]]]:
    if direction not in ("fuse_left", "fuse_right"):
        raise ValueError(f"unknown direction: {direction}")
    unknown = set(selected_targets) - set(TARGETS)
    if unknown:
        raise ValueError("unknown target(s): " + ", ".join(sorted(unknown)))
    squares: dict[str, str] = {}
    counts: dict[str, int] = {}
    for match in SQUARE.finditer(text):
        destination = match.group("dst")
        squares[destination] = match.group("a")
        counts[destination] = counts.get(destination, 0) + 1
    wanted = set(selected_targets)
    replacements: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        destination = match.group("dst")
        if destination not in wanted:
            return match.group(0)
        left, right = match.group("a"), match.group("b")
        if counts.get(left) != 1 or counts.get(right) != 1:
            raise ValueError(f"target {destination} is not fed by two unique squares")
        left_input, right_input = squares[left], squares[right]
        if direction == "fuse_right":
            fused_input, rounded_square = right_input, left
        else:
            fused_input, rounded_square = left_input, right
        replacements.append({
            "destination": destination,
            "left_square": left,
            "right_square": right,
            "fused_input": fused_input,
            "separately_rounded_square": rounded_square,
            "direction": direction,
        })
        return (f"fma.rn.f16x2 {destination}, {fused_input}, {fused_input}, "
                f"{rounded_square};")

    output = ADD.sub(replace, text)
    if {item["destination"] for item in replacements} != wanted:
        raise ValueError("not every requested square-pair target was replaced")
    return output, replacements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--direction", choices=("fuse_left", "fuse_right"), required=True)
    parser.add_argument("--target", action="append", choices=TARGETS)
    args = parser.parse_args()
    source = args.input.read_bytes()
    selected = tuple(args.target) if args.target else TARGETS
    output_text, replacements = lower(source.decode("utf-8"), args.direction, selected)
    output = output_text.encode("utf-8")
    report = {
        "schema": 1, "experiment": "slot3_square_pair_fusion_candidate",
        "status": "PASS", "classification": "LOCAL_AMD_PROPAGATION_CANDIDATE",
        "counts_as_s7": False, "direction": args.direction,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "replacement_count": len(replacements), "replacements": replacements,
        "acceptance": "rank by same-capture slot-6 NRMSE in a complete injection-free graph",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
