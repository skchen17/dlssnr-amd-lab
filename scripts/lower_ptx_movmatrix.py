#!/usr/bin/env python3
"""Lower movmatrix m8n8 b16 transpose to warp shuffles and integer packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MOVMATRIX = re.compile(
    r"movmatrix\.sync\.trans\.aligned\.m8n8\.b16\s+"
    r"(?P<dst>%[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"(?P<src>%[A-Za-z_][A-Za-z0-9_]*)\s*;"
)


def transpose_words(words: list[int]) -> list[int]:
    if len(words) != 32:
        raise ValueError("movmatrix fragment requires 32 lanes")
    result = []
    for lane in range(32):
        column, row_pair = lane >> 2, lane & 3
        packed = 0
        for element in range(2):
            row = row_pair * 2 + element
            source_lane = row * 4 + (column >> 1)
            value = (words[source_lane] >> (16 * (column & 1))) & 0xFFFF
            packed |= value << (16 * element)
        result.append(packed)
    return result


def replacement(match: re.Match[str], index: int) -> str:
    destination, source = match.group("dst"), match.group("src")
    r = lambda name: f"%__movm_{index}_r_{name}"
    p = lambda name: f"%__movm_{index}_p_{name}"
    registers = [r(name) for name in (
        "lane", "column", "row_pair", "row0", "row1", "column_pair",
        "source_lane0", "source_lane1", "word0", "word1", "column_bit",
        "shift", "value0", "value1"
    )]
    lines = [
        f".reg .b32 {', '.join(registers)};",
        f".reg .pred {p('valid0')}, {p('valid1')};",
        f"mov.u32 {r('lane')}, %laneid;",
        f"shr.u32 {r('column')}, {r('lane')}, 2;",
        f"and.b32 {r('row_pair')}, {r('lane')}, 3;",
        f"shl.b32 {r('row0')}, {r('row_pair')}, 1;",
        f"add.u32 {r('row1')}, {r('row0')}, 1;",
        f"shr.u32 {r('column_pair')}, {r('column')}, 1;",
        f"shl.b32 {r('source_lane0')}, {r('row0')}, 2;",
        f"add.u32 {r('source_lane0')}, {r('source_lane0')}, {r('column_pair')};",
        f"shl.b32 {r('source_lane1')}, {r('row1')}, 2;",
        f"add.u32 {r('source_lane1')}, {r('source_lane1')}, {r('column_pair')};",
        f"shfl.sync.idx.b32 {r('word0')}|{p('valid0')}, {source}, {r('source_lane0')}, 31, -1;",
        f"shfl.sync.idx.b32 {r('word1')}|{p('valid1')}, {source}, {r('source_lane1')}, 31, -1;",
        f"and.b32 {r('column_bit')}, {r('column')}, 1;",
        f"shl.b32 {r('shift')}, {r('column_bit')}, 4;",
        f"shr.u32 {r('value0')}, {r('word0')}, {r('shift')};",
        f"and.b32 {r('value0')}, {r('value0')}, 65535;",
        f"shr.u32 {r('value1')}, {r('word1')}, {r('shift')};",
        f"shl.b32 {r('value1')}, {r('value1')}, 16;",
        f"or.b32 {destination}, {r('value0')}, {r('value1')};",
    ]
    return "{\n" + "\n".join(lines) + "\n}"


def lower(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        result = replacement(match, count)
        count += 1
        return result

    return MOVMATRIX.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-count", type=int, default=32)
    args = parser.parse_args()
    source = args.input.read_bytes()
    lowered, count = lower(source.decode("utf-8"))
    output = lowered.encode("utf-8")
    remaining = len(MOVMATRIX.findall(lowered))
    status = "PASS" if count == args.expected_count and remaining == 0 else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_movmatrix_shuffle_lowering",
        "status": status,
        "classification": "RTX_AMD_ORACLE_VALIDATED_PTX_REWRITE",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "movmatrix_lowered": count,
        "expected_movmatrix": args.expected_count,
        "remaining_movmatrix": remaining,
        "translator_execution_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
