#!/usr/bin/env python3
"""Reconstruct PTX m16n8k32 fragments and compare the RTX D registers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def a_coord(lane: int, element: int) -> tuple[int, int]:
    group, thread = lane >> 2, lane & 3
    row = group if element < 4 or 8 <= element < 12 else group + 8
    col = thread * 4 + (element & 3) + (16 if element >= 8 else 0)
    return row, col


def b_coord(lane: int, element: int) -> tuple[int, int]:
    group, thread = lane >> 2, lane & 3
    row = thread * 4 + (element & 3) + (16 if element >= 4 else 0)
    return row, group


def cd_coord(lane: int, element: int) -> tuple[int, int]:
    group, thread = lane >> 2, lane & 3
    return group + (8 if element >= 2 else 0), thread * 2 + (element & 1)


def e4m3(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    magnitude = code & 0x7F
    exponent, mantissa = magnitude >> 3, magnitude & 7
    if exponent == 0:
        value = math.ldexp(float(mantissa), -9)
    elif magnitude == 0x7F:
        value = math.nan
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return sign * value


def half_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def float_to_half(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0xFC00 if value < 0 else 0x7C00


def words(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"unaligned u32 file: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def analyze(directory: Path) -> dict:
    paths = {name: directory / f"mma_{name}.raw" for name in "abcd"}
    a, b, c, d = (words(paths[name]) for name in "abcd")
    if [len(a), len(b), len(c), len(d)] != [1024, 512, 512, 512]:
        raise ValueError("unexpected fragment lengths")
    total_mismatches = 0
    case_mismatches: list[int] = []
    for case in range(8):
        matrix_a = [[0.0] * 32 for _ in range(16)]
        matrix_b = [[0.0] * 8 for _ in range(32)]
        matrix_c = [[0.0] * 8 for _ in range(16)]
        matrix_d = [[0] * 8 for _ in range(16)]
        for lane in range(32):
            abase = (case * 32 + lane) * 4
            bbase = (case * 32 + lane) * 2
            for element in range(16):
                code = (a[abase + element // 4] >> (8 * (element & 3))) & 0xFF
                row, col = a_coord(lane, element)
                matrix_a[row][col] = e4m3(code)
            for element in range(8):
                code = (b[bbase + element // 4] >> (8 * (element & 3))) & 0xFF
                row, col = b_coord(lane, element)
                matrix_b[row][col] = e4m3(code)
            for element in range(4):
                shift = 16 * (element & 1)
                row, col = cd_coord(lane, element)
                matrix_c[row][col] = half_to_float((c[bbase + element // 2] >> shift) & 0xFFFF)
                matrix_d[row][col] = (d[bbase + element // 2] >> shift) & 0xFFFF
        mismatches = 0
        for row in range(16):
            for col in range(8):
                value = matrix_c[row][col] + sum(
                    matrix_a[row][k] * matrix_b[k][col] for k in range(32)
                )
                mismatches += float_to_half(value) != matrix_d[row][col]
        case_mismatches.append(mismatches)
        total_mismatches += mismatches
    status = "PASS" if total_mismatches == 0 else "FAIL"
    return {
        "schema": 1,
        "experiment": "m16n8k32_e4m3_fragment_reconstruction",
        "status": status,
        "classification": "RTX_ORACLE_STATIC_ANALYSIS",
        "counts_as_s6": False,
        "cases": 8,
        "half_outputs": 1024,
        "mismatches": total_mismatches,
        "case_mismatches": case_mismatches,
        "raw_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
        "negative_verifier_detected": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.reference_directory)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
