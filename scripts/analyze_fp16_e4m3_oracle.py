#!/usr/bin/env python3
"""Analyze an exhaustive NVIDIA binary16 -> E4M3 conversion table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def encode_half_bits(bits: int, preserve_nan_sign: bool) -> int:
    sign = 0x80 if bits & 0x8000 else 0
    exponent = (bits >> 10) & 31
    mantissa16 = bits & 1023
    if exponent == 31 and mantissa16:
        return sign | 0x7F if preserve_nan_sign else 0x7F
    value = struct.unpack("<e", struct.pack("<H", bits))[0]
    magnitude = abs(value)
    if magnitude >= 448.0:
        return sign | 0x7E
    if magnitude < 0.015625:
        mantissa8 = round(magnitude * 512.0)
        return sign | (0x08 if mantissa8 >= 8 else mantissa8)
    fraction, exponent_plus_one = math.frexp(magnitude)
    encoded_exponent = exponent_plus_one - 1 + 7
    mantissa8 = round((fraction * 2.0 - 1.0) * 8.0)
    if mantissa8 == 8:
        mantissa8 = 0
        encoded_exponent += 1
    if encoded_exponent > 15 or (encoded_exponent == 15 and mantissa8 >= 7):
        return sign | 0x7E
    return sign | (encoded_exponent << 3) | mantissa8


def analyze(raw: bytes) -> dict:
    if len(raw) != 65536:
        raise ValueError(f"expected 65536 bytes, got {len(raw)}")
    initial = bytes(encode_half_bits(i, True) for i in range(65536))
    corrected = bytes(encode_half_bits(i, False) for i in range(65536))
    initial_indices = [i for i, (a, b) in enumerate(zip(raw, initial)) if a != b]
    corrected_indices = [i for i, (a, b) in enumerate(zip(raw, corrected)) if a != b]
    swapped_count = sum(raw[i] != corrected[i ^ 1] for i in range(65536))
    initial_nan = sum(
        ((i >> 10) & 31) == 31 and (i & 1023) != 0 for i in initial_indices
    )
    status = (
        "PASS"
        if len(initial_indices) == 1023
        and initial_nan == 1023
        and not corrected_indices
        and swapped_count > 0
        else "FAIL"
    )
    return {
        "schema": 1,
        "experiment": "fp16_e4m3_oracle_semantics",
        "status": status,
        "classification": "RTX_ORACLE_STATIC_ANALYSIS",
        "counts_as_s6": False,
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "unique_output_codes": len(set(raw)),
        "initial_total_mismatches": len(initial_indices),
        "initial_nan_mismatches": initial_nan,
        "initial_first_mismatch": initial_indices[0] if initial_indices else -1,
        "initial_last_mismatch": initial_indices[-1] if initial_indices else -1,
        "corrected_total_mismatches": len(corrected_indices),
        "pair_swapped_mismatches": swapped_count,
        "nan_canonical_code": 0x7F,
        "packed_lane_order": "low_f16_to_low_e4m3",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.raw.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
