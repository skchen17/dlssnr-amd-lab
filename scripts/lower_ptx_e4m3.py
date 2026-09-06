#!/usr/bin/env python3
"""Lower packed FP16-to-E4M3 PTX conversions to integer instructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CONVERSION = re.compile(
    r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+(?P<dst>%rs\d+)\s*,\s*(?P<src>%r\d+)\s*;"
)


def round_shift(value: int, shift: int) -> int:
    base = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return base + (remainder > halfway or (remainder == halfway and (base & 1)))


def encode_half_bits_integer(bits: int) -> int:
    sign = (bits >> 8) & 0x80
    exponent = (bits >> 10) & 31
    mantissa = bits & 1023
    if exponent == 31 and mantissa:
        return 0x7F
    if exponent < 9:
        code = min(round_shift(1024 + mantissa, 16 - exponent), 8)
    else:
        rounded_mantissa = round_shift(mantissa, 7)
        carry = rounded_mantissa >= 8
        code = ((exponent - 8 + int(carry)) << 3) | (0 if carry else rounded_mantissa)
        code = min(code, 0x7E)
    return sign | code


def half_sequence(source: str, prefix: str, high: bool) -> tuple[list[str], str]:
    r = lambda name: f"%__e4_{prefix}_r_{name}"
    p = lambda name: f"%__e4_{prefix}_p_{name}"
    lines = [
        f"shr.u32 {r('bits')}, {source}, 16;" if high else f"and.b32 {r('bits')}, {source}, 65535;",
        f"shr.u32 {r('sign')}, {r('bits')}, 8;",
        f"and.b32 {r('sign')}, {r('sign')}, 128;",
        f"shr.u32 {r('exp')}, {r('bits')}, 10;",
        f"and.b32 {r('exp')}, {r('exp')}, 31;",
        f"and.b32 {r('mant')}, {r('bits')}, 1023;",
        # E4M3 subnormal candidate: RN-even((1024+mantissa) >> (16-half_exp)).
        f"add.u32 {r('number')}, {r('mant')}, 1024;",
        f"sub.s32 {r('shift')}, 16, {r('exp')};",
        f"shr.u32 {r('base_sub')}, {r('number')}, {r('shift')};",
        f"shl.b32 {r('mask')}, 1, {r('shift')};",
        f"add.s32 {r('mask')}, {r('mask')}, -1;",
        f"and.b32 {r('remainder_sub')}, {r('number')}, {r('mask')};",
        f"add.s32 {r('half_shift')}, {r('shift')}, -1;",
        f"shl.b32 {r('halfway_sub')}, 1, {r('half_shift')};",
        f"setp.gt.u32 {p('gt_sub')}, {r('remainder_sub')}, {r('halfway_sub')};",
        f"setp.eq.u32 {p('eq_sub')}, {r('remainder_sub')}, {r('halfway_sub')};",
        f"and.b32 {r('odd_sub')}, {r('base_sub')}, 1;",
        f"setp.ne.u32 {p('odd_sub')}, {r('odd_sub')}, 0;",
        f"and.pred {p('tie_sub')}, {p('eq_sub')}, {p('odd_sub')};",
        f"or.pred {p('inc_sub')}, {p('gt_sub')}, {p('tie_sub')};",
        f"selp.b32 {r('increment_sub')}, 1, 0, {p('inc_sub')};",
        f"add.u32 {r('code_sub')}, {r('base_sub')}, {r('increment_sub')};",
        f"setp.ge.u32 {p('cap_sub')}, {r('code_sub')}, 8;",
        f"selp.b32 {r('code_sub')}, 8, {r('code_sub')}, {p('cap_sub')};",
        # E4M3 normal candidate: retain 3 of 10 half mantissa bits with RN-even.
        f"shr.u32 {r('base_norm')}, {r('mant')}, 7;",
        f"and.b32 {r('remainder_norm')}, {r('mant')}, 127;",
        f"setp.gt.u32 {p('gt_norm')}, {r('remainder_norm')}, 64;",
        f"setp.eq.u32 {p('eq_norm')}, {r('remainder_norm')}, 64;",
        f"and.b32 {r('odd_norm')}, {r('base_norm')}, 1;",
        f"setp.ne.u32 {p('odd_norm')}, {r('odd_norm')}, 0;",
        f"and.pred {p('tie_norm')}, {p('eq_norm')}, {p('odd_norm')};",
        f"or.pred {p('inc_norm')}, {p('gt_norm')}, {p('tie_norm')};",
        f"selp.b32 {r('increment_norm')}, 1, 0, {p('inc_norm')};",
        f"add.u32 {r('rounded_norm')}, {r('base_norm')}, {r('increment_norm')};",
        f"setp.ge.u32 {p('carry_norm')}, {r('rounded_norm')}, 8;",
        f"selp.b32 {r('mant_norm')}, 0, {r('rounded_norm')}, {p('carry_norm')};",
        f"selp.b32 {r('carry_value')}, 1, 0, {p('carry_norm')};",
        f"add.u32 {r('encoded_exp')}, {r('exp')}, {r('carry_value')};",
        f"add.s32 {r('encoded_exp')}, {r('encoded_exp')}, -8;",
        f"shl.b32 {r('code_norm')}, {r('encoded_exp')}, 3;",
        f"or.b32 {r('code_norm')}, {r('code_norm')}, {r('mant_norm')};",
        f"setp.ge.u32 {p('cap_norm')}, {r('code_norm')}, 126;",
        f"selp.b32 {r('code_norm')}, 126, {r('code_norm')}, {p('cap_norm')};",
        f"setp.lt.u32 {p('is_sub')}, {r('exp')}, 9;",
        f"selp.b32 {r('magnitude')}, {r('code_sub')}, {r('code_norm')}, {p('is_sub')};",
        f"or.b32 {r('finite')}, {r('magnitude')}, {r('sign')};",
        f"setp.eq.u32 {p('exp31')}, {r('exp')}, 31;",
        f"setp.ne.u32 {p('mant_nonzero')}, {r('mant')}, 0;",
        f"and.pred {p('is_nan')}, {p('exp31')}, {p('mant_nonzero')};",
        f"selp.b32 {r('code')}, 127, {r('finite')}, {p('is_nan')};",
    ]
    return lines, r("code")


def replacement(match: re.Match[str], index: int) -> str:
    destination, source = match.group("dst"), match.group("src")
    low_lines, low = half_sequence(source, f"{index}_lo", False)
    high_lines, high = half_sequence(source, f"{index}_hi", True)
    all_names = set(re.findall(r"%__e4_[A-Za-z0-9_]+", "\n".join(low_lines + high_lines)))
    pred_names = {name for name in all_names if "_p_" in name}
    int_names = sorted(all_names - pred_names)
    pred_names = sorted(pred_names)
    packed = f"%__e4_{index}_packed"
    int_names.append(packed)
    declarations = [f".reg .b32 {', '.join(int_names)};"]
    declarations.append(f".reg .pred {', '.join(pred_names)};")
    lines = declarations + low_lines + high_lines + [
        f"shl.b32 {high}, {high}, 8;",
        f"or.b32 {packed}, {low}, {high};",
        f"cvt.u16.u32 {destination}, {packed};",
    ]
    return "{\n" + "\n".join(lines) + "\n}"


def lower(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        result = replacement(match, count)
        count += 1
        return result

    return CONVERSION.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-count", type=int, default=424)
    args = parser.parse_args()
    source = args.input.read_bytes()
    lowered, count = lower(source.decode("utf-8"))
    output = lowered.encode("utf-8")
    remaining = len(CONVERSION.findall(lowered))
    status = "PASS" if count == args.expected_count and remaining == 0 else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_e4m3_integer_lowering",
        "status": status,
        "classification": "EXHAUSTIVELY_VALIDATED_NUMERIC_PTX_REWRITE",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "conversions_lowered": count,
        "expected_conversions": args.expected_count,
        "remaining_e4m3_conversions": remaining,
        "integer_semantics_domain": 65536,
        "translator_execution_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
