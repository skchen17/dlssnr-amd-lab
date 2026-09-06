#!/usr/bin/env python3
"""Lower packed f16x2 arithmetic to scalar RN f32 operations and f16 packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REG = r"%?[A-Za-z_][A-Za-z0-9_]*"
BINARY = re.compile(
    rf"(?P<op>mul|add|min|max)\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P<b>{REG})\s*;"
)
FMA = re.compile(
    rf"fma\.rn\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P<b>{REG})\s*,\s*(?P<c>{REG})\s*;"
)
ABS = re.compile(rf"abs\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<src>{REG})\s*;")


def scalar_block(index: str, dst: str, sources: list[str], operation: str) -> str:
    prefix = f"__f16x2_{index}"
    half_names = [[f"%{prefix}_{name}_lo", f"%{prefix}_{name}_hi"]
                  for name in [*(f"s{i}" for i in range(len(sources))), "out"]]
    float_names = [[f"%{prefix}_f_{name}_lo", f"%{prefix}_f_{name}_hi"]
                   for name in [*(f"s{i}" for i in range(len(sources))), "out"]]
    lines = [
        ".reg .b16 " + ", ".join(x for pair in half_names for x in pair) + ";",
        ".reg .f32 " + ", ".join(x for pair in float_names for x in pair) + ";",
    ]
    for source_index, source in enumerate(sources):
        lo, hi = half_names[source_index]
        flo, fhi = float_names[source_index]
        lines += [f"mov.b32 {{{lo}, {hi}}}, {source};",
                  f"cvt.f32.f16 {flo}, {lo};", f"cvt.f32.f16 {fhi}, {hi};"]
    out_half = half_names[-1]
    out_float = float_names[-1]
    if operation == "fma":
        for lane in range(2):
            lines.append(f"fma.rn.f32 {out_float[lane]}, {float_names[0][lane]}, "
                         f"{float_names[1][lane]}, {float_names[2][lane]};")
    else:
        suffix = ".rn.f32" if operation in ("mul", "add") else ".f32"
        for lane in range(2):
            lines.append(f"{operation}{suffix} {out_float[lane]}, {float_names[0][lane]}, "
                         f"{float_names[1][lane]};")
    for lane in range(2):
        lines.append(f"cvt.rn.f16.f32 {out_half[lane]}, {out_float[lane]};")
    lines.append(f"mov.b32 {dst}, {{{out_half[0]}, {out_half[1]}}};")
    return "{\n" + "\n".join(lines) + "\n}"


def lower(text: str) -> tuple[str, dict[str, int]]:
    counts = {"binary": 0, "fma": 0, "abs": 0}

    def binary(match: re.Match[str]) -> str:
        index = f"binary_{counts['binary']}"
        counts["binary"] += 1
        return scalar_block(index, match.group("dst"), [match.group("a"), match.group("b")],
                            match.group("op"))

    def fma(match: re.Match[str]) -> str:
        index = f"fma_{counts['fma']}"
        counts["fma"] += 1
        return scalar_block(index, match.group("dst"),
                            [match.group("a"), match.group("b"), match.group("c")], "fma")

    def absolute(match: re.Match[str]) -> str:
        counts["abs"] += 1
        return f"and.b32 {match.group('dst')}, {match.group('src')}, 2147450879;"

    output = BINARY.sub(binary, text)
    output = FMA.sub(fma, output)
    output = ABS.sub(absolute, output)
    return output, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-binary", type=int, required=True)
    parser.add_argument("--expected-fma", type=int, required=True)
    parser.add_argument("--expected-abs", type=int, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    lowered, counts = lower(source.decode("utf-8"))
    output = lowered.encode("utf-8")
    remaining = len(BINARY.findall(lowered)) + len(FMA.findall(lowered)) + len(ABS.findall(lowered))
    passed = (counts == {"binary": args.expected_binary, "fma": args.expected_fma,
                         "abs": args.expected_abs} and remaining == 0)
    report = {
        "schema": 1,
        "experiment": "ptx_f16x2_scalar_rn_lowering",
        "status": "PASS" if passed else "FAIL",
        "classification": "NUMERICAL_STABILITY_PTX_REWRITE",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "counts": counts,
        "expected": {"binary": args.expected_binary, "fma": args.expected_fma,
                     "abs": args.expected_abs},
        "remaining": remaining,
        "semantics": "Each packed lane is evaluated with RN f32 arithmetic then rounded once to f16",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
