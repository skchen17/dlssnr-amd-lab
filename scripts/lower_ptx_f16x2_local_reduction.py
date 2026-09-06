#!/usr/bin/env python3
"""Fuse the 16 known slot-3 four-term FP16 reduction nodes through FP32."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REG = r"%[A-Za-z_][A-Za-z0-9_]*"
ADD = re.compile(
    rf"add\.f16x2\s+(?P<dst>{REG})\s*,\s*(?P<a>{REG})\s*,\s*(?P<b>{REG})\s*;"
)
TARGETS = (
    "%r1835", "%r1845", "%r1855", "%r1861",
    "%r1871", "%r1877", "%r1887", "%r1893",
    "%r2235", "%r2241", "%r2251", "%r2257",
    "%r2267", "%r2273", "%r2283", "%r2289",
)


def fused_block(index: int, destination: str, sources: tuple[str, ...]) -> str:
    prefix = f"__slot3_red_{index}"
    halves = [(f"%{prefix}_h{i}_lo", f"%{prefix}_h{i}_hi") for i in range(4)]
    floats = [(f"%{prefix}_f{i}_lo", f"%{prefix}_f{i}_hi") for i in range(4)]
    pair0 = (f"%{prefix}_pair0_lo", f"%{prefix}_pair0_hi")
    pair1 = (f"%{prefix}_pair1_lo", f"%{prefix}_pair1_hi")
    total = (f"%{prefix}_total_lo", f"%{prefix}_total_hi")
    output = (f"%{prefix}_out_lo", f"%{prefix}_out_hi")
    lines = [
        ".reg .b16 " + ", ".join(item for pair in (*halves, output) for item in pair) + ";",
        ".reg .f32 " + ", ".join(
            item for pair in (*floats, pair0, pair1, total) for item in pair
        ) + ";",
    ]
    for source, (lo, hi), (flo, fhi) in zip(sources, halves, floats):
        lines.extend((
            f"mov.b32 {{{lo}, {hi}}}, {source};",
            f"cvt.f32.f16 {flo}, {lo};",
            f"cvt.f32.f16 {fhi}, {hi};",
        ))
    for lane in range(2):
        lines.extend((
            f"add.rn.f32 {pair0[lane]}, {floats[0][lane]}, {floats[1][lane]};",
            f"add.rn.f32 {pair1[lane]}, {floats[2][lane]}, {floats[3][lane]};",
            f"add.rn.f32 {total[lane]}, {pair0[lane]}, {pair1[lane]};",
            f"cvt.rn.f16.f32 {output[lane]}, {total[lane]};",
        ))
    lines.append(f"mov.b32 {destination}, {{{output[0]}, {output[1]}}};")
    return "{\n" + "\n".join(lines) + "\n}"


def lower(text: str, selected_targets: tuple[str, ...] = TARGETS) -> tuple[str, list[dict[str, object]]]:
    unknown = set(selected_targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown reduction target(s): {', '.join(sorted(unknown))}")
    active_targets = tuple(target for target in TARGETS if target in selected_targets)
    definitions: dict[str, tuple[str, str]] = {}
    definition_counts: dict[str, int] = {}
    for match in ADD.finditer(text):
        destination = match.group("dst")
        definitions[destination] = (match.group("a"), match.group("b"))
        definition_counts[destination] = definition_counts.get(destination, 0) + 1

    resolved: dict[str, tuple[str, ...]] = {}
    for target in active_targets:
        if definition_counts.get(target) != 1:
            raise ValueError(f"expected one definition for {target}")
        pair_a, pair_b = definitions[target]
        if definition_counts.get(pair_a) != 1 or definition_counts.get(pair_b) != 1:
            raise ValueError(f"expected unique pair definitions feeding {target}")
        resolved[target] = (*definitions[pair_a], *definitions[pair_b])

    reports: list[dict[str, object]] = []

    def replace(match: re.Match[str]) -> str:
        destination = match.group("dst")
        if destination not in resolved:
            return match.group(0)
        sources = resolved[destination]
        reports.append({
            "destination": destination,
            "pair_registers": [match.group("a"), match.group("b")],
            "half_inputs": list(sources),
        })
        return fused_block(len(reports) - 1, destination, sources)

    output = ADD.sub(replace, text)
    if tuple(item["destination"] for item in reports) != active_targets:
        raise ValueError("target reductions were not found in the expected order")
    return output, reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--target", action="append", choices=TARGETS)
    args = parser.parse_args()
    source = args.input.read_bytes()
    try:
        selected = tuple(args.target) if args.target else TARGETS
        expected = args.expected if args.expected is not None else len(selected)
        lowered, replacements = lower(source.decode("utf-8"), selected)
        error = None
    except ValueError as exception:
        lowered, replacements, error = source.decode("utf-8"), [], str(exception)
    output = lowered.encode("utf-8")
    passed = len(replacements) == expected and error is None
    report = {
        "schema": 1,
        "experiment": "slot3_f16x2_local_reduction_fp32_fusion",
        "status": "PASS" if passed else "FAIL",
        "classification": "TARGETED_NUMERICAL_SEMANTICS_CANDIDATE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "expected_replacements": expected,
        "replacement_count": len(replacements),
        "replacements": replacements,
        "error": error,
        "semantics": (
            "Each selected add-of-two-adds uses its four packed-half inputs, evaluates "
            "the two pair sums and final sum in RN FP32, then rounds once to packed FP16"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
