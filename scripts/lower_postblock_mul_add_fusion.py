#!/usr/bin/env python3
"""Fuse selected post-block second multiplies into packed-half adds via f32 FMA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.lower_ptx_f16x2_arithmetic import BINARY
except ModuleNotFoundError:
    from lower_ptx_f16x2_arithmetic import BINARY


def fused_block(index: int, dst: str, rounded: str, mul_a: str, mul_b: str) -> str:
    stem = f"__dlssnr_post_fuse_{index}"
    lines = [
        "{",
        f".reg .b16 %{stem}_rlo, %{stem}_rhi, %{stem}_alo, %{stem}_ahi, "
        f"%{stem}_blo, %{stem}_bhi, %{stem}_olo, %{stem}_ohi;",
        f".reg .f32 %{stem}_frlo, %{stem}_frhi, %{stem}_falo, %{stem}_fahi, "
        f"%{stem}_fblo, %{stem}_fbhi, %{stem}_folo, %{stem}_fohi;",
        f"mov.b32 {{%{stem}_rlo, %{stem}_rhi}}, {rounded};",
        f"mov.b32 {{%{stem}_alo, %{stem}_ahi}}, {mul_a};",
        f"mov.b32 {{%{stem}_blo, %{stem}_bhi}}, {mul_b};",
        f"cvt.f32.f16 %{stem}_frlo, %{stem}_rlo;",
        f"cvt.f32.f16 %{stem}_frhi, %{stem}_rhi;",
        f"cvt.f32.f16 %{stem}_falo, %{stem}_alo;",
        f"cvt.f32.f16 %{stem}_fahi, %{stem}_ahi;",
        f"cvt.f32.f16 %{stem}_fblo, %{stem}_blo;",
        f"cvt.f32.f16 %{stem}_fbhi, %{stem}_bhi;",
        f"fma.rn.f32 %{stem}_folo, %{stem}_falo, %{stem}_fblo, %{stem}_frlo;",
        f"fma.rn.f32 %{stem}_fohi, %{stem}_fahi, %{stem}_fbhi, %{stem}_frhi;",
        f"cvt.rn.f16.f32 %{stem}_olo, %{stem}_folo;",
        f"cvt.rn.f16.f32 %{stem}_ohi, %{stem}_fohi;",
        f"mov.b32 {dst}, {{%{stem}_olo, %{stem}_ohi}};",
        "}",
    ]
    return "\n".join(lines)


def lower(text: str, start: int = 0, count: int = 64) -> tuple[str, dict]:
    operations = list(BINARY.finditer(text))
    definitions = {match.group("dst"): match for match in operations}
    candidates = []
    for match in operations:
        if match.group("op") != "add":
            continue
        second = definitions.get(match.group("b"))
        if second is not None and second.group("op") == "mul":
            candidates.append((match, second))
    selected = candidates[start:start + count]
    replacements = {
        match.start(): (match, fused_block(index, match.group("dst"), match.group("a"),
                                            second.group("a"), second.group("b")))
        for index, (match, second) in enumerate(selected, start=start)
    }
    chunks = []
    cursor = 0
    for match in operations:
        replacement = replacements.get(match.start())
        if replacement is None:
            continue
        chunks.append(text[cursor:match.start()]); chunks.append(replacement[1]); cursor = match.end()
    chunks.append(text[cursor:])
    return "".join(chunks), {
        "candidate_count": len(candidates), "selected_start": start,
        "selected_count": len(selected),
        "selected_destinations": [match.group("dst") for match, _ in selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path); parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--expected-candidates", type=int, default=64)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, details = lower(source.decode("utf-8"), args.start, args.count)
    output = output_text.encode("utf-8")
    passed = (details["candidate_count"] == args.expected_candidates and
              details["selected_count"] == args.count)
    report = {
        "schema": 1, "experiment": "postblock_second_mul_into_add_fusion",
        "status": "PASS" if passed else "FAIL",
        "classification": "RTX_OBSERVED_PACKED_F16_CONTRACTION_CANDIDATE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "semantics": "round first add operand to f16; fuse second product in f32; round sum to f16",
        **details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
