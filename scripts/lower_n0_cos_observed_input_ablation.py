#!/usr/bin/env python3
"""Correct selected observed cosine inputs for causal N0 diagnostics only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGET = re.compile(r"cos\.approx\.ftz\.f32\s+(?P<dst>%r229)\s*,\s*(?P<src>%r226)\s*;")

# Each row is (angle input bits, additive correction bits, evidence sample).
# These inputs were selected after observing output residuals. This is expressly
# an overfit causal ablation, not a general NVIDIA cosine implementation.
CORRECTIONS = (
    (0x409FA99E, 0x33C00000, 5584),
    (0x4078749E, 0xB3800000, 5600),
    (0x409A4DC8, 0xB3C00000, 17654),
)


def lower(text: str, count: int) -> tuple[str, int, list[dict]]:
    if not 1 <= count <= len(CORRECTIONS):
        raise ValueError(f"correction count must be 1..{len(CORRECTIONS)}")
    selected = CORRECTIONS[:count]
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        index = replacements
        replacements += 1
        prefix = f"__n0_cos_observed_{index}"
        predicates = ", ".join(f"%{prefix}_p{i}" for i in range(count))
        lines = [
            match.group(0),
            "{",
            f".reg .pred {predicates};",
            f".reg .b32 %{prefix}_bits;",
            f"mov.b32 %{prefix}_bits, {match.group('src')};",
        ]
        for correction_index, (input_bits, delta_bits, _) in enumerate(selected):
            lines.extend((
                f"setp.eq.u32 %{prefix}_p{correction_index}, %{prefix}_bits, {input_bits};",
                f"@%{prefix}_p{correction_index} add.rn.f32 {match.group('dst')}, "
                f"{match.group('dst')}, 0f{delta_bits:08X};",
            ))
        lines.append("}")
        return "\n".join(lines)

    lowered = TARGET.sub(replace, text)
    metadata = [
        {
            "input_u32": f"0x{input_bits:08X}",
            "correction_u32": f"0x{delta_bits:08X}",
            "evidence_sample": sample,
        }
        for input_bits, delta_bits, sample in selected
    ]
    return lowered, replacements, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--count", type=int, choices=range(1, len(CORRECTIONS) + 1), required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, replacements, corrections = lower(source.decode("utf-8"), args.count)
    output = output_text.encode("utf-8")
    passed = replacements == 1
    report = {
        "schema": 1,
        "experiment": "ptx_n0_cos_observed_input_ablation",
        "status": "PASS" if passed else "FAIL",
        "classification": "DIAGNOSTIC_OVERFIT_NOT_GENERAL_COS_MODEL",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "cos_sites_corrected": replacements,
        "corrections": corrections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
