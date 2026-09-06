#!/usr/bin/env python3
"""Replace texture samples with zero under an explicit caller-provided approximation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TEXTURE_SAMPLE = re.compile(
    r"(?m)^(?P<i>[ \t]*)tex\.2d\.v4\.f32\.f32\s+"
    r"\{\s*(?P<d0>%r[0-9]+),\s*(?P<d1>%r[0-9]+),\s*"
    r"(?P<d2>%r[0-9]+),\s*(?P<d3>%r[0-9]+)\s*\},\s*"
    r"\[(?P<texture>%rd[0-9]+),\s*\{[^}]+\}\];[ \t]*$"
)


def lower(text: str) -> tuple[str, dict]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        indent = match.group("i")
        return "\n".join(
            indent + f"mov.b32 {match.group(f'd{index}')}, 0f00000000;"
            for index in range(4))

    lowered = TEXTURE_SAMPLE.sub(replace, text)
    return lowered, {
        "null_texture_samples_lowered": count,
        "remaining_texture_samples": len(TEXTURE_SAMPLE.findall(lowered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    lowered, counts = lower(source_bytes.decode("utf-8"))
    lowered_bytes = lowered.encode("utf-8")
    status = "PASS" if counts == {
        "null_texture_samples_lowered": args.expected_count,
        "remaining_texture_samples": 0,
    } else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_exact_state_null_texture_lowering",
        "status": status,
        "classification": "EXPLICIT_ZERO_TEXTURE_APPROXIMATION",
        "counts_as_s7": False,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(lowered_bytes).hexdigest(),
        "expected_count": args.expected_count,
        **counts,
        "semantics": (
            "Every matched texture sample is replaced with a zero vector. This is "
            "valid only when the caller has independently established null-resource "
            "semantics; the transform itself does not inspect descriptor state."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(lowered_bytes)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
