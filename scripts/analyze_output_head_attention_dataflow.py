#!/usr/bin/env python3
"""Recover Q/K/V producer groups from the instrumented output-head PTX."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REG = re.compile(r"%r\d+")
MMA = re.compile(
    r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*"
    r"\{(?P<dst>[^}]+)\}", re.MULTILINE
)
CONVERSION = re.compile(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+%rs\d+,\s*(%r\d+)")
SIMPLE_DEF = re.compile(
    r"(?:^|\n)\s*\{?(?:mul|add|max|min|abs|fma)\.[^\s]+\s+(%r\d+)\s*,([^;]+);"
    r"|(?:^|\n)\s*(?:shfl\.[^\s]+|mov\.b32|cvt\.[^\s]+)\s+(%r\d+)\s*,([^;]+);"
)


def analyze(ptx_path: Path) -> dict:
    text = ptx_path.read_text(encoding="utf-8")
    mma_producers: dict[str, tuple[int, int]] = {}
    for mma_index, match in enumerate(MMA.finditer(text)):
        for word, register in enumerate(REG.findall(match.group("dst"))):
            mma_producers[register] = (mma_index, word)

    definitions: dict[str, list[str]] = {}
    for match in SIMPLE_DEF.finditer(text):
        destination = match.group(1) or match.group(3)
        sources = match.group(2) or match.group(4)
        definitions[destination] = REG.findall(sources)

    def ancestors(register: str, seen: set[str] | None = None) -> set[tuple[int, int]]:
        if register in mma_producers:
            return {mma_producers[register]}
        seen = set() if seen is None else seen
        if register in seen:
            return set()
        seen.add(register)
        result: set[tuple[int, int]] = set()
        for source in definitions.get(register, []):
            result.update(ancestors(source, seen.copy()))
        return result

    conversions = []
    for operation, match in enumerate(CONVERSION.finditer(text)):
        if operation < 164 or operation > 291:
            continue
        source = match.group(1)
        producer_words = sorted(ancestors(source))
        conversions.append({
            "operation": operation,
            "source_register": source,
            "mma_producer_words": [list(item) for item in producer_words],
            "mma_producers": sorted({item[0] for item in producer_words}),
        })

    ranges = []
    for start in range(164, 292, 8):
        group = [entry for entry in conversions if start <= entry["operation"] < start + 8]
        ranges.append({
            "operations": [start, start + 7],
            "mma_producers": sorted({mma for entry in group for mma in entry["mma_producers"]}),
        })
    passed = len(mma_producers) >= 256 * 2 and len(conversions) == 128
    return {
        "schema": 1,
        "experiment": "output_head_attention_static_dataflow",
        "status": "PASS" if passed else "FAIL",
        "fp8_mma_output_registers": len(mma_producers),
        "conversion_range": [164, 291],
        "conversion_groups": ranges,
        "conversions": conversions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.ptx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
