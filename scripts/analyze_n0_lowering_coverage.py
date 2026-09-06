#!/usr/bin/env python3
"""Classify N0 parser failures against validated AMD semantic primitives."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STATEMENT = re.compile(r'Unrecognized statement "(.*?)"', re.DOTALL)
CACHE_HINT = re.compile(
    r"Not yet implemented: st instruction with cache policy/eviction priority/cache hints"
)


def classify(statement: str) -> str:
    if "cvt.rn.satfinite.e4m3x2.f16x2" in statement:
        return "e4m3_conversion"
    if "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16" in statement:
        return "fp8_mma"
    if "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16" in statement:
        return "f16_mma"
    if "movmatrix.sync.trans.aligned.m8n8.b16" in statement:
        return "movmatrix"
    if "mov.b64" in statement:
        return "tuple_mov_b64"
    if "mov.b128" in statement:
        return "tuple_mov_b128"
    return "other"


def analyze(text: str) -> dict:
    statements = STATEMENT.findall(text)
    counts = Counter(classify(statement) for statement in statements)
    counts["cache_hint_store"] = len(CACHE_HINT.findall(text))
    semantic_exact = counts["e4m3_conversion"] + counts["fp8_mma"] + counts["movmatrix"]
    semantic_scoped = counts["f16_mma"]
    mechanics = counts["tuple_mov_b64"] + counts["tuple_mov_b128"] + counts["cache_hint_store"]
    unresolved = mechanics + counts["other"]
    diagnostics = len(statements) + counts["cache_hint_store"]
    status = (
        "PASS"
        if diagnostics == 740
        and len(statements) == 736
        and mechanics == 12
        and not counts["other"]
        else "FAIL"
    )
    return {
        "schema": 1,
        "experiment": "n0_lowering_semantic_coverage",
        "status": status,
        "classification": "STATIC_TRANSLATION_INVENTORY",
        "counts_as_s6": False,
        "parser_diagnostics": diagnostics,
        "unrecognized_statements": len(statements),
        "category_counts": dict(sorted(counts.items())),
        "semantic_exact_statements": semantic_exact,
        "semantic_scoped_statements": semantic_scoped,
        "semantic_oracle_covered_statements": semantic_exact + semantic_scoped,
        "mechanical_lowering_statements": mechanics,
        "unresolved_statements": unresolved,
        "translator_integrated_statements": 0,
        "remaining_categories": ["tuple_mov_b64", "tuple_mov_b128", "cache_hint_store"],
        "warning": "semantic oracle coverage is not translator integration or fused N0 execution",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parser_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.parser_log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
