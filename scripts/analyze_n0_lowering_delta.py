#!/usr/bin/env python3
"""Verify that compatibility lowering removes only N0 tuple/cache diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_n0_lowering_coverage import CACHE_HINT, STATEMENT, classify  # noqa: E402


def inventory(text: str) -> dict:
    statements = STATEMENT.findall(text)
    counts = Counter(classify(statement) for statement in statements)
    counts["cache_hint_store"] = len(CACHE_HINT.findall(text))
    return {
        "parser_diagnostics": len(statements) + counts["cache_hint_store"],
        "unrecognized_statements": len(statements),
        "category_counts": dict(sorted(counts.items())),
    }


def analyze(source_text: str, lowered_text: str, probe: dict) -> dict:
    source = inventory(source_text)
    lowered = inventory(lowered_text)
    before = Counter(source["category_counts"])
    after = Counter(lowered["category_counts"])
    removed = before - after
    expected_remaining = {
        "e4m3_conversion": 424,
        "f16_mma": 16,
        "fp8_mma": 256,
        "movmatrix": 32,
    }
    expected_removed = {
        "cache_hint_store": 4,
        "tuple_mov_b128": 4,
        "tuple_mov_b64": 4,
    }
    remaining_nonzero = {key: value for key, value in after.items() if value}
    removed_nonzero = {key: value for key, value in removed.items() if value}
    module_loaded = probe.get("module_loaded") is True
    function_resolved = probe.get("function_resolved") is True
    device_name = str(probe.get("device_name", ""))
    status = (
        "PASS"
        if source["parser_diagnostics"] == 740
        and lowered["parser_diagnostics"] == 728
        and removed_nonzero == expected_removed
        and remaining_nonzero == expected_remaining
        and module_loaded
        and "RX 9070 XT" in device_name
        else "FAIL"
    )
    return {
        "schema": 1,
        "experiment": "n0_tuple_cache_lowering_delta",
        "status": status,
        "classification": "REAL_AMD_TRANSLATOR_DIAGNOSTIC_DELTA",
        "counts_as_s6": False,
        "source": source,
        "lowered": lowered,
        "removed_categories": removed_nonzero,
        "remaining_categories": remaining_nonzero,
        "removed_parser_diagnostics": source["parser_diagnostics"] - lowered["parser_diagnostics"],
        "amd_probe": {
            "device_name": device_name,
            "module_loaded": module_loaded,
            "function_resolved": function_resolved,
            "kernel_launched": probe.get("kernel_launched") is True,
        },
        "translator_integrated_numeric_statements": 0,
        "warning": "module load and parser delta do not constitute N0 function resolution or execution",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_parser_log", type=Path)
    parser.add_argument("lowered_parser_log", type=Path)
    parser.add_argument("probe_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.source_parser_log.read_text(encoding="utf-8", errors="replace"),
        args.lowered_parser_log.read_text(encoding="utf-8", errors="replace"),
        json.loads(args.probe_json.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
