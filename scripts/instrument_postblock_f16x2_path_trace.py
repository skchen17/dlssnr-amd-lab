#!/usr/bin/env python3
"""Trace the three packed-f16 operations that directly produce E4M3 input r944."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.instrument_postblock_f16x2_trace import (
        LANES, PARAM_DECL, RECORD_BYTES, find_operations, trace_block,
    )
except ModuleNotFoundError:
    from instrument_postblock_f16x2_trace import (
        LANES, PARAM_DECL, RECORD_BYTES, find_operations, trace_block,
    )


SELECTED = (0, 32, 64)
EXPECTED = (
    ("mul", "%r816", ("%r204", "%r205")),
    ("mul", "%r817", ("%r720", "%r721")),
    ("add", "%r944", ("%r816", "%r817")),
)
TRACE_PARAM_OFFSET = 184
TRACE_BYTES = len(SELECTED) * LANES * RECORD_BYTES


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, dict]:
    expanded, param_count = PARAM_DECL.subn(r"\g<1>192\g<2>", text)
    operations = find_operations(expanded)
    selected = [(source_index, operations[source_index]) for source_index in SELECTED]
    observed = tuple((op.kind, op.dst, op.sources) for _, op in selected)
    chunks: list[str] = []
    cursor = 0
    descriptions = []
    for trace_index, (source_index, operation) in enumerate(selected):
        chunks.append(expanded[cursor:operation.start])
        chunks.append(trace_block(operation, trace_index, target_x, target_y))
        cursor = operation.end
        descriptions.append({
            "trace_index": trace_index, "source_operation_index": source_index,
            "kind": operation.kind, "dst": operation.dst,
            "sources": list(operation.sources),
        })
    chunks.append(expanded[cursor:])
    return "".join(chunks), {
        "parameter_declarations_expanded": param_count,
        "source_operation_count": len(operations),
        "selected_operations": descriptions,
        "selection_matches_expected": observed == EXPECTED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--target-x", type=int, required=True)
    parser.add_argument("--target-y", type=int, required=True)
    args = parser.parse_args()
    if not (0 <= args.target_x < 81 and 0 <= args.target_y < 49):
        parser.error("target CTA must be inside the 81x49 post-block grid")
    source = args.input.read_bytes()
    output_text, details = instrument(source.decode("utf-8"), args.target_x, args.target_y)
    output = output_text.encode("utf-8")
    passed = (details["parameter_declarations_expanded"] == 1 and
              details["source_operation_count"] == 1522 and
              details["selection_matches_expected"])
    report = {
        "schema": 1,
        "experiment": "postblock_selected_cta_f16x2_r944_path_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "LOW_PERTURBATION_POSTBLOCK_R944_CAUSAL_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0],
        "grid": [81, 49, 1], "block": [32, 1, 1],
        "record_bytes": RECORD_BYTES, "trace_param_offset": TRACE_PARAM_OFFSET,
        "trace_bytes": TRACE_BYTES,
        **details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
