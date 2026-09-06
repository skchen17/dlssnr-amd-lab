#!/usr/bin/env python3
"""Trace every packed-f16 arithmetic operation for one post-block CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.lower_ptx_f16x2_arithmetic import ABS, BINARY, FMA
except ModuleNotFoundError:
    from lower_ptx_f16x2_arithmetic import ABS, BINARY, FMA


LANES = 32
RECORD_BYTES = 20
EXPECTED_BINARY = 1074
EXPECTED_FMA = 320
EXPECTED_ABS = 128
OPERATION_COUNT = EXPECTED_BINARY + EXPECTED_FMA + EXPECTED_ABS
TRACE_PARAM_OFFSET = 184
TRACE_BYTES = OPERATION_COUNT * LANES * RECORD_BYTES
PARAM_DECL = re.compile(
    r"(cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0\[)184(\])"
)
KIND_CODES = {"mul": 1, "add": 2, "min": 3, "max": 4, "fma": 5, "abs": 6}


@dataclass(frozen=True)
class Operation:
    start: int
    end: int
    text: str
    kind: str
    dst: str
    sources: tuple[str, ...]


def find_operations(text: str) -> list[Operation]:
    operations: list[Operation] = []
    for match in BINARY.finditer(text):
        operations.append(Operation(match.start(), match.end(), match.group(0),
                                    match.group("op"), match.group("dst"),
                                    (match.group("a"), match.group("b"))))
    for match in FMA.finditer(text):
        operations.append(Operation(match.start(), match.end(), match.group(0), "fma",
                                    match.group("dst"),
                                    (match.group("a"), match.group("b"), match.group("c"))))
    for match in ABS.finditer(text):
        operations.append(Operation(match.start(), match.end(), match.group(0), "abs",
                                    match.group("dst"), (match.group("src"),)))
    operations.sort(key=lambda operation: operation.start)
    for left, right in zip(operations, operations[1:]):
        if left.end > right.start:
            raise ValueError("overlapping packed-f16 operation matches")
    return operations


def trace_block(operation: Operation, index: int, target_x: int, target_y: int) -> str:
    stem = f"__dlssnr_post_f16x2_{index}"
    base = index * LANES * RECORD_BYTES
    sources = list(operation.sources) + [None] * (3 - len(operation.sources))
    lines = [
        "{",
        f".reg .pred %{stem}_px, %{stem}_py, %{stem}_selected;",
        f".reg .b32 %{stem}_x, %{stem}_y, %{stem}_lane, %{stem}_offset, "
        f"%{stem}_zero, %{stem}_kind;",
        f".reg .b64 %{stem}_base, %{stem}_wide, %{stem}_address;",
        f"mov.u32 %{stem}_x, %ctaid.x;",
        f"mov.u32 %{stem}_y, %ctaid.y;",
        f"setp.eq.u32 %{stem}_px, %{stem}_x, {target_x};",
        f"setp.eq.u32 %{stem}_py, %{stem}_y, {target_y};",
        f"and.pred %{stem}_selected, %{stem}_px, %{stem}_py;",
        f"mov.u32 %{stem}_lane, %laneid;",
        f"mad.lo.u32 %{stem}_offset, %{stem}_lane, {RECORD_BYTES}, {base};",
        f"ld.param.b64 %{stem}_base, [%rd11+{TRACE_PARAM_OFFSET}];",
        f"cvt.u64.u32 %{stem}_wide, %{stem}_offset;",
        f"add.s64 %{stem}_address, %{stem}_base, %{stem}_wide;",
        f"mov.u32 %{stem}_zero, 0;",
        f"mov.u32 %{stem}_kind, {KIND_CODES[operation.kind]};",
    ]
    for source_index, source in enumerate(sources):
        value = source if source is not None else f"%{stem}_zero"
        lines.append(
            f"@%{stem}_selected st.global.b32 [%{stem}_address+{source_index * 4}], {value};"
        )
    lines.extend((
        operation.text,
        f"@%{stem}_selected st.global.b32 [%{stem}_address+12], {operation.dst};",
        f"@%{stem}_selected st.global.b32 [%{stem}_address+16], %{stem}_kind;",
        "}",
    ))
    return "\n".join(lines)


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, dict]:
    expanded, param_count = PARAM_DECL.subn(r"\g<1>192\g<2>", text)
    operations = find_operations(expanded)
    chunks: list[str] = []
    cursor = 0
    counts = {"binary": 0, "fma": 0, "abs": 0}
    opcode_counts = {name: 0 for name in KIND_CODES}
    for index, operation in enumerate(operations):
        chunks.append(expanded[cursor:operation.start])
        chunks.append(trace_block(operation, index, target_x, target_y))
        cursor = operation.end
        counts["fma" if operation.kind == "fma" else
               "abs" if operation.kind == "abs" else "binary"] += 1
        opcode_counts[operation.kind] += 1
    chunks.append(expanded[cursor:])
    return "".join(chunks), {
        "parameter_declarations_expanded": param_count,
        "operation_count": len(operations),
        "counts": counts,
        "opcode_counts": opcode_counts,
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
    output_text, counts = instrument(source.decode("utf-8"), args.target_x, args.target_y)
    output = output_text.encode("utf-8")
    expected_counts = {"binary": EXPECTED_BINARY, "fma": EXPECTED_FMA,
                       "abs": EXPECTED_ABS}
    passed = (counts["parameter_declarations_expanded"] == 1 and
              counts["operation_count"] == OPERATION_COUNT and
              counts["counts"] == expected_counts)
    report = {
        "schema": 1,
        "experiment": "postblock_selected_cta_f16x2_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "POSTBLOCK_PRE_E4M3_FIRST_DIVERGENCE_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0],
        "grid": [81, 49, 1],
        "block": [32, 1, 1],
        "record_bytes": RECORD_BYTES,
        "trace_param_offset": TRACE_PARAM_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "kind_codes": KIND_CODES,
        **counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
