#!/usr/bin/env python3
"""Trace global-memory operations back to 64-bit PTX parameter fields.

This is intentionally conservative: it preserves a pointer origin and constant
offset through dynamic address arithmetic, but never invents bounds for the
dynamic terms. The output is static ABI evidence, not a tensor-shape claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


REG = r"%[a-zA-Z]+\d+"
LD_PARAM = re.compile(
    rf"ld\.param\.b64\s+({REG})\s*,\s*\[((?:{REG})|[A-Za-z_$][A-Za-z0-9_$]*)(?:\+(\d+))?\]"
)
COPY = re.compile(
    rf"(?:cvta\.to\.global\.u64|mov\.b64)\s+({REG})\s*,\s*({REG})"
)
ADD = re.compile(rf"add\.(?:s64|u64)\s+({REG})\s*,\s*([^,;]+)\s*,\s*([^;]+)")
MEMORY = re.compile(
    rf"\b(ld|st|atom)([^\s]*)\s+[^;]*?\[\s*({REG})(?:\+(-?(?:0x[0-9a-fA-F]+|\d+)))?\s*\]"
)


@dataclass(frozen=True)
class Expr:
    param_offset: int
    constant: int = 0
    dynamic: bool = False


def integer(token: str) -> int | None:
    token = token.strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def operand_expr(token: str, registers: dict[str, Expr]) -> Expr | int | None:
    token = token.strip()
    if token in registers:
        return registers[token]
    return integer(token)


def memory_width(suffix: str) -> int | None:
    vector_match = re.search(r"\.v(\d+)", suffix)
    vector = int(vector_match.group(1)) if vector_match else 1
    scalar_matches = re.findall(r"\.(?:[busf]|e\w*?)(8|16|32|64|128)(?:\b|\.)", suffix)
    if not scalar_matches:
        scalar_matches = re.findall(r"\.(8|16|32|64|128)(?:\b|\.)", suffix)
    if not scalar_matches:
        return None
    return vector * int(scalar_matches[-1]) // 8


def analyze(path: Path, pointer_offsets: set[int] | None = None) -> dict:
    data = path.read_bytes()
    lines = data.decode("utf-8", errors="replace").splitlines()
    registers: dict[str, Expr] = {}
    accesses: list[dict] = []
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        match = LD_PARAM.search(line)
        if match:
            destination, _base, offset_text = match.groups()
            offset = int(offset_text or 0)
            if pointer_offsets is None or offset in pointer_offsets:
                registers[destination] = Expr(offset)
            else:
                registers.pop(destination, None)
            continue

        match = COPY.search(line)
        if match:
            destination, source = match.groups()
            if source in registers:
                registers[destination] = registers[source]
            else:
                registers.pop(destination, None)
            continue

        match = ADD.search(line)
        if match:
            destination, left_text, right_text = match.groups()
            left = operand_expr(left_text, registers)
            right = operand_expr(right_text, registers)
            expression = None
            if isinstance(left, Expr) and isinstance(right, int):
                expression = Expr(left.param_offset, left.constant + right, left.dynamic)
            elif isinstance(right, Expr) and isinstance(left, int):
                expression = Expr(right.param_offset, right.constant + left, right.dynamic)
            elif isinstance(left, Expr) and right is None:
                expression = Expr(left.param_offset, left.constant, True)
            elif isinstance(right, Expr) and left is None:
                expression = Expr(right.param_offset, right.constant, True)
            if expression:
                registers[destination] = expression
            else:
                registers.pop(destination, None)

        match = MEMORY.search(line)
        if not match:
            continue
        operation, suffix, address_register, immediate_text = match.groups()
        expression = registers.get(address_register)
        if not expression:
            continue
        immediate = int(immediate_text, 0) if immediate_text else 0
        accesses.append(
            {
                "line": line_number,
                "operation": {"ld": "read", "st": "write", "atom": "atomic"}[operation],
                "opcode": operation + suffix,
                "width_bytes": memory_width(suffix),
                "address_register": address_register,
                "param_offset": expression.param_offset,
                "constant_offset": expression.constant + immediate,
                "has_dynamic_offset": expression.dynamic,
                "source": line[:500],
            }
        )

    summaries = []
    offsets = sorted(pointer_offsets or {a["param_offset"] for a in accesses})
    for offset in offsets:
        selected = [a for a in accesses if a["param_offset"] == offset]
        operations = {kind: sum(a["operation"] == kind for a in selected)
                      for kind in ("read", "write", "atomic")}
        constants = [a["constant_offset"] for a in selected]
        summaries.append(
            {
                "param_offset": offset,
                "access_count": len(selected),
                "operations": operations,
                "min_constant_offset": min(constants) if constants else None,
                "max_constant_offset": max(constants) if constants else None,
                "dynamic_access_count": sum(a["has_dynamic_offset"] for a in selected),
                "classification": (
                    "read_only_candidate" if selected and operations["read"] and not operations["write"] and not operations["atomic"]
                    else "write_only_candidate" if selected and operations["write"] and not operations["read"] and not operations["atomic"]
                    else "read_write_candidate" if selected
                    else "no_global_access_traced"
                ),
            }
        )
    return {
        "schema": 1,
        "experiment": "ptx_global_access_origin",
        "classification": "STATIC_ABI_METADATA_ONLY",
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "pointer_param_offsets": offsets,
        "summaries": summaries,
        "accesses": accesses,
        "limitations": [
            "Dynamic index bounds are not inferred.",
            "Control-flow joins are treated conservatively.",
            "Candidate roles require dynamic before/after evidence before becoming tensor semantics.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("--pointer-offset", type=int, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.ptx, set(args.pointer_offset) if args.pointer_offset else None)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
