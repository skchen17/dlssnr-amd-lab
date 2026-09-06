#!/usr/bin/env python3
"""Trace post-block E4M3 conversions and movmatrix operations for one CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.lower_ptx_e4m3 import CONVERSION
    from scripts.lower_ptx_movmatrix import MOVMATRIX
except ModuleNotFoundError:
    from lower_ptx_e4m3 import CONVERSION
    from lower_ptx_movmatrix import MOVMATRIX


LANES = 32
RECORD_BYTES = 8
E4M3_COUNT = 388
MOV_COUNT = 32
TRACE_PARAM_OFFSET = 184
TRACE_BYTES = (E4M3_COUNT + MOV_COUNT) * LANES * RECORD_BYTES
PARAM_DECL = re.compile(
    r"(cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0\[)184(\])"
)


def _guarded_prefix(stem: str, base: int, target_x: int, target_y: int) -> list[str]:
    return [
        "{",
        f".reg .pred %{stem}_px, %{stem}_py, %{stem}_selected;",
        f".reg .b32 %{stem}_x, %{stem}_y, %{stem}_lane, %{stem}_offset;",
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
    ]


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, dict]:
    expanded, param_count = PARAM_DECL.subn(r"\g<1>192\g<2>", text)
    e4_count = 0

    def e4_replace(match: re.Match[str]) -> str:
        nonlocal e4_count
        index = e4_count
        e4_count += 1
        stem = f"__dlssnr_post_e4_{index}"
        base = index * LANES * RECORD_BYTES
        lines = _guarded_prefix(stem, base, target_x, target_y)
        lines.extend((
            f"@%{stem}_selected st.global.b32 [%{stem}_address], {match.group('src')};",
            match.group(0),
            f"@%{stem}_selected st.global.b16 [%{stem}_address+4], {match.group('dst')};",
            "}",
        ))
        return "\n".join(lines)

    traced = CONVERSION.sub(e4_replace, expanded)
    mov_count = 0

    def mov_replace(match: re.Match[str]) -> str:
        nonlocal mov_count
        index = mov_count
        mov_count += 1
        stem = f"__dlssnr_post_mov_{index}"
        base = (E4M3_COUNT + index) * LANES * RECORD_BYTES
        lines = _guarded_prefix(stem, base, target_x, target_y)
        lines.extend((
            f"@%{stem}_selected st.global.b32 [%{stem}_address], {match.group('src')};",
            match.group(0),
            f"@%{stem}_selected st.global.b32 [%{stem}_address+4], {match.group('dst')};",
            "}",
        ))
        return "\n".join(lines)

    output = MOVMATRIX.sub(mov_replace, traced)
    counts = {
        "parameter_declarations_expanded": param_count,
        "e4m3_conversions_instrumented": e4_count,
        "movmatrix_instrumented": mov_count,
    }
    return output, counts


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
    expected = {
        "parameter_declarations_expanded": 1,
        "e4m3_conversions_instrumented": E4M3_COUNT,
        "movmatrix_instrumented": MOV_COUNT,
    }
    passed = counts == expected
    report = {
        "schema": 1,
        "experiment": "postblock_selected_cta_e4m3_mov_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "POSTBLOCK_PRE_FP8_FIRST_DIVERGENCE_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0],
        "grid": [81, 49, 1], "block": [32, 1, 1],
        "record_bytes": RECORD_BYTES, "trace_param_offset": TRACE_PARAM_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "families": [
            {"name": "e4m3", "record_start": 0, "operation_count": E4M3_COUNT},
            {"name": "movmatrix", "record_start": E4M3_COUNT,
             "operation_count": MOV_COUNT},
        ],
        **counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
