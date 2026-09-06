#!/usr/bin/env python3
"""Trace post-block FP8 and F16 MMA operands/results for one selected CTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.lower_ptx_fp8_mma import FP8_MMA
    from scripts.lower_ptx_f16_mma import F16_MMA
except ModuleNotFoundError:
    from lower_ptx_fp8_mma import FP8_MMA
    from lower_ptx_f16_mma import F16_MMA


LANES = 32
WORDS_PER_RECORD = 10
RECORD_BYTES = WORDS_PER_RECORD * 4
FP8_COUNT = 256
F16_COUNT = 16
PARAM_BYTES = 184
TRACE_PARAM_OFFSET = 184
TRACE_BYTES = (FP8_COUNT + F16_COUNT) * LANES * RECORD_BYTES
PARAM_DECL = re.compile(
    r"(cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0\[)184(\])"
)


def _instrument_family(text: str, pattern: re.Pattern[str], *, family: str,
                       count_base: int, expected: int, target_x: int,
                       target_y: int) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        index = count
        count += 1
        a = [match.group(f"a{i}") for i in range(4)]
        b = [match.group(f"b{i}") for i in range(2)]
        c = [match.group(f"c{i}") for i in range(2)]
        d = [match.group(f"d{i}") for i in range(2)]
        record = count_base + index
        base = record * LANES * RECORD_BYTES
        stem = f"__dlssnr_post_{family}_{index}"
        lines = [
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
        for offset, register in enumerate(a + b + c):
            lines.append(
                f"@%{stem}_selected st.global.b32 [%{stem}_address+{offset * 4}], {register};"
            )
        lines.append(match.group(0))
        lines.extend((
            f"@%{stem}_selected st.global.b32 [%{stem}_address+32], {d[0]};",
            f"@%{stem}_selected st.global.b32 [%{stem}_address+36], {d[1]};",
            "}",
        ))
        return "\n".join(lines)

    output = pattern.sub(replace, text)
    if count != expected:
        raise ValueError(f"{family} MMA count differs: {count} != {expected}")
    return output, count


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, dict]:
    expanded, param_count = PARAM_DECL.subn(r"\g<1>192\g<2>", text)
    fp8_text, fp8_count = _instrument_family(
        expanded, FP8_MMA, family="fp8", count_base=0, expected=FP8_COUNT,
        target_x=target_x, target_y=target_y)
    output, f16_count = _instrument_family(
        fp8_text, F16_MMA, family="f16", count_base=FP8_COUNT,
        expected=F16_COUNT, target_x=target_x, target_y=target_y)
    return output, {
        "parameter_declarations_expanded": param_count,
        "fp8_mma_instrumented": fp8_count,
        "f16_mma_instrumented": f16_count,
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
    output_text, counts = instrument(
        source.decode("utf-8"), args.target_x, args.target_y)
    output = output_text.encode("utf-8")
    expected = {
        "parameter_declarations_expanded": 1,
        "fp8_mma_instrumented": FP8_COUNT,
        "f16_mma_instrumented": F16_COUNT,
    }
    passed = counts == expected
    report = {
        "schema": 1,
        "experiment": "postblock_selected_cta_mma_register_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "POSTBLOCK_FP8_F16_MMA_FIRST_DIVERGENCE_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0],
        "grid": [81, 49, 1],
        "block": [32, 1, 1],
        "lanes": LANES,
        "record_bytes": RECORD_BYTES,
        "trace_param_offset": TRACE_PARAM_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "families": [
            {"name": "fp8", "record_start": 0, "mma_count": FP8_COUNT},
            {"name": "f16", "record_start": FP8_COUNT, "mma_count": F16_COUNT},
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
