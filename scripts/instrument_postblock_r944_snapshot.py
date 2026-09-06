#!/usr/bin/env python3
"""Snapshot the complete r816/r817/r944 chain only at E4M3 conversion 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts.lower_ptx_e4m3 import CONVERSION
except ModuleNotFoundError:
    from lower_ptx_e4m3 import CONVERSION


LANES = 32
RECORD_BYTES = 32
TRACE_BYTES = LANES * RECORD_BYTES
TRACE_PARAM_OFFSET = 184
TARGET_CONVERSION = 4
PARAM_DECL = re.compile(
    r"(cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0\[)184(\])"
)
EXPECTED_SRC = "%r944"
EXPECTED_DST = "%rs53"
WORDS = ("r204", "r205", "r720", "r721", "r816", "r817", "r944")


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, dict]:
    expanded, param_count = PARAM_DECL.subn(r"\g<1>192\g<2>", text)
    conversion_count = 0
    selected_match = False

    def replace(match: re.Match[str]) -> str:
        nonlocal conversion_count, selected_match
        index = conversion_count
        conversion_count += 1
        if index != TARGET_CONVERSION:
            return match.group(0)
        selected_match = match.group("src") == EXPECTED_SRC and match.group("dst") == EXPECTED_DST
        stem = "__dlssnr_post_r944_snapshot"
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
            f"mad.lo.u32 %{stem}_offset, %{stem}_lane, {RECORD_BYTES}, 0;",
            f"ld.param.b64 %{stem}_base, [%rd11+{TRACE_PARAM_OFFSET}];",
            f"cvt.u64.u32 %{stem}_wide, %{stem}_offset;",
            f"add.s64 %{stem}_address, %{stem}_base, %{stem}_wide;",
        ]
        for word_index, register in enumerate(WORDS):
            lines.append(
                f"@%{stem}_selected st.global.b32 [%{stem}_address+{word_index * 4}], %{register};"
            )
        lines.extend((
            match.group(0),
            f"@%{stem}_selected st.global.b16 [%{stem}_address+28], {match.group('dst')};",
            "}",
        ))
        return "\n".join(lines)

    output = CONVERSION.sub(replace, expanded)
    return output, {
        "parameter_declarations_expanded": param_count,
        "conversion_count": conversion_count,
        "selected_conversion": TARGET_CONVERSION,
        "selected_registers_match_expected": selected_match,
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
    passed = (details == {
        "parameter_declarations_expanded": 1,
        "conversion_count": 388,
        "selected_conversion": TARGET_CONVERSION,
        "selected_registers_match_expected": True,
    })
    report = {
        "schema": 1,
        "experiment": "postblock_selected_cta_r944_pre_e4_snapshot_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "NON_PRODUCER_SITE_R944_CAUSAL_SNAPSHOT",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "target_cta": [args.target_x, args.target_y, 0],
        "grid": [81, 49, 1], "block": [32, 1, 1],
        "record_bytes": RECORD_BYTES, "trace_param_offset": TRACE_PARAM_OFFSET,
        "trace_bytes": TRACE_BYTES, "words": list(WORDS),
        "e4m3_output": EXPECTED_DST,
        **details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
