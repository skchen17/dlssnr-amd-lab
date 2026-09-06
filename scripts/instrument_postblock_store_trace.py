#!/usr/bin/env python3
"""Trace both post-block RGB(A) vectors immediately before surface stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GRID_X = 81
GRID_Y = 49
LANES = 32
RECORD_BYTES = 32
THREADS = GRID_X * GRID_Y * LANES
SITE_BYTES = THREADS * RECORD_BYTES
TRACE_BYTES = 2 * SITE_BYTES
PARAM_BYTES = 184
TRACE_PARAM_OFFSET = 184

PARAM_DECL = re.compile(
    r"(cc_tinlayout_fused_post_block_swin_1h_32_fp8_param_0\[)184(\])"
)
SURFACE_STORE = re.compile(
    r"(?m)^(?P<i>[ \t]*)(?P<store>sust\.p\.2d\.v4\.b32\.zero\s+"
    r"\[(?P<surface>%rd[0-9]+),\s*\{(?P<x>%r[0-9]+),(?P<y>%r[0-9]+)\}\],\s*"
    r"\{(?P<v0>%r[0-9]+),(?P<v1>%r[0-9]+),(?P<v2>%r[0-9]+),(?P<v3>%r[0-9]+)\};)[ \t]*$"
)


def trace_block(match: re.Match[str], site: int) -> str:
    indent = match.group("i")
    stem = f"__dlssnr_post_trace_{site}"
    site_offset = site * SITE_BYTES
    lines = [
        "{",
        f".reg .b32 %{stem}_cx, %{stem}_cy, %{stem}_lane, %{stem}_cta, "
        f"%{stem}_thread, %{stem}_byte;",
        f".reg .b64 %{stem}_base, %{stem}_wide, %{stem}_address;",
        f"mov.u32 %{stem}_cx, %ctaid.x;",
        f"mov.u32 %{stem}_cy, %ctaid.y;",
        f"mov.u32 %{stem}_lane, %laneid;",
        f"mad.lo.u32 %{stem}_cta, %{stem}_cy, {GRID_X}, %{stem}_cx;",
        f"mad.lo.u32 %{stem}_thread, %{stem}_cta, {LANES}, %{stem}_lane;",
        f"shl.b32 %{stem}_byte, %{stem}_thread, 5;",
        f"add.u32 %{stem}_byte, %{stem}_byte, {site_offset};",
        f"ld.param.b64 %{stem}_base, [%rd11+{TRACE_PARAM_OFFSET}];",
        f"cvt.u64.u32 %{stem}_wide, %{stem}_byte;",
        f"add.s64 %{stem}_address, %{stem}_base, %{stem}_wide;",
        f"st.global.v4.b32 [%{stem}_address], "
        f"{{{match.group('v0')},{match.group('v1')},{match.group('v2')},{match.group('v3')}}};",
        f"st.global.b32 [%{stem}_address+16], {match.group('x')};",
        f"st.global.b32 [%{stem}_address+20], {match.group('y')};",
        f"st.global.u32 [%{stem}_address+24], {site + 1};",
        "}",
        match.group("store"),
    ]
    return "\n".join(indent + line for line in lines)


def instrument(text: str) -> tuple[str, dict]:
    expanded, param_count = PARAM_DECL.subn(r"\g<1>192\g<2>", text)
    site = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal site
        block = trace_block(match, site)
        site += 1
        return block

    output = SURFACE_STORE.sub(replace, expanded)
    return output, {
        "parameter_declarations_expanded": param_count,
        "surface_store_sites_instrumented": site,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, counts = instrument(source.decode("utf-8"))
    output = output_text.encode("utf-8")
    expected = {
        "parameter_declarations_expanded": 1,
        "surface_store_sites_instrumented": 2,
    }
    status = "PASS" if counts == expected else "FAIL"
    report = {
        "schema": 1,
        "experiment": "postblock_pre_surface_store_full_grid_trace",
        "status": status,
        "classification": "POSTBLOCK_RGB_FIRST_DIVERGENCE_TRACE",
        "grid": [GRID_X, GRID_Y, 1],
        "block": [LANES, 1, 1],
        "record_bytes": RECORD_BYTES,
        "threads_per_site": THREADS,
        "site_bytes": SITE_BYTES,
        "trace_bytes": TRACE_BYTES,
        "trace_param_offset": TRACE_PARAM_OFFSET,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        **counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
