#!/usr/bin/env python3
"""Lower formatted RGBA16F 2D surface stores to a linear-buffer ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SURFACE_STORE = re.compile(
    r"(?m)^(?P<i>[ \t]*)sust\.p\.2d\.v4\.b32\.zero\s+"
    r"\[(?P<surface>%rd[0-9]+),\s*\{(?P<x>%r[0-9]+),(?P<y>%r[0-9]+)\}\],\s*"
    r"\{(?P<v0>%r[0-9]+),(?P<v1>%r[0-9]+),(?P<v2>%r[0-9]+),(?P<v3>%r[0-9]+)\};[ \t]*$"
)


def lower(text: str, width_param_offset: int = 172) -> tuple[str, dict]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        index = count
        count += 1
        indent = match.group("i")
        stem = f"__dlssnr_surface_{index}"
        return "\n".join((
            indent + "{",
            indent + f".reg .b64 %{stem}_row, %{stem}_index, %{stem}_x64, "
                     f"%{stem}_byte, %{stem}_address;",
            indent + f".reg .b32 %{stem}_width, %{stem}_packed01, %{stem}_packed23;",
            indent + f".reg .b16 %{stem}_h0, %{stem}_h1, %{stem}_h2, %{stem}_h3;",
            indent + f"ld.param.b32 %{stem}_width, [%rd11+{width_param_offset}];",
            indent + f"mul.wide.u32 %{stem}_row, {match.group('y')}, %{stem}_width;",
            indent + f"cvt.u64.u32 %{stem}_x64, {match.group('x')};",
            indent + f"add.u64 %{stem}_index, %{stem}_row, %{stem}_x64;",
            indent + f"shl.b64 %{stem}_byte, %{stem}_index, 3;",
            indent + f"add.u64 %{stem}_address, {match.group('surface')}, %{stem}_byte;",
            indent + f"cvt.rn.f16.f32 %{stem}_h0, {match.group('v0')};",
            indent + f"cvt.rn.f16.f32 %{stem}_h1, {match.group('v1')};",
            indent + f"cvt.rn.f16.f32 %{stem}_h2, {match.group('v2')};",
            indent + f"cvt.rn.f16.f32 %{stem}_h3, {match.group('v3')};",
            indent + f"mov.b32 %{stem}_packed01, {{%{stem}_h0, %{stem}_h1}};",
            indent + f"mov.b32 %{stem}_packed23, {{%{stem}_h2, %{stem}_h3}};",
            indent + f"st.global.v2.b32 [%{stem}_address], "
                     f"{{%{stem}_packed01, %{stem}_packed23}};",
            indent + "}",
        ))

    lowered = SURFACE_STORE.sub(replace, text)
    return lowered, {
        "rgba16f_surface_stores_lowered": count,
        "remaining_rgba16f_surface_stores": len(SURFACE_STORE.findall(lowered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--width-param-offset", type=int, default=172)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    lowered, counts = lower(
        source_bytes.decode("utf-8"), args.width_param_offset)
    lowered_bytes = lowered.encode("utf-8")
    status = "PASS" if counts == {
        "rgba16f_surface_stores_lowered": args.expected_count,
        "remaining_rgba16f_surface_stores": 0,
    } else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_rgba16f_surface_linear_abi_lowering",
        "status": status,
        "classification": "EXPLICIT_RESOURCE_ABI_ADAPTATION",
        "counts_as_s7": False,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(lowered_bytes).hexdigest(),
        "expected_count": args.expected_count,
        "width_param_offset": args.width_param_offset,
        **counts,
        "semantics": (
            "The CUDA formatted-surface handle becomes a linear device pointer. "
            "Each pixel is addressed as (y * width + x) * 8 and the four f32 "
            "components are rounded to R16G16B16A16_FLOAT. RTX slot-155 final "
            "copy bytes are the normative oracle."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(lowered_bytes)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
