#!/usr/bin/env python3
"""Trace all N0 f16 MMA fragments for one CTA in the production full-grid launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.lower_ptx_f16_mma import F16_MMA
except ModuleNotFoundError:
    from lower_ptx_f16_mma import F16_MMA


LANES = 32
BYTES_PER_LANE_MMA = 40
BASE_SCRATCH_BYTES = 384 * 640 * 32
EXPECTED_MMA = 16
TRACE_BYTES = EXPECTED_MMA * LANES * BYTES_PER_LANE_MMA
TRACE_OFFSET = BASE_SCRATCH_BYTES


def instrument(text: str, target_x: int, target_y: int) -> tuple[str, int]:
    count = 0

    def replace(match):
        nonlocal count
        index = count
        count += 1
        a = [match.group(f"a{i}") for i in range(4)]
        b = [match.group(f"b{i}") for i in range(2)]
        c = [match.group(f"c{i}") for i in range(2)]
        d = [match.group(f"d{i}") for i in range(2)]
        base = TRACE_OFFSET + index * LANES * BYTES_PER_LANE_MMA
        prefix = f"__n0_cta_f16_mma_{index}"
        lines = [
            "{",
            f".reg .pred %{prefix}_px, %{prefix}_py, %{prefix}_selected;",
            f".reg .b32 %{prefix}_lane, %{prefix}_offset, %{prefix}_ctax, %{prefix}_ctay;",
            f".reg .b64 %{prefix}_wide, %{prefix}_address;",
            f"mov.u32 %{prefix}_ctax, %ctaid.x;",
            f"mov.u32 %{prefix}_ctay, %ctaid.y;",
            f"setp.eq.u32 %{prefix}_px, %{prefix}_ctax, {target_x};",
            f"setp.eq.u32 %{prefix}_py, %{prefix}_ctay, {target_y};",
            f"and.pred %{prefix}_selected, %{prefix}_px, %{prefix}_py;",
            f"mov.u32 %{prefix}_lane, %laneid;",
            f"mad.lo.u32 %{prefix}_offset, %{prefix}_lane, {BYTES_PER_LANE_MMA}, {base};",
            f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
            f"add.s64 %{prefix}_address, %rd6, %{prefix}_wide;",
        ]
        for offset, register in enumerate(a + b + c):
            lines.append(f"@%{prefix}_selected st.global.b32 [%{prefix}_address+{offset * 4}], {register};")
        lines.append(match.group(0))
        lines.extend((
            f"@%{prefix}_selected st.global.b32 [%{prefix}_address+32], {d[0]};",
            f"@%{prefix}_selected st.global.b32 [%{prefix}_address+36], {d[1]};",
            "}",
        ))
        return "\n".join(lines)

    return F16_MMA.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--target-x", type=int, required=True)
    parser.add_argument("--target-y", type=int, required=True)
    args = parser.parse_args()
    if not (0 <= args.target_x < 80 and 0 <= args.target_y < 48):
        parser.error("target CTA must be inside the 80x48 N0 grid")
    source = args.input.read_bytes()
    output_text, count = instrument(source.decode("utf-8"), args.target_x, args.target_y)
    output = output_text.encode("utf-8")
    passed = count == EXPECTED_MMA and len(F16_MMA.findall(output_text)) == EXPECTED_MMA
    report = {
        "schema": 1,
        "experiment": "n0_f16_mma_selected_cta_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "N0_FIRST_OBSERVABLE_FP8_INPUT_PRODUCER_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "mma_count": count,
        "lanes": LANES,
        "bytes_per_lane_mma": BYTES_PER_LANE_MMA,
        "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "scratch_extra_bytes": TRACE_BYTES,
        "scratch_register": "%rd6",
        "target_cta": [args.target_x, args.target_y, 0],
        "required_grid": [80, 48, 1],
        "required_block": [32, 1, 1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
