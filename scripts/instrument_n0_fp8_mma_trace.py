#!/usr/bin/env python3
"""Export all N0 FP8 MMA fragments into an unused one-CTA scratch window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.lower_ptx_fp8_mma import FP8_MMA
except ModuleNotFoundError:
    from lower_ptx_fp8_mma import FP8_MMA


LANES = 32
BYTES_PER_LANE_MMA = 40
TRACE_OFFSET = 4 * 1024 * 1024
EXPECTED_MMA = 256
TRACE_BYTES = EXPECTED_MMA * LANES * BYTES_PER_LANE_MMA


def instrument(text: str) -> tuple[str, int]:
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
        prefix = f"__n0_mma_{index}"
        lines = [
            "{",
            f".reg .b32 %{prefix}_lane, %{prefix}_offset;",
            f".reg .b64 %{prefix}_wide, %{prefix}_address;",
            f"mov.u32 %{prefix}_lane, %laneid;",
            f"mad.lo.u32 %{prefix}_offset, %{prefix}_lane, {BYTES_PER_LANE_MMA}, {base};",
            f"cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;",
            f"add.s64 %{prefix}_address, %rd6, %{prefix}_wide;",
        ]
        for offset, register in enumerate(a + b + c):
            lines.append(f"st.global.b32 [%{prefix}_address+{offset * 4}], {register};")
        lines.append(match.group(0))
        lines.extend((
            f"st.global.b32 [%{prefix}_address+32], {d[0]};",
            f"st.global.b32 [%{prefix}_address+36], {d[1]};",
            "}",
        ))
        return "\n".join(lines)

    return FP8_MMA.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, count = instrument(source.decode("utf-8"))
    output = output_text.encode("utf-8")
    passed = count == EXPECTED_MMA and len(FP8_MMA.findall(output_text)) == EXPECTED_MMA
    report = {
        "schema": 1,
        "experiment": "n0_fp8_mma_register_trace_instrumentation",
        "status": "PASS" if passed else "FAIL",
        "classification": "N0_INTERNAL_NUMERICAL_DIVERGENCE_TRACE",
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "mma_count": count,
        "lanes": LANES,
        "bytes_per_lane_mma": BYTES_PER_LANE_MMA,
        "trace_offset": TRACE_OFFSET,
        "trace_bytes": TRACE_BYTES,
        "scratch_register": "%rd6",
        "required_grid": [1, 1, 1],
        "required_block": [32, 1, 1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
