#!/usr/bin/env python3
"""Apply the observed zero-input RTX r2723 value as a scoped causal ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGET = re.compile(r"\{add\.f16x2 %r2723,%r2672,%r2673;\s*\}")


BLOCK = """{
.reg .pred %__r2723_lane_ge, %__r2723_lane_le, %__r2723_lane_group, %__r2723_value, %__r2723_apply;
.reg .u32 %__r2723_lane;
mov.u32 %__r2723_lane, %laneid;
setp.ge.u32 %__r2723_lane_ge, %__r2723_lane, 24;
setp.le.u32 %__r2723_lane_le, %__r2723_lane, 27;
and.pred %__r2723_lane_group, %__r2723_lane_ge, %__r2723_lane_le;
setp.eq.u32 %__r2723_value, %r2723, 0x411F411F;
and.pred %__r2723_apply, %__r2723_lane_group, %__r2723_value;
@%__r2723_apply mov.b32 %r2723, 0x41204120;
}"""


def lower(text: str) -> tuple[str, int]:
    matches = list(TARGET.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected one r2723 target, found {len(matches)}")
    return TARGET.sub(lambda match: match.group(0) + "\n" + BLOCK, text, count=1), 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    output_text, count = lower(source.decode("utf-8"))
    output = output_text.encode("utf-8")
    report = {
        "schema": 1,
        "experiment": "n0_r2723_observed_value_ablation",
        "status": "PASS",
        "classification": "CAUSAL_LOCALIZATION_ABLATION_NOT_GENERAL_LOWERING",
        "counts_as_s7": False,
        "source_sha256": hashlib.sha256(source).hexdigest().upper(),
        "output_sha256": hashlib.sha256(output).hexdigest().upper(),
        "targets_lowered": count,
        "scope": "lanes 24..27 and r2723 == 0x411F411F only",
        "replacement": "0x41204120",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
