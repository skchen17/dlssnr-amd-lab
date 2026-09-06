#!/usr/bin/env python3
"""Create a provenance-marked lg2 model with selected intercept ULP changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bias", action="append", required=True, help="SEGMENT:SIGNED_U32_DELTA")
    args = parser.parse_args()
    model = json.loads(args.input.read_text(encoding="utf-8"))
    biases = []
    for value in args.bias:
        segment_text, delta_text = value.split(":", 1)
        segment, delta = int(segment_text), int(delta_text)
        rows = model["coefficients_u32"]
        if not 0 <= segment < len(rows):
            raise ValueError(f"segment out of range: {segment}")
        before = rows[segment][0]
        rows[segment][0] = (before + delta) & 0xFFFFFFFF
        biases.append({"segment": segment, "intercept_u32_before": before,
                       "delta_u32": delta, "intercept_u32_after": rows[segment][0]})
    model["experiment"] = model["experiment"] + "_selected_intercept_bias"
    model["parent_model"] = str(args.input)
    model["intercept_biases"] = biases
    model["classification"] = "RTX5070_PRODUCTION_MODEL_WITH_HALF_SAFE_SEGMENT_BIASES"
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "biases": biases}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
