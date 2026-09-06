#!/usr/bin/env python3
"""Create an explicitly non-S7 plan that injects RTX state after selected slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_plan(base: Path, output: Path, after_slots: list[int]) -> dict:
    base = base.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("injection plan must remain beside the base plan")
    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    checkpoints = plan["checkpoints"]
    unique = sorted(set(after_slots))
    if len(unique) != len(after_slots):
        raise ValueError("duplicate injection slot")
    injections = []
    for slot in unique:
        checkpoint = checkpoints.get(str(slot))
        if checkpoint is None:
            raise ValueError(f"slot {slot} has no reference checkpoint")
        injections.append({
            "after_slot": slot,
            "arena_offset": int(checkpoint["arena_offset"]),
            "bytes": int(checkpoint["logical_bytes"]),
            "tensor_type": checkpoint["tensor_type"],
            "reference": checkpoint["reference"],
            "classification": "RTX_REFERENCE_STATE_DIAGNOSTIC_INJECTION_NOT_S7",
        })
    plan["experiment"] = "rx9070xt_full_graph_rtx_state_injection_sensitivity_plan"
    plan["diagnostic_injections"] = injections
    plan["counts_as_s7"] = False
    plan["injection_warning"] = (
        "RTX intermediate state is injected for causal sensitivity analysis; "
        "this plan can never count as S7 or end-to-end AMD evidence."
    )
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "after_slots": unique, "counts_as_s7": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after-slot", type=int, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.base, args.output, args.after_slot), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
