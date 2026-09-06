#!/usr/bin/env python3
"""Create a non-S7 plan that injects one slice of an RTX checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(base: Path, output: Path, after_slot: int,
              reference_offset: int, size: int) -> dict:
    base = base.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("partial injection plan must remain beside the base plan")
    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    checkpoint = plan["checkpoints"].get(str(after_slot))
    if checkpoint is None:
        raise ValueError(f"slot {after_slot} has no reference checkpoint")
    logical = int(checkpoint["logical_bytes"])
    if reference_offset < 0 or size <= 0 or reference_offset + size > logical:
        raise ValueError("partial injection slice is outside the logical checkpoint")
    reference_path = base.parent / checkpoint["reference"]["path"]
    raw = reference_path.read_bytes()
    if len(raw) < logical:
        raise ValueError("checkpoint reference is shorter than its logical tensor")
    data = raw[reference_offset:reference_offset + size]
    injection = {
        "after_slot": after_slot,
        "arena_offset": int(checkpoint["arena_offset"]) + reference_offset,
        "reference_offset": reference_offset,
        "bytes": size,
        "tensor_type": checkpoint["tensor_type"],
        "reference": {
            "path": checkpoint["reference"]["path"],
            "bytes": size,
            "sha256": sha256(data),
        },
        "classification": "RTX_REFERENCE_STATE_PARTIAL_DIAGNOSTIC_INJECTION_NOT_S7",
    }
    plan["experiment"] = "rx9070xt_full_graph_partial_rtx_state_injection_sensitivity_plan"
    plan["diagnostic_injections"] = [injection]
    plan["counts_as_s7"] = False
    plan["injection_warning"] = (
        "A slice of RTX intermediate state is injected for causal sensitivity analysis; "
        "this plan can never count as S7 or end-to-end AMD evidence."
    )
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "after_slot": after_slot,
        "reference_offset": reference_offset,
        "bytes": size,
        "sha256": injection["reference"]["sha256"],
        "counts_as_s7": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after-slot", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.base, args.output, args.after_slot,
                               args.offset, args.bytes), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
