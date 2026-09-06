#!/usr/bin/env python3
"""Inject the complete live RTX pre-slot154 activation buffer for diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make(base: Path, activation: Path, output: Path) -> dict:
    base = base.resolve(strict=True)
    activation = activation.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("diagnostic plan must remain beside the base plan")
    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    data = activation.read_bytes()
    if not (16 * 1024 * 1024 <= len(data) <= int(plan["activation_arena_bytes"])):
        raise ValueError("live activation buffer size is outside the plan arena")
    plan["experiment"] = "rx9070xt_full_graph_live_pre_slot154_activation_injection_plan"
    plan["counts_as_s7"] = False
    plan["diagnostic_injections"] = [{
        "after_slot": 153,
        "arena_offset": 0,
        "reference_offset": 0,
        "bytes": len(data),
        "tensor_type": "complete_live_rtx_activation_buffer_raw",
        "reference": {
            "path": os.path.relpath(activation, output.parent).replace("\\", "/"),
            "bytes": len(data),
            "sha256": sha256(data),
        },
        "classification": "RTX_LIVE_FULL_ACTIVATION_DIAGNOSTIC_INJECTION_NOT_S7",
    }]
    plan["injection_warning"] = (
        "The complete live RTX activation buffer is restored after slot 153. "
        "This causal diagnostic can never count as S7 or end-to-end AMD evidence."
    )
    plan.setdefault("variant", {})["live_pre_slot154_activation"] = {
        "after_slot": 153,
        "bytes": len(data),
        "sha256": sha256(data),
        "capture_timing": "same command list immediately before RTX frame-1 slot154",
    }
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan["variant"]["live_pre_slot154_activation"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(make(args.base, args.activation, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
