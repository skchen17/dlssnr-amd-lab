#!/usr/bin/env python3
"""Clone a 156-slot plan and replace only slot 1's N0 PTX module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def make_plan(base: Path, n0_ptx: Path, output: Path, variant: str) -> dict:
    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    slots = plan.get("slots", [])
    if len(slots) != 156 or [int(item["slot"]) for item in slots] != list(range(156)):
        raise ValueError("base plan must contain contiguous slots 0..155")
    slot = slots[1]
    if slot.get("function") != "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8":
        raise ValueError("slot 1 is not the expected N0 function")
    old_ptx = slot["ptx"]
    old_hash = slot["ptx_sha256"]
    slot["ptx"] = str(n0_ptx.resolve(strict=True))
    slot["ptx_sha256"] = sha256(n0_ptx)
    plan["experiment"] = "rx9070xt_full_graph_n0_variant_plan"
    plan["variant"] = {
        "name": variant,
        "changed_slots": [1],
        "old_ptx": old_ptx,
        "old_ptx_sha256": old_hash,
        "new_ptx": slot["ptx"],
        "new_ptx_sha256": slot["ptx_sha256"],
    }
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan["variant"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--base",required=True,type=Path); parser.add_argument("--n0-ptx",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); parser.add_argument("--variant",required=True); args=parser.parse_args()
    variant=make_plan(args.base.resolve(),args.n0_ptx.resolve(),args.output.resolve(),args.variant);print(json.dumps(variant,indent=2));return 0


if __name__ == "__main__":
    raise SystemExit(main())
