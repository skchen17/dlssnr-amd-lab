#!/usr/bin/env python3
"""Create a scoped full-graph plan that replaces only slot 154 PTX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    plan = json.loads(args.base.read_text(encoding="utf-8-sig"))
    if len(plan.get("slots", [])) != 156 or plan["slots"][154].get("slot") != 154:
        raise ValueError("base plan is not a contiguous 156-slot plan")
    ptx = args.ptx.resolve()
    if not ptx.is_file():
        raise ValueError(f"variant PTX not found: {ptx}")
    previous = plan["slots"][154]["ptx"]
    plan["experiment"] = "rx9070xt_full_graph_postblock_variant_plan"
    plan["classification"] = "SCOPE_CONTROLLED_SLOT154_FUSION_VARIANT"
    plan["variant"] = {
        "name": args.name, "slot": 154, "previous_ptx": previous,
        "ptx": str(ptx),
        "ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest().upper(),
    }
    plan["slots"][154]["ptx"] = str(ptx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan["variant"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
