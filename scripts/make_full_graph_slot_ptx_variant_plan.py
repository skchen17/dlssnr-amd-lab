#!/usr/bin/env python3
"""Create an auditable full-graph plan replacing explicitly selected slot PTX files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(plan: dict, replacements: dict[int, Path], name: str) -> dict:
    output = json.loads(json.dumps(plan))
    slots = output.get("slots", [])
    if len(slots) != 156 or [int(item.get("slot", -1)) for item in slots] != list(range(156)):
        raise ValueError("base plan is not a contiguous 156-slot plan")
    records = []
    for slot in sorted(replacements):
        if not 0 <= slot < 156:
            raise ValueError(f"invalid slot {slot}")
        ptx = replacements[slot].resolve(strict=True)
        previous = slots[slot]["ptx"]
        digest = hashlib.sha256(ptx.read_bytes()).hexdigest().upper()
        slots[slot]["ptx"] = str(ptx)
        slots[slot]["ptx_sha256"] = digest
        records.append({"slot": slot, "previous_ptx": previous,
                        "ptx": str(ptx), "ptx_sha256": digest})
    if not records:
        raise ValueError("at least one replacement is required")
    output["experiment"] = "rx9070xt_full_graph_explicit_slot_ptx_variant"
    output["classification"] = "SCOPE_CONTROLLED_MULTI_SLOT_ARITHMETIC_VARIANT"
    output["variant"] = {"name": name, "replacements": records,
                         "counts_as_s7": False}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replace", action="append", required=True,
                        help="SLOT=PTX, repeat for every explicit replacement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    replacements: dict[int, Path] = {}
    for spec in args.replace:
        slot_text, separator, path_text = spec.partition("=")
        if not separator:
            parser.error(f"invalid replacement {spec!r}; expected SLOT=PTX")
        slot = int(slot_text)
        if slot in replacements:
            parser.error(f"duplicate slot {slot}")
        replacements[slot] = Path(path_text)
    plan = json.loads(args.base.read_text(encoding="utf-8-sig"))
    output = build(plan, replacements, args.name)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["variant"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
