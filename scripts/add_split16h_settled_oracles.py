#!/usr/bin/env python3
"""Add synchronization-settled chained-output oracles to extracted split16h cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def asset(path: Path, data: bytes) -> dict:
    path.write_bytes(data)
    return {"path": path.name, "bytes": len(data), "sha256": sha256(data)}


def add_oracles(cases_root: Path) -> dict:
    manifest_path = cases_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    slots = {int(case["slot"]): case for case in manifest["slots"]}
    if set(range(24, 57)) - slots.keys():
        raise ValueError("split16h case set must contain every slot from 24 through 56")

    for slot in range(24, 55):
        case, consumer = slots[slot], slots[slot + 1]
        if (case["activation_arena_resource"] != consumer["activation_arena_resource"] or
                int(case["arena_views"]["output"]) !=
                int(consumer["arena_views"]["input"])):
            raise ValueError(f"slot {slot} output is not slot {slot + 1} input")
        data = (cases_root / f"slot{slot + 1}" / "input.raw").read_bytes()
        case["assets"]["settled_output_reference"] = asset(
            cases_root / f"slot{slot}" / "settled_output_reference.raw", data)
        case["main_oracle"] = "downstream_input_before"

    slot55, slot56 = slots[55], slots[56]
    if (slot55["activation_arena_resource"] != slot56["activation_arena_resource"] or
            int(slot55["arena_views"]["extra"]) !=
            int(slot56["arena_views"]["input"])):
        raise ValueError("slot 55 extra output is not slot 56 input")
    data = (cases_root / "slot56" / "input.raw").read_bytes()
    slot55["assets"]["settled_extra_output_reference"] = asset(
        cases_root / "slot55" / "settled_extra_output_reference.raw", data)
    slot55["extra_oracle"] = "downstream_input_before"
    slot55.setdefault("main_oracle", "immediate_after")
    slot56.setdefault("main_oracle", "immediate_after")
    slot56.setdefault("extra_oracle", None)

    manifest["oracle_policy"] = {
        "chained_outputs": "next consumer before snapshot after dependency wait",
        "terminal_outputs": "producer immediate after snapshot",
        "reason": (
            "launch capture is asynchronous; immediate producer after snapshots can "
            "be stale or partial, while the aliased next consumer input is observed "
            "after the graph dependency wait"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_root", type=Path)
    args = parser.parse_args()
    report = add_oracles(args.cases_root.resolve())
    print(json.dumps({"status": "PASS", "slot_count": len(report["slots"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
