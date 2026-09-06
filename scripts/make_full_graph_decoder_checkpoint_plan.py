#!/usr/bin/env python3
"""Add same-capture settled decoder checkpoints to an integrated graph plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(base: Path, cases_root: Path, output: Path,
              first_slot: int = 99, last_slot: int = 153) -> dict:
    base = base.resolve(strict=True)
    cases_root = cases_root.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("diagnostic plan must remain beside the base plan")
    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    manifest = json.loads((cases_root / "manifest.json").read_text(encoding="utf-8-sig"))
    if plan["source_archive"]["sha256"].upper() != manifest["source_archive"]["sha256"].upper():
        raise ValueError("integrated plan and decoder cases are not from the same RTX archive")
    slots = {int(record["slot"]): record for record in plan["slots"]}
    cases = {int(record["slot"]): record for record in manifest["slots"]}
    added = []

    for slot in range(first_slot, last_slot + 1):
        case = cases.get(slot)
        if case is None:
            raise ValueError(f"decoder case is missing slot {slot}")
        outputs = case.get("outputs", [])
        if len(outputs) != 1:
            raise ValueError(f"slot {slot} does not have exactly one settled main output")
        source = outputs[0]
        asset = source["asset"]
        reference = cases_root / f"slot{slot}" / asset["path"]
        data = reference.read_bytes()
        if len(data) != int(asset["bytes"]) or sha256(data) != asset["sha256"].upper():
            raise ValueError(f"slot {slot} reference integrity failed")
        logical_bytes = int(source["logical_bytes"])
        if len(data) < logical_bytes:
            raise ValueError(f"slot {slot} reference is shorter than its logical tensor")
        reference = data[:logical_bytes]

        record = slots[slot]
        matching_views = [
            view for view in record["activation_param_views"]
            if int(view["param_offset"]) == int(source["param_offset"])
            and int(view["arena_offset"]) == int(source["arena_offset"])
        ]
        if len(matching_views) != 1:
            raise ValueError(f"slot {slot} integrated output view differs from decoder case")
        relative = os.path.relpath(
            cases_root / f"slot{slot}" / asset["path"], output.parent
        ).replace("\\", "/")
        checkpoint = {
            "arena_offset": int(source["arena_offset"]),
            "logical_bytes": logical_bytes,
            "tensor_type": source["tensor_type"],
            "reference": {
                "path": relative,
                "bytes": logical_bytes,
                "sha256": sha256(reference),
            },
            "oracle_source": source["oracle_source"] + "_same_full_graph_capture",
        }
        plan["checkpoints"][str(slot)] = checkpoint
        record["checkpoint"] = checkpoint
        added.append(slot)

    plan["experiment"] = "rx9070xt_full_graph_integrated_decoder_same_capture_checkpoint_plan"
    variant = plan.setdefault("variant", {})
    prior = [int(slot) for slot in variant.get("diagnostic_checkpoints_added", [])]
    variant["diagnostic_checkpoints_added"] = sorted(set(prior + added))
    variant["decoder_checkpoints_added"] = added
    variant["checkpoint_capture_alignment"] = "same_full_graph_archive"
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "decoder_checkpoints_added": added,
        "checkpoint_capture_alignment": "same_full_graph_archive",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-slot", type=int, default=99)
    parser.add_argument("--last-slot", type=int, default=153)
    args = parser.parse_args()
    report = make_plan(
        args.base, args.cases, args.output, args.first_slot, args.last_slot
    )
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
