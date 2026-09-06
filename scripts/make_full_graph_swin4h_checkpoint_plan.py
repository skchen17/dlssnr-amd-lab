#!/usr/bin/env python3
"""Add integrated slots 10-14 FP16 checkpoints using existing RTX native states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SLOTS = range(10, 15)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(base: Path, cases_root: Path, output: Path) -> dict:
    base = base.resolve(strict=True)
    cases_root = cases_root.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("diagnostic plan must remain beside the base plan")
    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    slots = {int(record["slot"]): record for record in plan["slots"]}
    cases_manifest = json.loads((cases_root / "manifest.json").read_text(encoding="utf-8-sig"))
    cases = {int(case["slot"]): case for case in cases_manifest["slots"]}

    added = []
    for slot in SLOTS:
        record = slots[slot]
        outputs = [view for view in record["activation_param_views"]
                   if int(view["param_offset"]) == 8]
        if len(outputs) != 1:
            raise ValueError(f"slot {slot} does not have one main output view at +8")
        case = cases[slot]
        logical = int(case["main_logical_bytes"])
        if case.get("main_tensor_type", "fp16") != "fp16":
            raise ValueError(f"slot {slot} main tensor is not FP16")
        reference = cases_root / f"slot{slot}" / case["assets"]["output_reference"]["path"]
        data = reference.read_bytes()
        asset = case["assets"]["output_reference"]
        if len(data) != int(asset["bytes"]) or sha256(data) != asset["sha256"]:
            raise ValueError(f"slot {slot} RTX reference integrity failed")
        prefix = data[:logical]
        relative = os.path.relpath(reference, output.parent).replace("\\", "/")
        plan["checkpoints"][str(slot)] = {
            "arena_offset": int(outputs[0]["arena_offset"]),
            "logical_bytes": logical,
            "tensor_type": "fp16",
            "reference": {
                "path": relative,
                "bytes": logical,
                "sha256": sha256(prefix),
            },
            "oracle_source": f"slot{slot}_native_after_logical_prefix",
        }
        record["checkpoint"] = plan["checkpoints"][str(slot)]
        added.append(slot)

    plan["experiment"] = "rx9070xt_full_graph_n0_candidate_swin4h_diagnostic_plan"
    plan.setdefault("variant", {})["diagnostic_checkpoints_added"] = added
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "diagnostic_checkpoints_added": added}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = make_plan(args.base, args.cases, args.output)
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
