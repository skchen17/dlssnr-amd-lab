#!/usr/bin/env python3
"""Create a non-S7 plan that restores a complete exact-state activation arena."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(base: Path, cases_root: Path, output: Path,
              case_slot: int, after_slot: int,
              preserve_case_special_params: bool = False) -> dict:
    base = base.resolve(strict=True)
    cases_root = cases_root.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("diagnostic plan must remain beside the base plan")

    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    manifest_path = cases_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if plan["source_archive"]["sha256"].upper() != manifest["source_archive"]["sha256"].upper():
        raise ValueError("integrated plan and exact-state cases use different RTX archives")

    case = next(
        (record for record in manifest["slots"] if int(record["slot"]) == case_slot),
        None,
    )
    if case is None:
        raise ValueError(f"exact-state case slot {case_slot} is absent")
    arena_spec = case["activation_arena"]
    arena_path = cases_root / f"slot{case_slot}" / arena_spec["path"]
    arena = arena_path.read_bytes()
    expected_bytes = int(arena_spec["bytes"])
    expected_sha = arena_spec["sha256"].upper()
    if len(arena) != expected_bytes or sha256(arena) != expected_sha:
        raise ValueError(f"slot-{case_slot} exact-state activation arena integrity failed")
    if expected_bytes != int(plan["activation_arena_initial"]["bytes"]):
        raise ValueError("exact-state and integrated activation arena sizes differ")
    if not any(int(record["slot"]) == after_slot for record in plan["slots"]):
        raise ValueError(f"integrated plan slot {after_slot} is absent")
    case_record = next(
        (record for record in plan["slots"] if int(record["slot"]) == case_slot),
        None,
    )
    if case_record is None:
        raise ValueError(f"integrated plan case slot {case_slot} is absent")
    original_special = case_record.get("special", "normal")
    if preserve_case_special_params:
        case_record["special"] = "normal"

    reference_path = os.path.relpath(arena_path, output.parent).replace("\\", "/")
    plan["experiment"] = "rx9070xt_full_graph_exact_state_full_arena_injection_plan"
    plan["diagnostic_injections"] = [{
        "after_slot": after_slot,
        "arena_offset": 0,
        "reference_offset": 0,
        "bytes": expected_bytes,
        "tensor_type": "activation_arena_raw",
        "reference": {
            "path": reference_path,
            "bytes": expected_bytes,
            "sha256": expected_sha,
        },
        "classification": "RTX_FULL_ARENA_DIAGNOSTIC_INJECTION_NOT_S7",
    }]
    plan["counts_as_s7"] = False
    plan["injection_warning"] = (
        "The complete RTX activation arena is restored immediately after the selected slot; "
        "this causal diagnostic can never count as S7 or end-to-end AMD evidence."
    )
    plan.setdefault("variant", {})["full_arena_injection"] = {
        "case_slot": case_slot,
        "after_slot": after_slot,
        "capture_alignment": "same_full_graph_archive",
        "original_special": original_special,
        "preserve_case_special_params": preserve_case_special_params,
    }
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "case_slot": case_slot,
        "after_slot": after_slot,
        "bytes": expected_bytes,
        "sha256": expected_sha,
        "counts_as_s7": False,
        "preserve_case_special_params": preserve_case_special_params,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-slot", type=int, required=True)
    parser.add_argument("--after-slot", type=int, required=True)
    parser.add_argument("--preserve-case-special-params", action="store_true")
    args = parser.parse_args()
    print(json.dumps(make_plan(
        args.base, args.cases, args.output, args.case_slot, args.after_slot,
        args.preserve_case_special_params,
    ), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
