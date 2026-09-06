#!/usr/bin/env python3
"""Expose the same-capture N0 high-resolution scratch as slot-1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCRATCH_OFFSET = 110_592
SCRATCH_BYTES = 7_864_320


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(base: Path, cases_root: Path, output: Path,
              checkpoint_slot: int = 1) -> dict:
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
    slot1 = slots[1]
    slot154 = slots[154]
    if not any(int(view["param_offset"]) == 216
               and int(view["arena_offset"]) == SCRATCH_OFFSET
               for view in slot1["activation_param_views"]):
        raise ValueError("slot 1 does not expose the expected N0 scratch")
    if not any(int(view["param_offset"]) == 8
               and int(view["arena_offset"]) == SCRATCH_OFFSET
               for view in slot154["activation_param_views"]):
        raise ValueError("slot 154 does not consume the expected N0 scratch")

    case154 = next(record for record in manifest["slots"] if int(record["slot"]) == 154)
    arena_spec = case154["activation_arena"]
    arena_path = cases_root / "slot154" / arena_spec["path"]
    arena = arena_path.read_bytes()
    if len(arena) != int(arena_spec["bytes"]) or sha256(arena) != arena_spec["sha256"].upper():
        raise ValueError("slot-154 exact-state activation arena integrity failed")
    scratch = arena[SCRATCH_OFFSET:SCRATCH_OFFSET + SCRATCH_BYTES]
    if len(scratch) != SCRATCH_BYTES:
        raise ValueError("same-capture N0 scratch is truncated")
    refs = output.parent / "references"
    refs.mkdir(exist_ok=True)
    reference_path = refs / "slot1_scratch_same_capture.raw"
    if reference_path.exists() and reference_path.read_bytes() != scratch:
        raise ValueError(f"refusing to overwrite differing {reference_path}")
    reference_path.write_bytes(scratch)
    checkpoint = {
        "arena_offset": SCRATCH_OFFSET,
        "logical_bytes": SCRATCH_BYTES,
        "tensor_type": "e4m3",
        "reference": {
            "path": f"references/{reference_path.name}",
            "bytes": SCRATCH_BYTES,
            "sha256": sha256(scratch),
        },
        "oracle_source": "slot154_n0_scratch_input_before_same_full_graph_capture",
    }
    checkpoint_record = slots.get(checkpoint_slot)
    if checkpoint_record is None:
        raise ValueError(f"checkpoint slot {checkpoint_slot} is absent")
    plan["checkpoints"][str(checkpoint_slot)] = checkpoint
    checkpoint_record["checkpoint"] = checkpoint
    plan["experiment"] = "rx9070xt_full_graph_n0_scratch_same_capture_checkpoint_plan"
    variant = plan.setdefault("variant", {})
    variant["n0_scratch_checkpoint_slot"] = checkpoint_slot
    variant["checkpoint_capture_alignment"] = "same_full_graph_archive"
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "slot": checkpoint_slot,
        "arena_offset": SCRATCH_OFFSET,
        "bytes": SCRATCH_BYTES,
        "sha256": sha256(scratch),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-slot", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(
        make_plan(args.base, args.cases, args.output, args.checkpoint_slot),
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
