#!/usr/bin/env python3
"""Add same-capture integrated checkpoints for early 1h/2h Swin slots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


SPECS = {
    3: ("e4m3", 1_966_080),
    4: ("e4m3", 1_966_080),
    6: ("fp16", 983_040),
    7: ("fp16", 983_040),
    8: ("fp16", 983_040),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(
    base: Path,
    archive_path: Path,
    output: Path,
    specs: dict[int, tuple[str, int]] = SPECS,
) -> dict:
    base = base.resolve(strict=True)
    archive_path = archive_path.resolve(strict=True)
    output = output.resolve()
    if output.parent != base.parent:
        raise ValueError("diagnostic plan must remain beside the base plan")

    plan = json.loads(base.read_text(encoding="utf-8-sig"))
    slots = {int(record["slot"]): record for record in plan["slots"]}
    refs_dir = output.parent / "references"
    refs_dir.mkdir(exist_ok=True)
    added = []

    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(archive.read(members["full_graph_capture.json"]).decode("utf-8-sig"))
        windows = capture["windows"]

        def checked_before(window: dict) -> bytes:
            name = window["before_blob"].replace("\\", "/")
            data = archive.read(members[name])
            if (len(data) != int(window["capture_bytes"])
                    or sha256(data) != window["before_sha256"].upper()):
                raise ValueError(
                    f"slot {window['slot']} +{window['param_offset']} before blob mismatch"
                )
            return data

        for slot, (tensor_type, logical_bytes) in specs.items():
            record = slots[slot]
            outputs = [
                view for view in record["activation_param_views"]
                if int(view["param_offset"]) == 8
            ]
            if len(outputs) != 1:
                raise ValueError(f"slot {slot} does not have one main output view at +8")
            arena_offset = int(outputs[0]["arena_offset"])
            consumers = [
                item for item in windows
                if int(item["slot"]) == slot + 1
                and int(item["resource_offset"]) == arena_offset
            ]
            if not consumers:
                raise ValueError(f"slot {slot} output has no adjacent same-capture consumer")
            # Resource offsets are unique within the activation arena. If the
            # capture contains aliases, require their settled states to agree.
            references = [checked_before(item)[:logical_bytes] for item in consumers]
            if any(len(data) != logical_bytes for data in references):
                raise ValueError(f"slot {slot} settled reference is shorter than the logical tensor")
            if any(data != references[0] for data in references[1:]):
                raise ValueError(f"slot {slot} adjacent consumers disagree on settled state")
            reference = references[0]
            ref_path = refs_dir / f"slot{slot}_same_capture.raw"
            if ref_path.exists() and ref_path.read_bytes() != reference:
                raise ValueError(f"refusing to overwrite differing {ref_path}")
            ref_path.write_bytes(reference)
            checkpoint = {
                "arena_offset": arena_offset,
                "logical_bytes": logical_bytes,
                "tensor_type": tensor_type,
                "reference": {
                    "path": f"references/{ref_path.name}",
                    "bytes": logical_bytes,
                    "sha256": sha256(reference),
                },
                "oracle_source": f"slot{slot + 1}_input_before_same_full_graph_capture",
            }
            plan["checkpoints"][str(slot)] = checkpoint
            record["checkpoint"] = checkpoint
            added.append(slot)

    plan["experiment"] = "rx9070xt_full_graph_n0_candidate_early_swin_same_capture_plan"
    variant = plan.setdefault("variant", {})
    variant["diagnostic_checkpoints_added"] = added
    variant["checkpoint_capture_alignment"] = "same_full_graph_archive"
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "diagnostic_checkpoints_added": added,
            "checkpoint_capture_alignment": "same_full_graph_archive"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = make_plan(args.base, args.archive, args.output)
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
