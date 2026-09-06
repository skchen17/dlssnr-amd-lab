#!/usr/bin/env python3
"""Extract one exact RTX launch state for lowering-variant diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def extract(archive_path: Path, plan_path: Path, slot: int, output_param: int,
            logical_bytes: int, output_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    record = plan["slots"][slot]
    output_dir.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive_path) as archive:
        members = {item.filename.replace("\\", "/"): item for item in archive.infolist()}
        capture = json.loads(archive.read(members["full_graph_capture.json"]).decode("utf-8-sig"))
        windows = capture["windows"]
        current = [item for item in windows if int(item["slot"]) == slot
                   and item["resource"] == plan["activation_resource"]]

        def blob(window: dict, phase: str) -> bytes:
            data = archive.read(members[window[f"{phase}_blob"].replace("\\", "/")])
            if sha256(data) != window[f"{phase}_sha256"].upper():
                raise ValueError("capture blob hash mismatch")
            return data

        arena = bytearray(int(plan["activation_arena_bytes"]))
        for window in current:
            start = int(window["resource_offset"])
            data = blob(window, "before")
            arena[start:start + len(data)] = data
        arena_path = output_dir / "activation_arena_before.raw"
        arena_path.write_bytes(arena)
        producer = next(item for item in current if int(item["param_offset"]) == output_param)
        producer_offset = int(producer["resource_offset"])
        consumers = [item for item in windows if int(item["slot"]) == slot + 1
                     and item["resource"] == plan["activation_resource"]
                     and int(item["resource_offset"]) == producer_offset]
        if not consumers:
            raise ValueError("diagnostic output has no adjacent settled consumer")
        reference = blob(consumers[0], "before")[:logical_bytes]
        reference_path = output_dir / "output_reference.raw"
        reference_path.write_bytes(reference)
    result = {
        "schema": 1,
        "status": "PASS",
        "slot": slot,
        "record": record,
        "arena": {"path": str(arena_path), "bytes": len(arena), "sha256": sha256(arena)},
        "output": {"param_offset": output_param, "arena_offset": producer_offset,
                   "logical_bytes": logical_bytes, "path": str(reference_path),
                   "sha256": sha256(reference)},
        "model_arena": str((plan_path.parent / plan["model_arena"]["path"]).resolve()),
        "params": str((plan_path.parent / record["params"]["path"]).resolve()),
    }
    (output_dir / "case.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--slot", required=True, type=int)
    parser.add_argument("--output-param", required=True, type=int)
    parser.add_argument("--logical-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = extract(args.archive.resolve(), args.plan.resolve(), args.slot,
                     args.output_param, args.logical_bytes, args.output.resolve())
    print(json.dumps({"status": result["status"], "slot": result["slot"],
                      "output_sha256": result["output"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
