#!/usr/bin/env python3
"""Extract one captured graph slot parameter block with integrity metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def extract(csv_path: Path, slot: int, expected_function: str) -> tuple[bytes, dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["slot"]) == slot]
    if len(rows) != 1:
        raise ValueError(f"expected one row for slot {slot}, found {len(rows)}")
    row = rows[0]
    if row["function_name"] != expected_function:
        raise ValueError(f"slot {slot} function is {row['function_name']}")
    raw = bytes.fromhex(row["frame1_param_hex"])
    expected_size = int(row["param_size"])
    if len(raw) != expected_size:
        raise ValueError(f"parameter bytes {len(raw)} != captured size {expected_size}")
    return raw, {
        "schema": 1,
        "experiment": "captured_graph_slot_parameters",
        "source_csv": str(csv_path.resolve()),
        "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest().upper(),
        "slot": slot,
        "function": expected_function,
        "module_dll_offset": row["module_dll_offset"],
        "module_fnv1a64": row["module_fnv1a64"],
        "grid": [int(row["grid_x"]), int(row["grid_y"]), int(row["grid_z"])],
        "block": [int(row["block_x"]), int(row["block_y"]), int(row["block_z"])],
        "dynamic_shared": int(row["dynamic_shared"]),
        "param_size": expected_size,
        "parameters_stable_across_5_frames": row["parameters_stable_across_5_frames"] == "True",
        "param_sha256": hashlib.sha256(raw).hexdigest().upper(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--slot", required=True, type=int)
    parser.add_argument("--function", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    raw, report = extract(args.csv, args.slot, args.function)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
