#!/usr/bin/env python3
"""Build the immutable 156-slot inventory shipped with the one-shot RTX package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build(sequence_csv: Path, module_manifest: Path) -> dict:
    modules = json.loads(module_manifest.read_text(encoding="utf-8-sig"))["modules"]
    module_entries = {
        item["dll_offset"].upper().replace("0X", ""): set(item.get("entry_names", []))
        for item in modules
    }
    slots = []
    with sequence_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        raw = bytes.fromhex(row["frame1_param_hex"])
        aligned_u64 = [
            {"offset": offset, "value": f"0x{struct.unpack_from('<Q', raw, offset)[0]:016X}"}
            for offset in range(0, len(raw) - 7, 8)
            if struct.unpack_from("<Q", raw, offset)[0] != 0
        ]
        module_key = row["module_dll_offset"].upper().replace("0X", "")
        slots.append({
            "slot": int(row["slot"]),
            "function": row["function_name"],
            "module_dll_offset": row["module_dll_offset"],
            "module_fnv1a64": row["module_fnv1a64"],
            "grid": [int(row[f"grid_{axis}"]) for axis in "xyz"],
            "block": [int(row[f"block_{axis}"]) for axis in "xyz"],
            "dynamic_shared": int(row["dynamic_shared"]),
            "param_size": int(row["param_size"]),
            "parameters_stable_across_5_frames": row["parameters_stable_across_5_frames"] == "True",
            "frame1_param_sha256": hashlib.sha256(raw).hexdigest().upper(),
            "aligned_nonzero_u64": aligned_u64,
            "ptx_entry_present": row["function_name"] in module_entries.get(module_key, set()),
        })
    functions = Counter(item["function"] for item in slots)
    passed = (
        len(slots) == 156
        and [item["slot"] for item in slots] == list(range(156))
        and len(functions) == 43
        and all(item["ptx_entry_present"] for item in slots)
    )
    return {
        "schema": 1,
        "experiment": "full_graph_static_inventory",
        "status": "PASS" if passed else "FAIL",
        "sequence_csv_sha256": sha256(sequence_csv),
        "module_manifest_sha256": sha256(module_manifest),
        "slot_count": len(slots),
        "unique_function_count": len(functions),
        "module_count": len(modules),
        "all_slots_contiguous_0_155": [item["slot"] for item in slots] == list(range(156)),
        "all_used_ptx_entries_present": all(item["ptx_entry_present"] for item in slots),
        "function_use_counts": dict(sorted(functions.items())),
        "slots": slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_csv", type=Path)
    parser.add_argument("module_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = build(args.sequence_csv, args.module_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "slot_count", "unique_function_count", "module_count",
        "all_slots_contiguous_0_155", "all_used_ptx_entries_present")}, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
