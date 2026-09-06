#!/usr/bin/env python3
"""Validate N0 pre-block before/after raw windows and activity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fnv1a64(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"0x{value:016X}"


def analyze(metadata_path: Path, require_activity: bool = False) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    windows = []
    integrity_ok = metadata.get("status") == "PASS"
    for item in metadata.get("windows", []):
        before = (metadata_path.parent / item["before_file"]).read_bytes()
        after = (metadata_path.parent / item["after_file"]).read_bytes()
        expected = int(item["capture_bytes"])
        positions = [index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
        before_fnv = fnv1a64(before)
        after_fnv = fnv1a64(after)
        valid = (
            len(before) == expected
            and len(after) == expected
            and before_fnv.upper() == str(item["before_fnv1a64"]).upper()
            and after_fnv.upper() == str(item["after_fnv1a64"]).upper()
            and len(positions) == int(item["changed_bytes"])
            and (positions[0] if positions else -1) == int(item["first_changed_offset"])
            and (positions[-1] if positions else -1) == int(item["last_changed_offset"])
        )
        integrity_ok &= valid
        windows.append(
            {
                "name": item["name"],
                "param_offset": int(item["param_offset"]),
                "capture_bytes": expected,
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
                "before_fnv1a64": before_fnv,
                "after_fnv1a64": after_fnv,
                "changed_bytes": len(positions),
                "first_changed_offset": positions[0] if positions else -1,
                "last_changed_offset": positions[-1] if positions else -1,
                "before_nonzero_bytes": sum(value != 0 for value in before),
                "after_nonzero_bytes": sum(value != 0 for value in after),
                "metadata_integrity": valid,
            }
        )
    by_name = {item["name"]: item for item in windows}
    roles_present = all(name in by_name for name in ("scratch", "weights", "output"))
    read_only_weight_observed = roles_present and by_name["weights"]["changed_bytes"] == 0
    neural_write_activity = (
        roles_present
        and by_name["scratch"]["changed_bytes"] > 0
        and by_name["output"]["changed_bytes"] > 0
    )
    pass_condition = integrity_ok and roles_present
    if require_activity:
        pass_condition &= read_only_weight_observed and neural_write_activity
    return {
        "schema": 1,
        "experiment": "n0_preblock_capture_analysis",
        "status": "PASS" if pass_condition else "FAIL",
        "source_status": metadata.get("status"),
        "function": metadata.get("function"),
        "frame": metadata.get("frame"),
        "slot": metadata.get("slot"),
        "padded_height": metadata.get("padded_height"),
        "padded_width": metadata.get("padded_width"),
        "metadata_integrity": integrity_ok,
        "roles_present": roles_present,
        "read_only_weight_observed": read_only_weight_observed,
        "neural_write_activity": neural_write_activity,
        "activity_required": require_activity,
        "windows": windows,
        "counts_as_s6": False,
        "interpretation_limit": (
            "Write activity proves the selected RTX kernel changed the bounded "
            "windows; it does not prove an AMD neural implementation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--require-activity", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.metadata, args.require_activity)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
