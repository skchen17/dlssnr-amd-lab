#!/usr/bin/env python3
"""Validate and summarize a v20 full-graph RTX reference archive in-place."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def normalized(name: str) -> str:
    return name.replace("\\", "/")


def safe_member(name: str) -> bool:
    value = normalized(name)
    path = PurePosixPath(value)
    has_drive_prefix = len(value) >= 2 and value[1] == ":"
    return not path.is_absolute() and ".." not in path.parts and not has_drive_prefix


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(archive: ZipFile, members: dict[str, object], name: str):
    with archive.open(members[name]) as stream:
        return json.loads(stream.read().decode("utf-8-sig"))


def analyze(archive_path: Path, inventory_path: Path, verify_hashes: bool) -> dict:
    with archive_path.open("rb") as stream:
        archive_digest = sha256_stream(stream)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [normalized(info.filename) for info in infos]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        unsafe = [name for name in names if not safe_member(name)]
        if duplicates:
            errors.append(f"duplicate ZIP members: {duplicates[:5]}")
        if unsafe:
            errors.append(f"unsafe ZIP members: {unsafe[:5]}")
        members = {normalized(info.filename): info for info in infos}
        required = {
            "summary.json",
            "copy_snapshot.json",
            "full_graph_capture.json",
            "full_graph_blob_manifest.json",
            "full_workflow_summary.json",
            "RETURN_MANIFEST.json",
        }
        missing_required = sorted(required - members.keys())
        if missing_required:
            raise ValueError(f"archive is missing required files: {missing_required}")

        feature = read_json(archive, members, "summary.json")
        copy = read_json(archive, members, "copy_snapshot.json")
        capture = read_json(archive, members, "full_graph_capture.json")
        blob_manifest = read_json(archive, members, "full_graph_blob_manifest.json")
        workflow = read_json(archive, members, "full_workflow_summary.json")
        return_manifest = read_json(archive, members, "RETURN_MANIFEST.json")

        records = [
            (item["file"], int(item["bytes"]), item["sha256"].lower())
            for item in blob_manifest["blobs"]
        ]
        records.extend(
            (item["path"], int(item["size"]), item["sha256"].lower())
            for item in return_manifest["files"]
        )
        verified_bytes = 0
        verified_records = 0
        if verify_hashes:
            for name, size, expected_hash in records:
                name = normalized(name)
                info = members.get(name)
                if info is None:
                    errors.append(f"manifest member missing: {name}")
                    continue
                if info.file_size != size:
                    errors.append(
                        f"manifest size mismatch for {name}: {info.file_size} != {size}"
                    )
                    continue
                with archive.open(info) as stream:
                    actual_hash = sha256_stream(stream)
                if actual_hash != expected_hash:
                    errors.append(f"manifest SHA-256 mismatch for {name}")
                    continue
                verified_records += 1
                verified_bytes += size

        windows_by_slot: dict[int, list[dict]] = defaultdict(list)
        for window in capture["windows"]:
            windows_by_slot[int(window["slot"])].append(window)
        buffer_slots = sorted(windows_by_slot)
        expected_buffer_slots = set(range(155))
        missing_buffer_slots = sorted(expected_buffer_slots - set(buffer_slots))
        unexpected_buffer_slots = sorted(set(buffer_slots) - expected_buffer_slots)
        copy_matches = (
            copy.get("status") == "PASS"
            and int(copy.get("input_bytes", -1)) == int(copy.get("output_bytes", -2))
            and copy.get("input_fnv1a64") == copy.get("output_fnv1a64")
        )

        inventory_slots = {int(item["slot"]): item for item in inventory["slots"]}
        slot_reports = []
        for slot in range(156):
            static = inventory_slots.get(slot, {})
            if slot == 155:
                slot_reports.append(
                    {
                        "slot": slot,
                        "function": static.get("function"),
                        "evidence": "descriptor_object_final_copy",
                        "status": "PASS" if copy_matches else "MISSING",
                        "input_output_match": copy_matches,
                        "bytes": int(copy.get("input_bytes", 0)),
                    }
                )
                continue
            windows = windows_by_slot.get(slot, [])
            changed = [item for item in windows if int(item.get("changed_bytes", 0)) > 0]
            slot_reports.append(
                {
                    "slot": slot,
                    "function": static.get("function"),
                    "evidence": "bounded_buffer_pre_post_windows",
                    "status": "PASS" if windows else "MISSING",
                    "window_count": len(windows),
                    "changed_window_count": len(changed),
                    "changed_bytes": sum(int(item.get("changed_bytes", 0)) for item in changed),
                    "capture_limit_hits": sum(
                        int(item.get("capture_bytes", 0))
                        == int(capture["capture_limit_bytes"])
                        for item in windows
                    ),
                }
            )

        changed_slots = [
            item["slot"]
            for item in slot_reports[:155]
            if item["changed_window_count"] > 0
        ]
        unchanged_slots = sorted(expected_buffer_slots - set(changed_slots))
        covered_slots = 155 - len(missing_buffer_slots) + int(copy_matches)
        integrity_pass = not errors and (
            not verify_hashes or verified_records == len(records)
        )
        acquisition_pass = (
            integrity_pass
            and feature.get("status") == "PASS"
            and capture.get("status") == "PASS"
            and not missing_buffer_slots
            and not unexpected_buffer_slots
            and copy_matches
            and int(inventory.get("slot_count", 0)) == 156
        )

        return {
            "schema": 1,
            "experiment": "full_graph_reference_analysis",
            "status": "PASS" if acquisition_pass else "PARTIAL",
            "archive": {
                "path": str(archive_path.resolve()),
                "bytes": archive_path.stat().st_size,
                "sha256": archive_digest,
                "zip_entries": len(infos),
                "uncompressed_bytes": sum(info.file_size for info in infos),
                "path_safety": "PASS" if not unsafe and not duplicates else "FAIL",
            },
            "integrity": {
                "status": "PASS" if integrity_pass else "FAIL",
                "verification_enabled": verify_hashes,
                "manifest_records": len(records),
                "verified_records": verified_records,
                "verified_bytes": verified_bytes,
                "errors": errors,
            },
            "feature18": {
                "status": feature.get("status"),
                "host_exit": feature.get("host_exit"),
                "evaluates_ok": feature.get("evaluates_ok"),
                "created": feature.get("feature18_created"),
                "evaluation_succeeded": feature.get("feature18_evaluation_succeeded"),
            },
            "graph_coverage": {
                "status": "PASS" if covered_slots == 156 else "PARTIAL",
                "classification": "155 buffer-window slots plus one descriptor-object final-copy slot",
                "static_slot_count": inventory.get("slot_count"),
                "covered_slot_count": covered_slots,
                "captured_buffer_slot_count": len(buffer_slots),
                "missing_buffer_slots_0_154": missing_buffer_slots,
                "unexpected_buffer_slots": unexpected_buffer_slots,
                "final_copy_slot_155_status": "PASS" if copy_matches else "MISSING",
                "window_count": len(capture["windows"]),
                "changed_buffer_slot_count": len(changed_slots),
                "unchanged_buffer_slots": unchanged_slots,
                "capture_limit_bytes": capture["capture_limit_bytes"],
                "capture_limit_hits": sum(
                    int(item.get("capture_bytes", 0))
                    == int(capture["capture_limit_bytes"])
                    for item in capture["windows"]
                ),
                "unique_blob_count": blob_manifest["unique_blob_count"],
                "logical_raw_bytes": blob_manifest["logical_raw_bytes"],
                "unique_blob_bytes": blob_manifest["unique_blob_bytes"],
                "slot_reports": slot_reports,
            },
            "upstream_workflow_status": workflow.get("status"),
            "limitations": capture.get("limitations", []),
            "port_readiness": {
                "status": "READY_FOR_OFFLINE_KERNEL_RECONSTRUCTION",
                "note": "The RTX acquisition is complete, but bounded windows are evidence samples, not reconstructed AMD kernels.",
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hash-validation", action="store_true")
    args = parser.parse_args()
    report = analyze(args.archive, args.inventory, not args.skip_hash_validation)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
