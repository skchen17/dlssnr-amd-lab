#!/usr/bin/env python3
"""Validate and ingest a queue-complete, non-intrusive frame-1 RTX output."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.process_frame_aligned_post_output_result import normalized, sha256
except ModuleNotFoundError:
    from process_frame_aligned_post_output_result import normalized, sha256


EXPECTED_REVISION = "v25_feature18_deferred_frame1_output"
EXPECTED_BYTES = 640 * 360 * 8
NATIVE_ORIGINAL_HASH = (
    "8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6")


def process(archive: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if not infos or len(infos) > 256 or sum(i.file_size for i in infos) > 256 * 1024**2:
            raise ValueError("invalid ZIP size/count")
        files = [(normalized(i.filename), i) for i in infos if not i.is_dir()]

        def one(name: str) -> zipfile.ZipInfo:
            matches = [i for path, i in files if path.name.casefold() == name.casefold()]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        summary_bytes = zf.read(one("summary.json"))
        snapshot_bytes = zf.read(one("copy_snapshot.json"))
        trace_bytes = zf.read(one("module_trace.jsonl"))
        input_data = zf.read(one("copy_input.raw"))
        output_data = zf.read(one("copy_output.raw"))
        summary = json.loads(summary_bytes.decode("utf-8-sig"))
        snapshot = json.loads(snapshot_bytes.decode("utf-8-sig"))
        events = [json.loads(line) for line in trace_bytes.decode("utf-8-sig").splitlines()
                  if line.strip()]
        digest = sha256(input_data)
        deferred = summary.get("deferred_frame1_output", {})
        launches = [event for event in events
                    if event.get("ev") == "nvapi_launch_cu_kernel"
                    and event.get("frame") == 1]
        checks = {
            "revision": summary.get("package_revision") == EXPECTED_REVISION,
            "runner_status": summary.get("status") == "PASS",
            "host": summary.get("host_exit") == 0 and summary.get("evaluates_ok") == 1,
            "feature18": summary.get("feature18_created") is True
            and summary.get("feature18_evaluation_succeeded") is True,
            "deferred_mode": deferred.get("inline_snapshots_disabled") is True
            and deferred.get("bitwise_equal") is True,
            "snapshot": snapshot.get("status") == "PASS"
            and snapshot.get("width") == 640 and snapshot.get("height") == 360
            and snapshot.get("format") == 10
            and snapshot.get("input_bytes") == EXPECTED_BYTES
            and snapshot.get("output_bytes") == EXPECTED_BYTES,
            "raw": len(input_data) == len(output_data) == EXPECTED_BYTES
            and input_data == output_data,
            "summary_hashes": str(deferred.get("slot154_input_sha256", "")).upper() == digest
            and str(deferred.get("slot155_output_sha256", "")).upper() == sha256(output_data),
            "one_graph": len(launches) == 156
            and any(e.get("slot") == 154 and e.get("frame") == 1 for e in launches)
            and any(e.get("slot") == 155 and e.get("frame") == 1 for e in launches),
            "no_inline_snapshots": not any(event.get("ev") in {
                "post_texture_prelaunch_capture_arm",
                "frame1_post_output_capture_arm",
                "frame1_post_surface_initial_capture_arm",
                "post_activation_prelaunch_capture_arm",
            } for event in events),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("result validation failed: " + ", ".join(failed))

    output.mkdir(parents=True)
    for name, data in {
        "summary.json": summary_bytes,
        "copy_snapshot.json": snapshot_bytes,
        "module_trace.jsonl": trace_bytes,
        "copy_input.raw": input_data,
        "copy_output.raw": output_data,
    }.items():
        (output / name).write_bytes(data)
    native_exact = digest == NATIVE_ORIGINAL_HASH
    receipt = {
        "schema": 1,
        "experiment": "deferred_frame1_postblock_output_ingestion",
        "status": "PASS",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "output_bytes": len(input_data),
        "output_sha256": digest,
        "slot155_copy_bitwise_exact": True,
        "native_original_sha256": NATIVE_ORIGINAL_HASH,
        "native_original_bitwise_exact": native_exact,
        "classification": (
            "DEFERRED_ORACLE_MATCHES_NATIVE_ORIGINAL" if native_exact
            else "DEFERRED_ORACLE_DIFFERS_FROM_NATIVE_ORIGINAL"),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                          encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / (
        f"{datetime.now():%Y%m%d_%H%M%S}_deferred_frame1_output_rtx")
    try:
        receipt = process(args.archive, output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
