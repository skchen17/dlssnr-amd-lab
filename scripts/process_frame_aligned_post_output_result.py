#!/usr/bin/env python3
"""Validate and ingest the RTX frame-1 slot-154 output oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


EXPECTED_REVISION = "v23_feature18_frame_aligned_post_output"
EXPECTED_BYTES = 640 * 360 * 8
EXPECTED_COPY_HASH = "CD556E0D9C2B958CF1D189412B33AA7BC52AB178369D4F5469F22E85E71FD243"
MIN_ARENA_BYTES = 16 * 1024 * 1024
MAX_ARENA_BYTES = 64 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def normalized(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return path


def process(archive: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if (not infos or len(infos) > 512 or
                sum(item.file_size for item in infos) > 4 * 1024**3):
            raise ValueError("invalid ZIP size/count")
        files = [(normalized(item.filename), item) for item in infos if not item.is_dir()]

        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for path, item in files if path.name.casefold() == name.casefold()]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        names = (
            "summary.json", "module_trace.jsonl",
            "frame1_post_output_pre_copy.json", "frame1_post_output_pre_copy.raw",
            "post_activation_arena_prelaunch.json", "post_activation_arena_prelaunch.raw",
            "post_texture_input.raw", "copy_input.raw", "copy_output.raw",
        )
        payload = {name: zf.read(one(name)) for name in names}

    summary = json.loads(payload["summary.json"].decode("utf-8-sig"))
    metadata = json.loads(payload["frame1_post_output_pre_copy.json"].decode("utf-8-sig"))
    activation_metadata = json.loads(
        payload["post_activation_arena_prelaunch.json"].decode("utf-8-sig"))
    events = []
    invalid_lines = 0
    for line in payload["module_trace.jsonl"].decode(
            "utf-8-sig", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            invalid_lines += 1
    arms = [event for event in events
            if event.get("ev") == "frame1_post_output_capture_arm"
            and event.get("frame") == 1]
    frame_output = payload["frame1_post_output_pre_copy.raw"]
    activation = payload["post_activation_arena_prelaunch.raw"]
    texture = payload["post_texture_input.raw"]
    copy_input = payload["copy_input.raw"]
    copy_output = payload["copy_output.raw"]
    output_summary = summary.get("frame1_post_output_snapshot", {})
    digest = sha256(frame_output)
    checks = {
        "revision": summary.get("package_revision") == EXPECTED_REVISION,
        "runner": summary.get("status") == "PASS" and summary.get("host_exit") == 0,
        "feature18": summary.get("evaluates_ok", 0) >= 250
        and summary.get("feature18_created") is True
        and summary.get("feature18_evaluation_succeeded") is True,
        "metadata": metadata.get("status") == "PASS"
        and metadata.get("classification") == "FRAME1_SLOT154_OUTPUT_BEFORE_SLOT155_COPY"
        and metadata.get("capture_frame") == 1
        and metadata.get("width") == 640 and metadata.get("height") == 360
        and metadata.get("format") == 10
        and metadata.get("file") == "frame1_post_output_pre_copy.raw",
        "output_size": len(frame_output) == EXPECTED_BYTES
        and metadata.get("bytes") == len(frame_output)
        and output_summary.get("bytes") == len(frame_output),
        "output_hash": str(output_summary.get("sha256", "")).upper() == digest,
        "trace": invalid_lines == 0 and len(arms) == 1
        and arms[0].get("status") == "PASS",
        "activation_control": MIN_ARENA_BYTES <= len(activation) <= MAX_ARENA_BYTES
        and activation_metadata.get("status") == "PASS"
        and activation_metadata.get("frame") == 1
        and activation_metadata.get("bytes") == len(activation),
        "texture_control": len(texture) == EXPECTED_BYTES and not any(texture),
        "final_copy_control": copy_input == copy_output
        and sha256(copy_input) == EXPECTED_COPY_HASH,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result validation failed: " + ", ".join(failed))

    output.mkdir(parents=True)
    for name, data in payload.items():
        (output / name).write_bytes(data)
    final_digest = sha256(copy_input)
    comparison = {
        "frame1_output_sha256": digest,
        "frame300_output_sha256": final_digest,
        "bitwise_equal": frame_output == copy_input,
        "differing_bytes": sum(a != b for a, b in zip(frame_output, copy_input)),
    }
    (output / "temporal_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n")
    receipt = {
        "schema": 1,
        "experiment": "frame1_slot154_output_result_ingestion",
        "status": "PASS",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "frame1_output_sha256": digest,
        "frame1_output_bytes": len(frame_output),
        "frame300_output_sha256": final_digest,
        "frame1_equals_frame300": comparison["bitwise_equal"],
        "interpretation": (
            "This is the RTX slot-154 output from the same frame as the captured "
            "full-graph parameters, activation arena, and texture input. It is the "
            "correct native/AMD replay oracle; the end-of-run copy remains a control."
        ),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / (
        f"{datetime.now():%Y%m%d_%H%M%S}_frame_aligned_post_output_rtx")
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
