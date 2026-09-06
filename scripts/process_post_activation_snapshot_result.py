#!/usr/bin/env python3
"""Validate and ingest the RTX slot-154 full pre-launch activation snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


EXPECTED_REVISION = "v22_feature18_slot154_full_activation_snapshot"
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


def process(archive: Path, output: Path, reconstructed_path: Path | None = None) -> dict:
    archive = archive.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[1]
    if reconstructed_path is None:
        reconstructed_path = repo / (
            "results/20260831_230000_decoder_full_graph_exact_state/"
            "cases/slot154/activation_arena_initial.raw")

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

        summary_bytes = zf.read(one("summary.json"))
        metadata_bytes = zf.read(one("post_activation_arena_prelaunch.json"))
        trace_bytes = zf.read(one("module_trace.jsonl"))
        arena = zf.read(one("post_activation_arena_prelaunch.raw"))
        texture = zf.read(one("post_texture_input.raw"))
        copy_input = zf.read(one("copy_input.raw"))
        copy_output = zf.read(one("copy_output.raw"))

    summary = json.loads(summary_bytes.decode("utf-8-sig"))
    metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
    events = []
    for line in trace_bytes.decode("utf-8-sig", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    arms = [event for event in events
            if event.get("ev") == "post_activation_prelaunch_capture_arm"
            and event.get("frame") == 1]
    arm = arms[0] if len(arms) == 1 else {}
    arena_summary = summary.get("post_activation_snapshot", {})
    digest = sha256(arena)
    checks = {
        "revision": summary.get("package_revision") == EXPECTED_REVISION,
        "runner": summary.get("status") == "PASS" and summary.get("host_exit") == 0,
        "feature18": summary.get("evaluates_ok", 0) >= 250
        and summary.get("feature18_created") is True
        and summary.get("feature18_evaluation_succeeded") is True,
        "metadata": metadata.get("status") == "PASS"
        and metadata.get("classification") == "SLOT154_PRELAUNCH_FULL_ACTIVATION_BUFFER"
        and metadata.get("frame") == 1
        and metadata.get("file") == "post_activation_arena_prelaunch.raw",
        "size": MIN_ARENA_BYTES <= len(arena) <= MAX_ARENA_BYTES
        and metadata.get("bytes") == len(arena)
        and arena_summary.get("bytes") == len(arena),
        "hash": str(arena_summary.get("sha256", "")).upper() == digest,
        "content": any(arena),
        "trace": arm.get("status") == "PASS"
        and arm.get("bytes") == len(arena)
        and arm.get("offset0") == 13_873_152
        and arm.get("offset8") == 110_592
        and str(arm.get("resource", "")).lower()
        == str(metadata.get("resource", "")).lower(),
        "copy_control": copy_input == copy_output
        and sha256(copy_input) == EXPECTED_COPY_HASH,
        "zero_texture_control": len(texture) == 640 * 360 * 8 and not any(texture),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result validation failed: " + ", ".join(failed))

    reconstructed = reconstructed_path.resolve(strict=True).read_bytes()
    overlap = min(len(arena), len(reconstructed))
    differences = sum(a != b for a, b in zip(arena[:overlap], reconstructed[:overlap]))
    first_difference = next(
        (index for index, (a, b) in enumerate(zip(arena[:overlap], reconstructed[:overlap]))
         if a != b), None)
    comparison = {
        "captured_bytes": len(arena),
        "reconstructed_bytes": len(reconstructed),
        "overlap_bytes": overlap,
        "differing_overlap_bytes": differences,
        "first_difference": first_difference,
        "captured_sha256": digest,
        "reconstructed_sha256": sha256(reconstructed),
        "reconstruction_exact": len(arena) == len(reconstructed) and arena == reconstructed,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_bytes(summary_bytes)
    (output / "post_activation_arena_prelaunch.json").write_bytes(metadata_bytes)
    (output / "post_activation_arena_prelaunch.raw").write_bytes(arena)
    (output / "module_trace.jsonl").write_bytes(trace_bytes)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    receipt = {
        "schema": 1,
        "experiment": "slot154_full_activation_snapshot_result_ingestion",
        "status": "PASS",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "activation_sha256": digest,
        "activation_bytes": len(arena),
        "reconstruction_exact": comparison["reconstruction_exact"],
        "differing_overlap_bytes": differences,
        "interpretation": (
            "This is the complete live RTX activation-buffer state immediately before "
            "slot 154. It replaces the bounded-window reconstruction for native replay."
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
        f"{datetime.now():%Y%m%d_%H%M%S}_post_activation_snapshot_rtx")
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
