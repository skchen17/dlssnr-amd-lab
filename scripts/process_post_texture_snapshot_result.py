#!/usr/bin/env python3
"""Validate and ingest the slot-154 texture snapshot from a returned RTX run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


EXPECTED_REVISION = "v21_feature18_slot154_prelaunch_texture_snapshot"
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 360
EXPECTED_FORMAT = 10  # DXGI_FORMAT_R16G16B16A16_FLOAT
EXPECTED_BYTES = EXPECTED_WIDTH * EXPECTED_HEIGHT * 8
EXPECTED_COPY_HASH = "CD556E0D9C2B958CF1D189412B33AA7BC52AB178369D4F5469F22E85E71FD243"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def normalized_member(name: str) -> PurePosixPath:
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
                sum(item.file_size for item in infos) > 4 * 1024 * 1024 * 1024):
            raise ValueError("invalid ZIP size/count")
        files = [(normalized_member(item.filename), item)
                 for item in infos if not item.is_dir()]

        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for path, item in files if path.name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        summary_bytes = zf.read(one("summary.json"))
        snapshot_bytes = zf.read(one("post_texture_snapshot.json"))
        trace_bytes = zf.read(one("module_trace.jsonl"))
        texture = zf.read(one("post_texture_input.raw"))
        copy_input = zf.read(one("copy_input.raw"))
        copy_output = zf.read(one("copy_output.raw"))

    summary = json.loads(summary_bytes.decode("utf-8-sig"))
    snapshot = json.loads(snapshot_bytes.decode("utf-8-sig"))
    summary_texture = summary.get("post_texture_snapshot", {})
    resource_trace = summary.get("d3d12_resource_trace", {})
    digest = sha256(texture)
    nonzero_bytes = sum(value != 0 for value in texture)
    texture_all_zero = nonzero_bytes == 0
    trace_events = []
    for line in trace_bytes.decode("utf-8-sig", errors="replace").splitlines():
        try:
            trace_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    resource = str(snapshot.get("texture_resource", "")).lower()
    mapping_events = {
        "resource_create": any(
            event.get("ev") == "d3d12_resource_create"
            and str(event.get("resource", "")).lower() == resource
            and event.get("width") == EXPECTED_WIDTH
            and event.get("height") == EXPECTED_HEIGHT
            and event.get("format") == EXPECTED_FORMAT
            for event in trace_events),
        "srv_create": any(
            event.get("ev") == "d3d12_create_srv"
            and str(event.get("resource", "")).lower() == resource
            for event in trace_events),
        "merged_handle": any(
            event.get("ev") == "nvapi_get_cuda_merged_texture_sampler_call"
            and str(event.get("descriptor_resource", "")).lower() == resource
            for event in trace_events),
        "prelaunch_arm": any(
            event.get("ev") == "post_texture_prelaunch_capture_arm"
            and event.get("frame") == 1
            and event.get("status") == "PASS"
            and str(event.get("texture_resource", "")).lower() == resource
            for event in trace_events),
        "slot154_bind": any(
            event.get("ev") == "post_texture_resource_bind"
            and event.get("frame") == 1
            and event.get("slot") == 154
            and event.get("prelaunch_capture") is True
            and str(event.get("texture_resource", "")).lower() == resource
            for event in trace_events),
    }
    copy_summary = summary.get("copy_content_snapshot", {})
    checks = {
        "revision": summary.get("package_revision") == EXPECTED_REVISION,
        # v21 intentionally marked an otherwise valid all-zero capture FAIL.
        # Once the pre-launch timing and descriptor mapping are proven, zero is
        # a legitimate observation that falsifies the old content hypothesis.
        "runner_status": summary.get("status") == ("FAIL" if texture_all_zero else "PASS"),
        "host": summary.get("host_exit") == 0,
        "feature18": summary.get("evaluates_ok", 0) >= 250
        and summary.get("feature18_created") is True
        and summary.get("feature18_evaluation_succeeded") is True,
        "resource_binding": resource_trace.get("post_texture_resource_binds", 0) >= 1,
        "snapshot_status": snapshot.get("status") == "PASS"
        and snapshot.get("classification") == "SLOT154_PRELAUNCH_TEXTURE_INPUT"
        and snapshot.get("capture_frame") == 1
        and summary_texture.get("status") == "PASS",
        "layout": snapshot.get("width") == EXPECTED_WIDTH
        and snapshot.get("height") == EXPECTED_HEIGHT
        and snapshot.get("format") == EXPECTED_FORMAT
        and snapshot.get("row_size") == EXPECTED_WIDTH * 8
        and snapshot.get("num_rows") == EXPECTED_HEIGHT
        and snapshot.get("bytes") == EXPECTED_BYTES
        and summary_texture.get("width") == EXPECTED_WIDTH
        and summary_texture.get("height") == EXPECTED_HEIGHT
        and summary_texture.get("format") == EXPECTED_FORMAT
        and summary_texture.get("bytes") == EXPECTED_BYTES,
        "raw_size": len(texture) == EXPECTED_BYTES,
        "raw_hash": str(summary_texture.get("sha256", "")).upper() == digest,
        "raw_content_classified": summary_texture.get("nonzero_bytes") == nonzero_bytes,
        "raw_name": snapshot.get("file") == "post_texture_input.raw",
        "non_null_resource": str(snapshot.get("texture_object", "0x0"))
        not in {"0", "0x0", "0x0000000000000000"}
        and str(snapshot.get("texture_resource", "0x0"))
        not in {"0", "0x0", "0x0000000000000000"},
        "descriptor_mapping": all(mapping_events.values()),
        "copy_output_control": len(copy_input) == EXPECTED_BYTES
        and len(copy_output) == EXPECTED_BYTES
        and copy_input == copy_output
        and sha256(copy_input) == EXPECTED_COPY_HASH
        and str(copy_summary.get("input_sha256", "")).upper() == EXPECTED_COPY_HASH
        and str(copy_summary.get("output_sha256", "")).upper() == EXPECTED_COPY_HASH,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result validation failed: " + ", ".join(failed))

    output.mkdir(parents=True)
    (output / "summary.json").write_bytes(summary_bytes)
    (output / "post_texture_snapshot.json").write_bytes(snapshot_bytes)
    (output / "module_trace.jsonl").write_bytes(trace_bytes)
    (output / "post_texture_input.raw").write_bytes(texture)
    (output / "copy_input.raw").write_bytes(copy_input)
    (output / "copy_output.raw").write_bytes(copy_output)
    receipt = {
        "schema": 1,
        "experiment": "slot154_texture_snapshot_result_ingestion",
        "status": "PASS",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "package_revision": EXPECTED_REVISION,
        "evaluates_ok": summary["evaluates_ok"],
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "format": EXPECTED_FORMAT,
        "bytes": EXPECTED_BYTES,
        "texture_sha256": digest,
        "texture_nonzero_bytes": nonzero_bytes,
        "texture_all_zero": texture_all_zero,
        "descriptor_mapping_checks": mapping_events,
        "copy_output_sha256": sha256(copy_output),
        "classification": (
            "VALID_ZERO_TEXTURE_REJECTS_RESOURCE_CONTENT_HYPOTHESIS"
            if texture_all_zero else "VALID_NONZERO_TEXTURE_CAPTURE"
        ),
        "interpretation": (
            "The real non-null slot-154 Texture2D resource was captured immediately "
            "before launch and its descriptor-to-resource mapping was verified. "
            "An all-zero result is therefore valid and rejects missing texture "
            "content as the cause of the translated output mismatch."
        ),
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / (
        f"{datetime.now():%Y%m%d_%H%M%S}_post_texture_snapshot_rtx")
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
