#!/usr/bin/env python3
"""Validate and ingest frame-1 slot-154 output-surface before/after captures."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.process_frame_aligned_post_output_result import (
        EXPECTED_BYTES, EXPECTED_COPY_HASH, MAX_ARENA_BYTES, MIN_ARENA_BYTES,
        normalized, sha256,
    )
except ModuleNotFoundError:
    from process_frame_aligned_post_output_result import (
        EXPECTED_BYTES, EXPECTED_COPY_HASH, MAX_ARENA_BYTES, MIN_ARENA_BYTES,
        normalized, sha256,
    )


EXPECTED_REVISION = "v24_feature18_frame_aligned_post_surface_before_after"


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
            "frame1_post_surface_pre_slot154.json", "frame1_post_surface_pre_slot154.raw",
            "frame1_post_output_pre_copy.json", "frame1_post_output_pre_copy.raw",
            "post_activation_arena_prelaunch.json", "post_activation_arena_prelaunch.raw",
            "post_texture_input.raw", "copy_input.raw", "copy_output.raw",
        )
        payload = {name: zf.read(one(name)) for name in names}

    summary = json.loads(payload["summary.json"].decode("utf-8-sig"))
    before_meta = json.loads(
        payload["frame1_post_surface_pre_slot154.json"].decode("utf-8-sig"))
    after_meta = json.loads(
        payload["frame1_post_output_pre_copy.json"].decode("utf-8-sig"))
    activation_meta = json.loads(
        payload["post_activation_arena_prelaunch.json"].decode("utf-8-sig"))
    before = payload["frame1_post_surface_pre_slot154.raw"]
    after = payload["frame1_post_output_pre_copy.raw"]
    activation = payload["post_activation_arena_prelaunch.raw"]
    texture = payload["post_texture_input.raw"]
    copy_input = payload["copy_input.raw"]
    copy_output = payload["copy_output.raw"]
    before_summary = summary.get("frame1_post_surface_initial_snapshot", {})
    after_summary = summary.get("frame1_post_output_snapshot", {})
    events = []
    invalid_lines = 0
    for line in payload["module_trace.jsonl"].decode(
            "utf-8-sig", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            invalid_lines += 1
    before_arms = [e for e in events
                   if e.get("ev") == "frame1_post_surface_initial_capture_arm"
                   and e.get("frame") == 1 and e.get("status") == "PASS"]
    after_arms = [e for e in events
                  if e.get("ev") == "frame1_post_output_capture_arm"
                  and e.get("frame") == 1 and e.get("status") == "PASS"]
    common_meta = (
        before_meta.get("status") == after_meta.get("status") == "PASS"
        and before_meta.get("capture_frame") == after_meta.get("capture_frame") == 1
        and before_meta.get("width") == after_meta.get("width") == 640
        and before_meta.get("height") == after_meta.get("height") == 360
        and before_meta.get("format") == after_meta.get("format") == 10
        and str(before_meta.get("surface_resource", "")).lower()
        == str(after_meta.get("surface_resource", "")).lower()
    )
    checks = {
        "revision": summary.get("package_revision") == EXPECTED_REVISION,
        "runner": summary.get("status") == "PASS" and summary.get("host_exit") == 0,
        "feature18": summary.get("evaluates_ok", 0) >= 250
        and summary.get("feature18_created") is True
        and summary.get("feature18_evaluation_succeeded") is True,
        "metadata": common_meta
        and before_meta.get("classification") == "FRAME1_SLOT154_OUTPUT_SURFACE_INITIAL"
        and after_meta.get("classification") == "FRAME1_SLOT154_OUTPUT_BEFORE_SLOT155_COPY",
        "surface_files": len(before) == len(after) == EXPECTED_BYTES
        and before_meta.get("bytes") == after_meta.get("bytes") == EXPECTED_BYTES,
        "surface_hashes": str(before_summary.get("sha256", "")).upper() == sha256(before)
        and str(after_summary.get("sha256", "")).upper() == sha256(after),
        "trace": invalid_lines == 0 and len(before_arms) == len(after_arms) == 1,
        "activation": MIN_ARENA_BYTES <= len(activation) <= MAX_ARENA_BYTES
        and activation_meta.get("status") == "PASS"
        and activation_meta.get("frame") == 1
        and activation_meta.get("bytes") == len(activation),
        "texture": len(texture) == EXPECTED_BYTES and not any(texture),
        "final_copy": copy_input == copy_output
        and sha256(copy_input) == EXPECTED_COPY_HASH,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("result validation failed: " + ", ".join(failed))

    changed = sum(a != b for a, b in zip(before, after))
    output.mkdir(parents=True)
    for name, data in payload.items():
        (output / name).write_bytes(data)
    comparison = {
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
        "bitwise_equal": before == after,
        "changed_bytes": changed,
        "changed_fraction": changed / len(before),
    }
    (output / "surface_before_after_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n")
    receipt = {
        "schema": 1,
        "experiment": "frame1_slot154_surface_before_after_ingestion",
        "status": "PASS",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "surface_before_sha256": comparison["before_sha256"],
        "surface_after_sha256": comparison["after_sha256"],
        "surface_changed_bytes": changed,
        "surface_changed_fraction": comparison["changed_fraction"],
        "interpretation": (
            "The same frame-1 slot154 output surface is captured immediately before "
            "slot154 and immediately before slot155. This establishes whether native "
            "standalone replay must preload destination pixels."
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
        f"{datetime.now():%Y%m%d_%H%M%S}_frame_aligned_post_surface_rtx")
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
