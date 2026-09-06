#!/usr/bin/env python3
"""Validate and ingest the returned RTX slot-154 native-resource stage sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

try:
    from scripts.analyze_full_graph_integrated import compare_rgba16f_rgb
except ModuleNotFoundError:
    from analyze_full_graph_integrated import compare_rgba16f_rgb


EXPECTED_EXPERIMENT = "rtx_postblock_native_resource_lowering_stage_sweep"
EXPECTED_STAGES = (
    "original", "compat", "e4m3", "movmatrix", "fp8_mma", "f16_mma",
    "decoder_compat",
)
EXPECTED_OUTPUT_BYTES = 640 * 360 * 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def normalized_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return path


def process(archive: Path, output: Path, reference_path: Path | None = None) -> dict:
    archive = archive.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[1]
    if reference_path is None:
        reference_path = repo / (
            "results/20260831_230000_decoder_full_graph_exact_state/"
            "cases/slot154/output0_reference.raw"
        )
    reference = reference_path.resolve(strict=True).read_bytes()

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if (not infos or len(infos) > 256 or
                sum(item.file_size for item in infos) > 512 * 1024 * 1024):
            raise ValueError("invalid ZIP size/count")
        members: dict[str, zipfile.ZipInfo] = {}
        for item in infos:
            if item.is_dir():
                continue
            path = normalized_member(item.filename)
            key = path.as_posix().casefold()
            if key in members:
                raise ValueError(f"duplicate ZIP member: {item.filename}")
            members[key] = item

        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for key, item in members.items()
                       if PurePosixPath(key).name == name.casefold()]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        manifest_bytes = zf.read(one("manifest.json"))
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        stage_records = manifest.get("stages", [])
        checks = {
            "experiment": manifest.get("experiment") == EXPECTED_EXPERIMENT,
            "runner_status": manifest.get("status") in {"PASS", "FAIL"},
            "classification": manifest.get("classification")
            == "RTX_POSTBLOCK_FIRST_BREAKING_LOWERING_STAGE",
            "not_s7": manifest.get("counts_as_s7") is False,
            "payload_integrity": manifest.get("payload_integrity") is True,
            "layout": manifest.get("grid") == [81, 49, 1]
            and manifest.get("block") == [32, 1, 1],
            "stage_order": [stage.get("name") for stage in stage_records]
            == list(EXPECTED_STAGES)
            and [stage.get("order") for stage in stage_records] == list(range(7)),
            "reference_size": len(reference) == EXPECTED_OUTPUT_BYTES,
            "reference_hash": str(manifest.get("output_reference_sha256", "")).upper()
            == sha256(reference),
        }

        raw_by_stage: dict[str, bytes] = {}
        preserved: dict[str, bytes] = {"manifest.json": manifest_bytes}
        stage_reports = []
        for index, name in enumerate(EXPECTED_STAGES):
            if index >= len(stage_records):
                continue
            stage = stage_records[index]
            runs = stage.get("runs", [])
            declared_valid = stage.get("valid") is True
            stage_ok = stage.get("repeat_bitwise_exact") is True and len(runs) == 2
            run_data = []
            for run_index, run in enumerate(runs, 1):
                expected_name = f"{name}_run{run_index}.raw"
                try:
                    data = zf.read(one(expected_name))
                except ValueError:
                    stage_ok = False
                    continue
                preserved[expected_name] = data
                run_data.append(data)
                common_ok = (
                    run.get("stage") == name
                    and run.get("run") == run_index
                    and "NVIDIA" in str(run.get("device_name", "")).upper()
                    and run.get("output_file") == expected_name
                    and str(run.get("output_sha256", "")).upper() == sha256(data)
                )
                if declared_valid:
                    stage_ok = stage_ok and common_ok and (
                        run.get("probe_exit") == 0
                        and run.get("probe_pass") is True
                        and run.get("execution_verified") is True
                        and run.get("output_bytes") == EXPECTED_OUTPUT_BYTES
                        and len(data) == EXPECTED_OUTPUT_BYTES
                    )
                else:
                    probe_name = f"{name}_run{run_index}.json"
                    try:
                        probe_bytes = zf.read(one(probe_name))
                        probe = json.loads(probe_bytes.decode("utf-8-sig"))
                        preserved[probe_name] = probe_bytes
                    except (ValueError, json.JSONDecodeError):
                        probe = {}
                    steps = probe.get("steps", [])
                    invalid_ptx = any(
                        step.get("name") == "cuModuleLoadData"
                        and step.get("code") == 218 for step in steps)
                    stage_ok = stage_ok and common_ok and (
                        run.get("probe_exit") != 0
                        and run.get("probe_pass") is False
                        and run.get("execution_verified") is False
                        and run.get("output_bytes") == 0
                        and len(data) == 0
                        and probe.get("module_loaded") is False
                        and invalid_ptx
                    )
            if len(run_data) == 2:
                stage_ok = stage_ok and run_data[0] == run_data[1]
                if declared_valid:
                    raw_by_stage[name] = run_data[0]
            checks[f"stage_{name}"] = stage_ok

        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("result validation failed: " + ", ".join(failed))
        if "original" not in raw_by_stage:
            raise ValueError("original native-resource replay did not execute")

        first_breaking_stage = None
        first_stage_different_from_original = None
        original = raw_by_stage["original"]
        for name in EXPECTED_STAGES:
            data = raw_by_stage.get(name)
            if data is None:
                stage_reports.append({
                    "stage": name,
                    "execution": "CUDA_ERROR_INVALID_PTX",
                    "reference_exact": False,
                })
                continue
            comparison = compare_rgba16f_rgb(reference, data)
            exact = data == reference
            stage_reports.append({
                "stage": name,
                "output_sha256": sha256(data),
                "reference_exact": exact,
                "comparison": comparison,
            })
            if first_breaking_stage is None and not exact:
                first_breaking_stage = name
            if (first_stage_different_from_original is None and name != "original"
                    and data != original):
                first_stage_different_from_original = name

        # Preserve small probe JSON/logs when present, but raw files and the
        # validated manifest remain the normative evidence.
        for item in infos:
            if item.is_dir():
                continue
            path = normalized_member(item.filename)
            if path.suffix.lower() in {".json", ".log"} and path.name != "manifest.json":
                preserved[path.name] = zf.read(item)

    output.mkdir(parents=True)
    for name, data in preserved.items():
        (output / name).write_bytes(data)
    comparison_record = {
        "schema": 1,
        "reference_sha256": sha256(reference),
        "first_breaking_stage": first_breaking_stage,
        "first_stage_different_from_original": first_stage_different_from_original,
        "all_stages_reference_exact": first_breaking_stage is None,
        "stages": stage_reports,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison_record, indent=2) + "\n", encoding="utf-8")
    baseline_exact = original == reference
    failed_execution_stages = [
        name for name in EXPECTED_STAGES if name not in raw_by_stage]
    diagnostic_only = not baseline_exact or bool(failed_execution_stages)
    receipt = {
        "schema": 1,
        "experiment": "postblock_native_stage_sweep_result_ingestion",
        "status": "DIAGNOSTIC_VALID" if diagnostic_only else "PASS",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "runner_status": manifest.get("status"),
        "baseline_reference_exact": baseline_exact,
        "baseline_output_sha256": sha256(original),
        "first_breaking_stage": first_breaking_stage,
        "first_stage_different_from_original": first_stage_different_from_original,
        "failed_execution_stages": failed_execution_stages,
        "classification": (
            "STANDALONE_BASELINE_CONTRACT_FAIL"
            if not baseline_exact else
            "LOWERING_STAGE_INVALID_PTX"
            if failed_execution_stages else
            "FIRST_BREAKING_LOWERING_STAGE_IDENTIFIED"
        ),
        "interpretation": (
            "A deterministic but non-exact original stage means the standalone input/"
            "resource contract is incomplete, so no lowering stage can yet be blamed. "
            "Invalid-PTX stages are recorded separately from numerical divergence."
            if not baseline_exact else
            "The original RTX native-resource replay is exact. The first cumulative "
            "lowering stage whose output changes identifies the correction target."
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
        f"{datetime.now():%Y%m%d_%H%M%S}_postblock_native_stage_sweep_rtx")
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
