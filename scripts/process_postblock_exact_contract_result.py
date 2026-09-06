#!/usr/bin/env python3
"""Validate and ingest the native RTX slot154 exact-frame contract sweep."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.analyze_full_graph_integrated import compare_rgba16f_rgb
    from scripts.process_frame_aligned_post_output_result import normalized, sha256
except ModuleNotFoundError:
    from analyze_full_graph_integrated import compare_rgba16f_rgb
    from process_frame_aligned_post_output_result import normalized, sha256


EXPECTED_REVISION = "v2_postblock_exact_frame1_contract_zero_texture_fix"
EXPECTED_STAGES = ("original", "compat", "e4m3", "movmatrix")
EXPECTED_BYTES = 640 * 360 * 8
EXPECTED_SURFACE_INITIAL_HASH = (
    "8DF6D450B5A7CB358B9E8373AF9FD9304E5912389C644F6C4BC66068380E88A3")
EXPECTED_REFERENCE_HASH = (
    "94A5E9E672FAD2E139F610472687FCA72992C0564E10004806216BCBEEBA5DF3")


def process(archive: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if not infos or len(infos) > 128 or sum(i.file_size for i in infos) > 64 * 1024**2:
            raise ValueError("invalid ZIP size/count")
        files = [(normalized(i.filename), i) for i in infos if not i.is_dir()]

        def one(name: str) -> zipfile.ZipInfo:
            matches = [i for path, i in files if path.name.casefold() == name.casefold()]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        manifest_bytes = zf.read(one("manifest.json"))
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        stage_records = manifest.get("stages", [])
        checks = {
            "revision": manifest.get("package_revision") == EXPECTED_REVISION,
            "integrity": manifest.get("payload_integrity") is True,
            "classification": manifest.get("classification")
            == "RTX_POSTBLOCK_EXACT_FRAME1_CONTRACT",
            "not_s7": manifest.get("counts_as_s7") is False,
            "surface_hash": str(manifest.get("surface_initial_sha256", "")).upper()
            == EXPECTED_SURFACE_INITIAL_HASH,
            "reference_hash": str(manifest.get("output_reference_sha256", "")).upper()
            == EXPECTED_REFERENCE_HASH,
            "zero_texture_fix": manifest.get("texture_upload_source")
            == "dedicated_zero_buffer",
            "unperturbed_reference": manifest.get("reference_capture")
            == "v23_unperturbed_post_slot154",
            "stage_order": [s.get("name") for s in stage_records] == list(EXPECTED_STAGES),
        }
        preserved = {"manifest.json": manifest_bytes}
        reports = []
        for name, stage in zip(EXPECTED_STAGES, stage_records):
            runs = stage.get("runs", [])
            run_data = []
            valid = stage.get("valid") is True and len(runs) == 2
            for index, run in enumerate(runs, 1):
                filename = f"{name}_run{index}.raw"
                data = zf.read(one(filename))
                preserved[filename] = data
                run_data.append(data)
                valid &= (
                    run.get("stage") == name and run.get("run") == index
                    and run.get("probe_exit") == 0 and run.get("probe_pass") is True
                    and run.get("execution_verified") is True
                    and run.get("surface_initial_loaded") is True
                    and "NVIDIA" in str(run.get("device_name", "")).upper()
                    and run.get("output_bytes") == len(data) == EXPECTED_BYTES
                    and str(run.get("output_sha256", "")).upper() == sha256(data)
                )
                for suffix in ("json", "log"):
                    extra = f"{name}_run{index}.{suffix}"
                    try:
                        preserved[extra] = zf.read(one(extra))
                    except ValueError:
                        valid = False
            repeat = len(run_data) == 2 and run_data[0] == run_data[1]
            reference_exact = repeat and sha256(run_data[0]) == EXPECTED_REFERENCE_HASH
            checks[f"stage_{name}"] = valid and repeat
            reports.append({
                "stage": name,
                "output_sha256": sha256(run_data[0]) if run_data else None,
                "repeat_bitwise_exact": repeat,
                "reference_exact": reference_exact,
            })
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("result validation failed: " + ", ".join(failed))

    output.mkdir(parents=True)
    for name, data in preserved.items():
        (output / name).write_bytes(data)
    exact_stages = [item["stage"] for item in reports if item["reference_exact"]]
    baseline_exact = reports[0]["reference_exact"]
    receipt = {
        "schema": 1,
        "experiment": "postblock_exact_frame1_contract_result_ingestion",
        "status": "PASS" if baseline_exact else "DIAGNOSTIC_VALID",
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "baseline_reference_exact": baseline_exact,
        "exact_stages": exact_stages,
        "stages": reports,
        "classification": (
            "EXACT_CONTRACT_BASELINE_CONFIRMED" if baseline_exact
            else "EXACT_CONTRACT_BASELINE_STILL_FAILS"),
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
        f"{datetime.now():%Y%m%d_%H%M%S}_postblock_exact_contract_rtx")
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
