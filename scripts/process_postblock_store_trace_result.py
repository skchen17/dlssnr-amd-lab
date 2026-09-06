#!/usr/bin/env python3
"""Validate a returned RTX post-block trace and compare it with RX 9070 XT."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

try:
    from scripts.analyze_postblock_store_trace import analyze_bytes, reconstruct_rgba16f
    from scripts.analyze_full_graph_integrated import compare_rgba16f_rgb
except ModuleNotFoundError:
    from analyze_postblock_store_trace import analyze_bytes, reconstruct_rgba16f
    from analyze_full_graph_integrated import compare_rgba16f_rgb


EXPECTED_TRACE_BYTES = 8_128_512
EXPECTED_RECORD_BYTES = 32
EXPECTED_THREADS_PER_SITE = 127_008
EXPECTED_LOGICAL_OUTPUT_BYTES = 1_843_200


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def normalized_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return path


def process(archive: Path, amd_trace: Path, output: Path,
            reference_output: Path | None = None) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    amd = amd_trace.read_bytes()
    if len(amd) != EXPECTED_TRACE_BYTES:
        raise ValueError("AMD trace size invalid")
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if not infos or len(infos) > 32 or sum(info.file_size for info in infos) > 64 * 1024 * 1024:
            raise ValueError("invalid ZIP size/count")
        files = [(normalized_member(info.filename), info) for info in infos if not info.is_dir()]

        def one(name: str) -> zipfile.ZipInfo:
            matches = [info for path, info in files if path.name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        manifest = json.loads(zf.read(one("manifest.json")).decode("utf-8-sig"))
        trace1 = zf.read(one("trace_run1.raw"))
        trace2 = zf.read(one("trace_run2.raw"))
        output1 = zf.read(one("output_run1.raw"))
        output2 = zf.read(one("output_run2.raw"))

    runs = manifest.get("runs")
    checks = {
        "experiment": manifest.get("experiment") == "rtx_postblock_pre_surface_store_full_grid_trace",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(runs, list) and len(runs) == 2 and all(
            isinstance(run.get("device_name"), str) and "NVIDIA" in run["device_name"].upper()
            for run in runs
        ),
        "execution": isinstance(runs, list) and len(runs) == 2 and all(
            run.get("probe_exit") == 0 and run.get("probe_pass") is True
            and run.get("execution_verified") is True for run in runs
        ),
        "launch": manifest.get("grid") == [81, 49, 1] and manifest.get("block") == [32, 1, 1],
        "layout": manifest.get("record_bytes") == EXPECTED_RECORD_BYTES
        and manifest.get("threads_per_site") == EXPECTED_THREADS_PER_SITE
        and manifest.get("trace_bytes") == EXPECTED_TRACE_BYTES,
        "trace_size": len(trace1) == EXPECTED_TRACE_BYTES and len(trace2) == EXPECTED_TRACE_BYTES,
        "trace_hash": isinstance(runs, list) and len(runs) == 2
        and str(runs[0].get("trace_sha256", "")).upper() == sha256(trace1)
        and str(runs[1].get("trace_sha256", "")).upper() == sha256(trace2),
        "repeat": manifest.get("trace_repeat_bitwise_exact") is True and trace1 == trace2,
        "nonzero": manifest.get("trace_nonzero") is True,
        "output_size": len(output1) == EXPECTED_LOGICAL_OUTPUT_BYTES
        and len(output2) == EXPECTED_LOGICAL_OUTPUT_BYTES,
        "output_hash": isinstance(runs, list) and len(runs) == 2
        and str(runs[0].get("output_sha256", "")).upper() == sha256(output1)
        and str(runs[1].get("output_sha256", "")).upper() == sha256(output2),
        "output_repeat": output1 == output2,
    }
    output_control = manifest.get("output_reference_preserved") is True
    expected_runner_status = "PASS" if all(checks.values()) and output_control else "FAIL"
    checks["runner_status_consistent"] = manifest.get("status") == expected_runner_status
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    output.mkdir(parents=True)
    rtx_path = output / "rtx_postblock_store_trace.raw"
    amd_path = output / "amd_postblock_store_trace.raw"
    rtx_path.write_bytes(trace1)
    shutil.copyfile(amd_trace, amd_path)
    (output / "rtx_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = analyze_bytes(trace1, amd)
    rtx_reconstructed, rtx_surface = reconstruct_rgba16f(trace1)
    amd_reconstructed, amd_surface = reconstruct_rgba16f(amd)
    rtx_surface["logical_output_matches_trace"] = (
        rtx_reconstructed[:EXPECTED_LOGICAL_OUTPUT_BYTES] == output1 == output2)
    if not rtx_surface["complete_surface"] or not rtx_surface["logical_output_matches_trace"]:
        raise ValueError("RTX trace does not reconstruct its surface output")
    if not amd_surface["complete_surface"]:
        raise ValueError("AMD trace does not reconstruct a complete surface")
    transformed_surface_comparison = compare_rgba16f_rgb(
        rtx_reconstructed[:EXPECTED_LOGICAL_OUTPUT_BYTES],
        amd_reconstructed[:EXPECTED_LOGICAL_OUTPUT_BYTES])
    reference_comparison = None
    if reference_output is not None:
        reference_output = reference_output.resolve(strict=True)
        reference = reference_output.read_bytes()
        if len(reference) != EXPECTED_LOGICAL_OUTPUT_BYTES:
            raise ValueError("reference output size invalid")
        expected_reference_hash = str(
            manifest.get("payload_sha256", {}).get("output_reference.raw", "")).upper()
        if expected_reference_hash and sha256(reference) != expected_reference_hash:
            raise ValueError("local reference output hash differs from package manifest")
        reference_comparison = compare_rgba16f_rgb(reference, output1)
    surface_report = {
        "rtx": rtx_surface,
        "amd": amd_surface,
        "rtx_vs_amd_transformed_rgba16f_rgb": transformed_surface_comparison,
        "rtx_transformed_vs_original_capture_rgba16f_rgb": reference_comparison,
    }
    (output / "surface_reconstruction.json").write_text(
        json.dumps(surface_report, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if comparison["pre_surface_numerical_gate"] and not output_control:
        interpretation = (
            "RTX and RX 9070 XT agree numerically immediately before both stores in the "
            "resource-lowered kernel, but the RTX transformed output does not reproduce the "
            "captured original output. The dominant unresolved path is resource semantics "
            "removed or adapted by the transform (especially texture sampling), not the "
            "post-block neural arithmetic tested by this trace."
        )
    else:
        interpretation = comparison["interpretation"]
    receipt = {
        "schema": 1,
        "experiment": "postblock_store_trace_result_ingestion",
        "status": "PASS" if output_control else "DIAGNOSTIC_VALID",
        "diagnostic_valid": True,
        "counts_as_s7": False,
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "rtx_device_name": runs[0]["device_name"],
        "runner_status": manifest.get("status"),
        "reference_output_preserved": output_control,
        "rtx_repeat_bitwise_exact": True,
        "trace_reconstructs_rtx_surface": True,
        "pre_surface_bitwise_gate": comparison["pre_surface_bitwise_gate"],
        "pre_surface_numerical_gate": comparison["pre_surface_numerical_gate"],
        "transformed_surface_numerical_gate": transformed_surface_comparison["parity_gate"]["pass"],
        "original_capture_comparison_available": reference_comparison is not None,
        "first_mismatch": comparison["first_mismatch"],
        "interpretation": interpretation,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, default=(
        repo / "results/20260904_510000_postblock_store_trace/amd_run1/trace.raw"))
    parser.add_argument("--reference-output", type=Path, default=(
        repo / "results/20260831_230000_decoder_full_graph_exact_state/cases/slot154/output0_reference.raw"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_postblock_store_trace_cross_vendor"
    try:
        receipt = process(args.archive, args.amd_trace, output, args.reference_output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
