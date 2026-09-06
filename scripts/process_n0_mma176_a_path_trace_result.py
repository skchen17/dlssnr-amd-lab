#!/usr/bin/env python3
"""Validate and compare a returned selected-CTA MMA-176 A-path trace."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.analyze_n0_norm_path_trace import analyze, sha256
    from scripts.instrument_n0_mma176_a_path_trace import STAGES, TRACE_BYTES
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from analyze_n0_norm_path_trace import analyze, sha256
    from instrument_n0_mma176_a_path_trace import STAGES, TRACE_BYTES
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


def process(archive: Path, amd_trace: Path, output: Path, target_cta: list[int]) -> dict:
    archive = archive.resolve(strict=True); amd_trace = amd_trace.resolve(strict=True)
    if amd_trace.stat().st_size != TRACE_BYTES: raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists(): raise FileExistsError(output)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if not infos or len(infos) > 32 or sum(item.file_size for item in infos) > 8 * 1024 * 1024: raise ValueError("invalid ZIP size/count")
        files = [(normalized_member(item.filename), item) for item in infos if not item.is_dir()]
        manifests = [item for path, item in files if path.name == "manifest.json"]
        traces = [item for path, item in files if path.name == "mma176_a_path_trace.raw"]
        if len(manifests) != 1 or len(traces) != 1 or traces[0].file_size != TRACE_BYTES: raise ValueError("archive trace layout invalid")
        manifest = json.loads(zf.read(manifests[0]).decode("utf-8-sig")); rtx = zf.read(traces[0])
    expected = [{"index": i, "name": name, "register": register} for i, (name, _, register) in enumerate(STAGES)]
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_selected_cta_mma176_a_path_trace",
        "status_control": manifest.get("status") == ("PASS" if manifest.get("reference_output_preserved") is True else "FAIL"),
        "integrity": manifest.get("payload_integrity") is True,
        "probe": manifest.get("probe_exit") == 0 and manifest.get("probe_pass") is True,
        "baseline": manifest.get("baseline_probe_exit") == 0 and manifest.get("baseline_probe_pass") is True,
        "device": isinstance(manifest.get("device_name"), str) and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80,48,1] and manifest.get("block") == [32,1,1] and manifest.get("target_cta") == target_cta,
        "stages": manifest.get("stage_count") == len(STAGES) and manifest.get("stages") == expected,
        "bytes": manifest.get("checkpoint_bytes") == TRACE_BYTES,
        "hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(rtx),
        "perturbation_control": isinstance(manifest.get("reference_output_preserved"), bool)
        and manifest.get("instrumentation_perturbed") is (not manifest.get("reference_output_preserved")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed: raise ValueError("manifest validation failed: " + ", ".join(failed))
    output.mkdir(parents=True); rtx_path = output / "rtx_mma176_a_path_trace.raw"; amd_path = output / "amd_mma176_a_path_trace.raw"
    rtx_path.write_bytes(rtx); shutil.copyfile(amd_trace, amd_path)
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = analyze(rtx_path, amd_path, stages_layout=STAGES, trace_bytes=TRACE_BYTES)
    comparison["experiment"] = "n0_selected_cta_mma176_a_path_rtx_vs_rx9070xt"; comparison["target_cta"] = target_cta
    comparison["admissible_as_clean_oracle"] = manifest["reference_output_preserved"]
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt_status = "PASS" if manifest["reference_output_preserved"] else "REJECTED_PERTURBED"
    receipt = {"schema":1,"experiment":"n0_mma176_a_path_result_ingestion","status":receipt_status,"source_archive":str(archive),
        "source_archive_sha256":archive_sha256(archive),"rtx_device_name":manifest["device_name"],"target_cta":target_cta,
        "reference_output_preserved":manifest["reference_output_preserved"],
        "diagnostic_first_divergent_stage":comparison["first_divergent_stage"],
        "diagnostic_first_divergent_stage_name":comparison["first_divergent_stage_name"]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8"); return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]; parser = argparse.ArgumentParser(); parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, required=True); parser.add_argument("--output", type=Path)
    parser.add_argument("--target-x", type=int, default=35); parser.add_argument("--target-y", type=int, default=0); args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_n0_mma176_a_path_cross_vendor"
    try: receipt = process(args.archive, args.amd_trace, output, [args.target_x,args.target_y,0])
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error: print(f"ERROR: {error}", file=sys.stderr); return 2
    print(json.dumps(receipt, indent=2)); print(f"Result: {output.resolve()}"); return 0


if __name__ == "__main__": raise SystemExit(main())
