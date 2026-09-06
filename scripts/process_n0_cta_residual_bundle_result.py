#!/usr/bin/env python3
"""Validate and compare a returned N0 selected-CTA residual bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.analyze_n0_box_muller_cta_trace import analyze_bytes
    from scripts.analyze_slot3_mma_trace import analyze as analyze_mma
    from scripts.instrument_n0_box_muller_outputs_cta_trace import STAGES as BOX_STAGES
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member, sha256
except ModuleNotFoundError:
    from analyze_n0_box_muller_cta_trace import analyze_bytes
    from analyze_slot3_mma_trace import analyze as analyze_mma
    from instrument_n0_box_muller_outputs_cta_trace import STAGES as BOX_STAGES
    from process_slot3_mma_trace_result import archive_sha256, normalized_member, sha256


BOX_BYTES = 64 * len(BOX_STAGES) * 4
MMA_BYTES = 256 * 32 * 40


def process(
    archive: Path,
    amd_box: Path,
    amd_mma: Path,
    output: Path,
    expected_target: tuple[int, int] | None = None,
) -> dict:
    archive = archive.resolve(strict=True)
    amd_box = amd_box.resolve(strict=True)
    amd_mma = amd_mma.resolve(strict=True)
    if amd_box.stat().st_size != BOX_BYTES or amd_mma.stat().st_size != MMA_BYTES:
        raise ValueError("AMD trace size invalid")
    if output.exists():
        raise FileExistsError(output)

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if not infos or len(infos) > 32 or sum(item.file_size for item in infos) > 8 * 1024 * 1024:
            raise ValueError("invalid ZIP size/count")
        files = [(normalized_member(item.filename), item) for item in infos if not item.is_dir()]
        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for path, item in files if path.name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]
        manifest = json.loads(zf.read(one("manifest.json")).decode("utf-8-sig"))
        box = zf.read(one("box_outputs_trace.raw"))
        mma = zf.read(one("mma_trace.raw"))

    cases = manifest.get("cases")
    case_map = {item.get("name"): item for item in cases} if isinstance(cases, list) else {}
    box_case = case_map.get("box_outputs", {})
    mma_case = case_map.get("mma", {})
    target_cta = manifest.get("target_cta")
    target_valid = (
        isinstance(target_cta, list)
        and len(target_cta) == 3
        and all(isinstance(value, int) for value in target_cta)
        and 0 <= target_cta[0] < 80
        and 0 <= target_cta[1] < 48
        and target_cta[2] == 0
    )
    if expected_target is not None:
        target_valid = target_valid and target_cta[:2] == list(expected_target)
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_selected_cta_residual_bundle",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str) and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1] and manifest.get("block") == [32, 1, 1]
        and target_valid,
        "baseline": manifest.get("baseline_probe_exit") == 0 and manifest.get("baseline_probe_pass") is True,
        "preservation": manifest.get("reference_outputs_preserved") is True,
        "case_layout": set(case_map) == {"box_outputs", "mma"},
        "case_runs": all(item.get("probe_exit") == 0 and item.get("probe_pass") is True
                         and item.get("execution_verified") is True for item in case_map.values()),
        "box": len(box) == BOX_BYTES and box_case.get("trace_bytes") == BOX_BYTES
        and str(box_case.get("trace_sha256", "")).upper() == sha256(box)
        and box_case.get("trace_nonzero_bytes") == sum(value != 0 for value in box),
        "mma": len(mma) == MMA_BYTES and mma_case.get("trace_bytes") == MMA_BYTES
        and str(mma_case.get("trace_sha256", "")).upper() == sha256(mma)
        and mma_case.get("trace_nonzero_bytes") == sum(value != 0 for value in mma),
        "output_hashes": box_case.get("output_sha256") == manifest.get("baseline_output_sha256")
        and mma_case.get("output_sha256") == manifest.get("baseline_output_sha256"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    amd_box_bytes = amd_box.read_bytes()
    output.mkdir(parents=True)
    rtx_box_path = output / "rtx_box_outputs_trace.raw"
    rtx_mma_path = output / "rtx_mma_trace.raw"
    amd_box_path = output / "amd_box_outputs_trace.raw"
    amd_mma_path = output / "amd_mma_trace.raw"
    rtx_box_path.write_bytes(box)
    rtx_mma_path.write_bytes(mma)
    shutil.copyfile(amd_box, amd_box_path)
    shutil.copyfile(amd_mma, amd_mma_path)
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    box_comparison = analyze_bytes(box, amd_box_bytes, stages=BOX_STAGES)
    box_comparison["experiment"] = "n0_selected_cta_box_outputs_rtx_vs_rx9070xt"
    box_comparison["target_cta"] = target_cta
    box_comparison["bitwise_gate"] = box_comparison["first_mismatch_stage"] is None
    (output / "box_comparison.json").write_text(json.dumps(box_comparison, indent=2) + "\n", encoding="utf-8")

    mma_comparison = analyze_mma(rtx_mma_path, amd_mma_path, include_models=True)
    mma_comparison["experiment"] = "n0_selected_cta_internal_fp8_mma_rtx_vs_rx9070xt"
    mma_comparison["target_cta"] = target_cta
    mma_comparison["reference_output_preserved"] = True
    (output / "mma_comparison.json").write_text(json.dumps(mma_comparison, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "schema": 1,
        "experiment": "n0_selected_cta_residual_bundle_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"],
        "target_cta": target_cta,
        "reference_outputs_preserved": True,
        "box_outputs_bitwise_gate": box_comparison["bitwise_gate"],
        "first_box_output_mismatch_stage": box_comparison["first_mismatch_stage"],
        "first_pre_fragment_mismatch_mma": mma_comparison["first_pre_fragment_mismatch_mma"],
        "first_d_mismatch_mma": mma_comparison["first_d_mismatch_mma"],
        "pre_fragment_byte_mismatches": mma_comparison["pre_fragment_byte_mismatches"],
        "d_half_mismatches": mma_comparison["d_half_mismatches"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-box", type=Path, required=True)
    parser.add_argument("--amd-mma", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-target-x", type=int)
    parser.add_argument("--expected-target-y", type=int)
    args = parser.parse_args()
    if (args.expected_target_x is None) != (args.expected_target_y is None):
        parser.error("--expected-target-x and --expected-target-y must be used together")
    expected_target = (
        None if args.expected_target_x is None
        else (args.expected_target_x, args.expected_target_y)
    )
    target_tag = (
        "selected" if expected_target is None
        else f"cta{expected_target[0]}_{expected_target[1]}"
    )
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_n0_{target_tag}_residual_cross_vendor"
    try:
        receipt = process(args.archive, args.amd_box, args.amd_mma, output, expected_target)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
