#!/usr/bin/env python3
"""Validate and compare a selected CTA's clean sqrt/cos single-stage sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path


VARIANTS = ("sqrt0", "cos0")
SAMPLES = 64
BYTES_PER_VARIANT = SAMPLES * 4
TRACE_BYTES = len(VARIANTS) * BYTES_PER_VARIANT


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def process(
    archive: Path,
    amd_root: Path,
    output: Path,
    expected_target: tuple[int, int] | None = None,
) -> dict:
    archive = archive.resolve(strict=True)
    amd_root = amd_root.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    with zipfile.ZipFile(archive) as zf:
        manifests = [i for i in zf.infolist() if Path(i.filename).name == "manifest.json"]
        traces = [i for i in zf.infolist() if Path(i.filename).name == "single_stage_traces.raw"]
        if len(manifests) != 1 or len(traces) != 1 or traces[0].file_size != TRACE_BYTES:
            raise ValueError("archive layout or trace size is invalid")
        manifest = json.loads(zf.read(manifests[0]).decode("utf-8-sig"))
        combined = zf.read(traces[0])

    rows = manifest.get("variants")
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
        "experiment": manifest.get("experiment") == "rtx_n0_box_stage_single_observation_sweep",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "target": target_valid,
        "shape": manifest.get("trace_bytes_per_variant") == BYTES_PER_VARIANT,
        "combined_hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(combined),
        "variant_layout": isinstance(rows, list)
        and [row.get("name") for row in rows] == list(VARIANTS),
        "output_preserved": isinstance(rows, list)
        and all(row.get("probe_pass") is True and row.get("output_preserved") is True for row in rows),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    output.mkdir(parents=True)
    comparisons = []
    for index, name in enumerate(VARIANTS):
        rtx = combined[index * BYTES_PER_VARIANT : (index + 1) * BYTES_PER_VARIANT]
        amd_path = amd_root / f"stage_{index}_{name}_trace.raw"
        amd = amd_path.read_bytes()
        if len(amd) != BYTES_PER_VARIANT:
            raise ValueError(f"AMD {name} trace must be {BYTES_PER_VARIANT} bytes")
        if str(rows[index].get("trace_sha256", "")).upper() != sha256(rtx):
            raise ValueError(f"manifest hash mismatch for {name}")
        mismatches = []
        for sample in range(SAMPLES):
            rb = struct.unpack_from("<I", rtx, sample * 4)[0]
            ab = struct.unpack_from("<I", amd, sample * 4)[0]
            if rb != ab:
                mismatches.append({
                    "sample": sample,
                    "rtx_bits": f"0x{rb:08X}",
                    "amd_bits": f"0x{ab:08X}",
                    "rtx_value": f32(rb),
                    "amd_value": f32(ab),
                    "raw_bit_delta": rb - ab,
                })
        comparisons.append({
            "stage": name,
            "mismatch_samples": len(mismatches),
            "first_mismatches": mismatches[:16],
            "rtx_sha256": sha256(rtx),
            "amd_sha256": sha256(amd),
        })
        (output / f"rtx_{name}_trace.raw").write_bytes(rtx)
        shutil.copyfile(amd_path, output / f"amd_{name}_trace.raw")

    sqrt_mismatches = comparisons[0]["mismatch_samples"]
    cos_mismatches = comparisons[1]["mismatch_samples"]
    if sqrt_mismatches == 0 and cos_mismatches > 0:
        classification = "FIRST_CLEAN_OBSERVABLE_CTA_DIVERGENCE_AT_COS_APPROX"
    elif sqrt_mismatches == 0 and cos_mismatches == 0:
        classification = "CLEAN_SQRT_AND_COS_OBSERVATIONS_BITWISE_EXACT"
    else:
        classification = "MIXED_OR_INCONCLUSIVE_STAGE_DIVERGENCE"
    comparison = {
        "schema": 1,
        "experiment": "n0_selected_cta_clean_box_stage_rtx_vs_rx9070xt",
        "classification": classification,
        "target_cta": target_cta,
        "sample_count": SAMPLES,
        "per_stage": comparisons,
        "scope_note": "Clean rounded stage observations; does not expose hidden fused precision consumed downstream.",
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1,
        "experiment": "n0_box_stage_sweep_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "rtx_device_name": manifest["device_name"],
        "target_cta": target_cta,
        "reference_output_preserved": True,
        "classification": classification,
        "sqrt0_mismatch_samples": sqrt_mismatches,
        "cos0_mismatch_samples": cos_mismatches,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--amd-root",
        type=Path,
        default=repo / "results" / "20260902_190000_n0_cta7_1_box_stage_sweep",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-target-x", type=int)
    parser.add_argument("--expected-target-y", type=int)
    args = parser.parse_args()
    if (args.expected_target_x is None) != (args.expected_target_y is None):
        parser.error("--expected-target-x and --expected-target-y must be used together")
    expected_target = (
        None if args.expected_target_x is None
        else (args.expected_target_x, args.expected_target_y)
    )
    try:
        result = process(args.archive, args.amd_root, args.output, expected_target)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print(f"Result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
