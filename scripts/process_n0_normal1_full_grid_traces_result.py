#!/usr/bin/env python3
"""Validate and compare RTX/RX full-grid first Box-Muller sine-path traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


SAMPLES = 80 * 48 * 64
BYTES_PER_SAMPLE = 8
TRACE_BYTES = SAMPLES * BYTES_PER_SAMPLE


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return path


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rounded_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def predicted_half(sqrt_bits: int, sin_bits: int) -> int:
    return half_bits(rounded_f32(f32(sqrt_bits) * f32(sin_bits)))


def process(archive: Path, amd_root: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_root = amd_root.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    amd_sin_path = amd_root / "amd_sin0_trace.raw"
    amd_boundary_path = amd_root / "amd_normal1_boundary_trace.raw"
    amd_sin = amd_sin_path.read_bytes()
    amd_boundary = amd_boundary_path.read_bytes()
    if len(amd_sin) != TRACE_BYTES or len(amd_boundary) != TRACE_BYTES:
        raise ValueError("AMD trace size invalid")

    with zipfile.ZipFile(archive) as zf:
        files = [(safe_name(item.filename), item) for item in zf.infolist() if not item.is_dir()]
        if not files or len(files) > 16 or sum(item.file_size for _, item in files) > 5 * 1024 * 1024:
            raise ValueError("invalid ZIP size/count")

        def one(name: str) -> zipfile.ZipInfo:
            matches = [item for path, item in files if path.name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}")
            return matches[0]

        manifest = json.loads(zf.read(one("manifest.json")).decode("utf-8-sig"))
        rtx_sin = zf.read(one("sin0_trace.raw"))
        rtx_boundary = zf.read(one("normal1_boundary_trace.raw"))

    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError("manifest variants missing")
    by_name = {row.get("name"): row for row in variants if isinstance(row, dict)}
    expected = {
        "sin0": (["phase0_f32", "sin0_f32"], rtx_sin, "sin0_trace.raw"),
        "normal1_boundary": (
            ["sqrt0_f32", "normal1_f16_as_u32"],
            rtx_boundary,
            "normal1_boundary_trace.raw",
        ),
    }
    variant_checks = []
    for name, (fields, data, filename) in expected.items():
        row = by_name.get(name, {})
        variant_checks.append(
            row.get("fields") == fields
            and row.get("probe_exit") == 0
            and row.get("probe_pass") is True
            and row.get("execution_verified") is True
            and row.get("reference_output_preserved") is True
            and row.get("output_sha256") == manifest.get("baseline_output_sha256")
            and row.get("trace_filename") == filename
            and str(row.get("trace_sha256", "")).upper() == sha256(data)
        )
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_normal1_full_grid_traces",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str)
        and "NVIDIA" in manifest["device_name"].upper(),
        "launch": manifest.get("grid") == [80, 48, 1]
        and manifest.get("block") == [32, 1, 1],
        "shape": manifest.get("sample_count") == SAMPLES
        and manifest.get("bytes_per_sample") == BYTES_PER_SAMPLE
        and manifest.get("trace_bytes_per_variant") == TRACE_BYTES
        and len(rtx_sin) == TRACE_BYTES
        and len(rtx_boundary) == TRACE_BYTES,
        "variants": len(variants) == 2 and all(variant_checks),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    phase_mismatches = sin_mismatches = sqrt_mismatches = normal_mismatches = 0
    rtx_prediction_mismatches = amd_prediction_mismatches = 0
    rtx_sqrt_amd_sin_mismatches = amd_sqrt_rtx_sin_mismatches = 0
    categories: Counter[str] = Counter()
    first = []
    for sample in range(SAMPLES):
        rphase, rsin = struct.unpack_from("<II", rtx_sin, sample * BYTES_PER_SAMPLE)
        aphase, asin = struct.unpack_from("<II", amd_sin, sample * BYTES_PER_SAMPLE)
        rsqrt, rnormal_word = struct.unpack_from("<II", rtx_boundary, sample * BYTES_PER_SAMPLE)
        asqrt, anormal_word = struct.unpack_from("<II", amd_boundary, sample * BYTES_PER_SAMPLE)
        if rnormal_word >> 16 or anormal_word >> 16:
            raise ValueError(f"nonzero upper half at sample {sample}")
        rnormal, anormal = rnormal_word & 0xFFFF, anormal_word & 0xFFFF
        phase_equal, sin_equal, sqrt_equal = rphase == aphase, rsin == asin, rsqrt == asqrt
        phase_mismatches += not phase_equal
        sin_mismatches += not sin_equal
        sqrt_mismatches += not sqrt_equal
        rtx_prediction_mismatches += predicted_half(rsqrt, rsin) != rnormal
        amd_prediction_mismatches += predicted_half(asqrt, asin) != anormal
        rtx_sqrt_amd_sin_mismatches += predicted_half(rsqrt, asin) != rnormal
        amd_sqrt_rtx_sin_mismatches += predicted_half(asqrt, rsin) != rnormal
        if rnormal != anormal:
            normal_mismatches += 1
            categories[f"sqrt_{'equal' if sqrt_equal else 'different'}__sin_{'equal' if sin_equal else 'different'}"] += 1
            if len(first) < 64:
                linear_cta, local = divmod(sample, 64)
                cta_y, cta_x = divmod(linear_cta, 80)
                first.append({
                    "sample": sample,
                    "cta": [cta_x, cta_y, 0],
                    "local_sample": local,
                    "rtx_phase_u32": f"0x{rphase:08X}",
                    "amd_phase_u32": f"0x{aphase:08X}",
                    "rtx_sin_u32": f"0x{rsin:08X}",
                    "amd_sin_u32": f"0x{asin:08X}",
                    "rtx_sqrt_u32": f"0x{rsqrt:08X}",
                    "amd_sqrt_u32": f"0x{asqrt:08X}",
                    "rtx_normal_f16": f"0x{rnormal:04X}",
                    "amd_normal_f16": f"0x{anormal:04X}",
                })

    comparison = {
        "schema": 1,
        "experiment": "n0_normal1_full_grid_rtx_vs_rx9070xt",
        "status": "PASS",
        "classification": "FIRST_BOX_MULLER_SINE_CONSUMED_FP16_BOUNDARY",
        "sample_count": SAMPLES,
        "phase0_mismatch_samples": phase_mismatches,
        "sin0_mismatch_samples": sin_mismatches,
        "sqrt0_mismatch_samples": sqrt_mismatches,
        "normal1_f16_mismatch_samples": normal_mismatches,
        "normal1_f16_exact_fraction": 1.0 - normal_mismatches / SAMPLES,
        "normal_mismatch_categories": dict(sorted(categories.items())),
        "rtx_rounded_stage_reconstruction_mismatches": rtx_prediction_mismatches,
        "amd_rounded_stage_reconstruction_mismatches": amd_prediction_mismatches,
        "ideal_rtx_sqrt_with_amd_sin_mismatch_samples": rtx_sqrt_amd_sin_mismatches,
        "ideal_amd_sqrt_with_rtx_sin_mismatch_samples": amd_sqrt_rtx_sin_mismatches,
        "first_normal_mismatches": first,
        "rtx_sin0_trace_sha256": sha256(rtx_sin),
        "amd_sin0_trace_sha256": sha256(amd_sin),
        "rtx_boundary_trace_sha256": sha256(rtx_boundary),
        "amd_boundary_trace_sha256": sha256(amd_boundary),
    }
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "rtx_sin0_trace.raw").write_bytes(rtx_sin)
    (output / "rtx_normal1_boundary_trace.raw").write_bytes(rtx_boundary)
    shutil.copyfile(amd_sin_path, output / "amd_sin0_trace.raw")
    shutil.copyfile(amd_boundary_path, output / "amd_normal1_boundary_trace.raw")
    receipt = {
        "schema": 1,
        "experiment": "n0_normal1_full_grid_traces_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive.read_bytes()),
        "rtx_device_name": manifest["device_name"],
        "reference_outputs_preserved": True,
        **comparison,
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
        default=repo / "results/20260904_999700_n0_normal1_full_grid_traces",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = process(args.archive, args.amd_root, args.output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print(f"Result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
