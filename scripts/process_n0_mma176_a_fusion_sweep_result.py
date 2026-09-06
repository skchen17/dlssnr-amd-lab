#!/usr/bin/env python3
"""Validate an RTX MMA-176 A-path fusion sweep and compare clean stages to RX."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.instrument_n0_mma176_a_path_trace import STAGES, TRACE_BYTES
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from instrument_n0_mma176_a_path_trace import STAGES, TRACE_BYTES
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


LANES = 32
STAGE_BYTES = LANES * 4


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def compare_stage(rtx: bytes, amd: bytes, index: int) -> dict:
    base = index * STAGE_BYTES
    mismatches = []
    byte_mismatches = 0
    half_mismatches = 0
    for lane in range(LANES):
        offset = base + lane * 4
        left = rtx[offset:offset + 4]
        right = amd[offset:offset + 4]
        byte_mismatches += sum(a != b for a, b in zip(left, right))
        if left != right:
            lh = struct.unpack("<2H", left)
            rh = struct.unpack("<2H", right)
            half_mismatches += sum(a != b for a, b in zip(lh, rh))
            mismatches.append({
                "lane": lane,
                "rtx_u32_hex": f"{struct.unpack('<I', left)[0]:08X}",
                "amd_u32_hex": f"{struct.unpack('<I', right)[0]:08X}",
                "rtx_half_hex": [f"{value:04X}" for value in lh],
                "amd_half_hex": [f"{value:04X}" for value in rh],
            })
    return {
        "bitwise_gate": not mismatches,
        "byte_mismatches": byte_mismatches,
        "word_mismatches": len(mismatches),
        "half_mismatches": half_mismatches,
        "mismatches": mismatches,
    }


def process(archive: Path, amd_trace: Path, output: Path) -> dict:
    archive = archive.resolve(strict=True)
    amd_trace = amd_trace.resolve(strict=True)
    amd = amd_trace.read_bytes()
    if len(amd) != TRACE_BYTES:
        raise ValueError(f"AMD trace must be {TRACE_BYTES} bytes")
    if output.exists():
        raise FileExistsError(output)

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if not infos or len(infos) > 32 or sum(item.file_size for item in infos) > 8 * 1024 * 1024:
            raise ValueError("invalid ZIP size/count")
        files = [(normalized_member(item.filename), item) for item in infos if not item.is_dir()]
        manifests = [item for path, item in files if path.name == "manifest.json"]
        traces = [item for path, item in files if path.name == "single_stage_traces.raw"]
        if len(manifests) != 1 or len(traces) != 1 or traces[0].file_size != TRACE_BYTES:
            raise ValueError("archive sweep layout invalid")
        manifest = json.loads(zf.read(manifests[0]).decode("utf-8-sig"))
        rtx = zf.read(traces[0])

    expected = [(index, name, register) for index, (name, _, register) in enumerate(STAGES)]
    variants = manifest.get("variants")
    variant_layout = [] if not isinstance(variants, list) else [
        (item.get("index"), item.get("name"), item.get("register")) for item in variants
    ]
    hash_layout = [] if not isinstance(variants, list) else [
        str(item.get("trace_sha256", "")).upper() == sha256(rtx[index * STAGE_BYTES:(index + 1) * STAGE_BYTES])
        for index, item in enumerate(variants)
    ]
    checks = {
        "experiment": manifest.get("experiment") == "rtx_n0_selected_cta_mma176_a_fusion_sweep",
        "status": manifest.get("status") == "PASS",
        "integrity": manifest.get("payload_integrity") is True,
        "device": isinstance(manifest.get("device_name"), str) and "NVIDIA" in manifest["device_name"].upper(),
        "baseline": manifest.get("baseline_probe_exit") == 0 and manifest.get("baseline_probe_pass") is True,
        "launch": manifest.get("grid") == [80, 48, 1] and manifest.get("block") == [32, 1, 1]
        and manifest.get("target_cta") == [35, 0, 0],
        "layout": manifest.get("variant_count") == len(STAGES) and variant_layout == expected,
        "bytes": manifest.get("trace_bytes_per_variant") == STAGE_BYTES
        and manifest.get("checkpoint_bytes") == TRACE_BYTES,
        "hash": str(manifest.get("checkpoint_sha256", "")).upper() == sha256(rtx) and all(hash_layout),
        "runs": isinstance(variants, list) and all(
            item.get("probe_exit") == 0 and item.get("probe_pass") is True
            and isinstance(item.get("output_preserved"), bool) for item in variants
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))

    stages = []
    clean_divergent = []
    preserved = []
    perturbed = []
    for index, ((name, _, register), variant) in enumerate(zip(STAGES, variants)):
        comparison = compare_stage(rtx, amd, index)
        admissible = variant["output_preserved"]
        if admissible:
            preserved.append(index)
            if not comparison["bitwise_gate"]:
                clean_divergent.append(index)
        else:
            perturbed.append(index)
        stages.append({
            "index": index,
            "name": name,
            "register": register,
            "output_preserved": admissible,
            "admissible_as_clean_oracle": admissible,
            "output_sha256": variant["output_sha256"],
            **comparison,
        })

    output.mkdir(parents=True)
    rtx_path = output / "rtx_single_stage_traces.raw"
    amd_path = output / "amd_mma176_a_path_trace.raw"
    rtx_path.write_bytes(rtx)
    shutil.copyfile(amd_trace, amd_path)
    (output / "rtx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comparison = {
        "schema": 1,
        "experiment": "n0_cta35_mma176_a_fusion_sweep_rtx_vs_rx9070xt",
        "status": "LOCALIZED" if clean_divergent else "NO_CLEAN_DIVERGENCE",
        "classification": "RTX_HIDDEN_FUSION_SINGLE_OBSERVATION_SENSITIVITY",
        "target_cta": [35, 0, 0],
        "baseline_output_sha256": manifest["baseline_output_sha256"],
        "rtx_trace_sha256": sha256(rtx),
        "amd_trace_sha256": sha256(amd),
        "preserved_stage_indices": preserved,
        "perturbed_stage_indices": perturbed,
        "first_clean_divergent_stage": clean_divergent[0] if clean_divergent else None,
        "first_clean_divergent_stage_name": STAGES[clean_divergent[0]][0] if clean_divergent else None,
        "clean_divergent_stage_indices": clean_divergent,
        "stages": stages,
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": 1,
        "experiment": "n0_mma176_a_fusion_sweep_result_ingestion",
        "status": "PASS",
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha256(archive),
        "rtx_device_name": manifest["device_name"],
        "target_cta": [35, 0, 0],
        "output_preserving_variants": len(preserved),
        "output_perturbing_variants": len(perturbed),
        "perturbed_stage_indices": perturbed,
        "first_clean_divergent_stage": comparison["first_clean_divergent_stage"],
        "first_clean_divergent_stage_name": comparison["first_clean_divergent_stage_name"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--amd-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or repo / "results" / f"{datetime.now():%Y%m%d_%H%M%S}_n0_mma176_a_fusion_sweep_return"
    try:
        receipt = process(args.archive, args.amd_trace, output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    print(f"Result: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
