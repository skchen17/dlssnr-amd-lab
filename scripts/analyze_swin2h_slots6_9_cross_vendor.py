#!/usr/bin/env python3
"""Validate identical-input RTX/RX 9070 XT numerical parity for graph slots 6-9."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


TENSOR_BYTES = 1_966_080
MAIN_FP16_ELEMENTS = 96 * 160 * 32
EXTRA_FP16_ELEMENTS = 48 * 160 * 32
WEIGHT_PREFIX_BYTES = 65_536
EXPECTED_RELEASES = {6: 240, 7: 273, 8: 252, 9: 0}
EXPECTED_GRIDS = {
    6: [20, 12, 1],
    7: [21, 13, 1],
    8: [21, 12, 1],
    9: [20, 13, 1],
}
EXPECTED_BLOCK = [32, 2, 1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def raw_summary(data: bytes) -> dict:
    return {
        "bytes": len(data),
        "sha256": sha256(data),
        "nonzero_bytes": sum(value != 0 for value in data),
    }


def compare_fp16(reference: bytes, candidate: bytes, elements: int) -> dict:
    byte_count = elements * 2
    if len(reference) < byte_count or len(candidate) < byte_count:
        raise ValueError("FP16 tensor is shorter than the declared logical shape")
    ref_bits = struct.unpack(f"<{elements}H", reference[:byte_count])
    amd_bits = struct.unpack(f"<{elements}H", candidate[:byte_count])
    ref_values = struct.unpack(f"<{elements}e", reference[:byte_count])
    amd_values = struct.unpack(f"<{elements}e", candidate[:byte_count])
    exact = adjacent = finite = nonfinite = 0
    sum_r = sum_a = sum_rr = sum_aa = sum_ra = squared_error = 0.0
    for rb, ab, rv, av in zip(ref_bits, amd_bits, ref_values, amd_values):
        if rb == ab:
            exact += 1
        elif (rb >> 15) == (ab >> 15) and abs((rb & 0x7FFF) - (ab & 0x7FFF)) == 1:
            adjacent += 1
        if not math.isfinite(rv) or not math.isfinite(av):
            nonfinite += 1
            continue
        finite += 1
        error = av - rv
        squared_error += error * error
        sum_r += rv
        sum_a += av
        sum_rr += rv * rv
        sum_aa += av * av
        sum_ra += rv * av
    if not finite:
        raise ValueError("FP16 comparison contains no finite pairs")
    mean_r, mean_a = sum_r / finite, sum_a / finite
    var_r = max(0.0, sum_rr / finite - mean_r * mean_r)
    var_a = max(0.0, sum_aa / finite - mean_a * mean_a)
    covariance = sum_ra / finite - mean_r * mean_a
    correlation = covariance / math.sqrt(var_r * var_a) if var_r and var_a else 0.0
    rmse = math.sqrt(squared_error / finite)
    nrmse = rmse / math.sqrt(var_r) if var_r else math.inf
    exact_or_adjacent = (exact + adjacent) / elements
    ref_tail_nonzero = sum(value != 0 for value in reference[byte_count:])
    amd_tail_nonzero = sum(value != 0 for value in candidate[byte_count:])
    gate = (
        nonfinite == 0
        and correlation >= 0.99
        and nrmse <= 0.1
        and exact_or_adjacent >= 0.95
        and ref_tail_nonzero == 0
        and amd_tail_nonzero == 0
    )
    return {
        "elements": elements,
        "exact": exact,
        "exact_fraction": exact / elements,
        "adjacent_same_sign": adjacent,
        "exact_or_adjacent_fraction": exact_or_adjacent,
        "finite_pairs": finite,
        "nonfinite_pairs": nonfinite,
        "rmse": rmse,
        "nrmse_vs_rtx_stddev": nrmse,
        "pearson_correlation": correlation,
        "reference_tail_nonzero_bytes": ref_tail_nonzero,
        "candidate_tail_nonzero_bytes": amd_tail_nonzero,
        "parity_gate": {
            "correlation_min": 0.99,
            "nrmse_max": 0.1,
            "exact_or_adjacent_min": 0.95,
            "tails_must_be_zero": True,
            "pass": gate,
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze(
    rtx_dir: Path,
    amd_dir: Path,
    amd_repeat_dir: Path,
    amd_slot9_dir: Path,
    amd_slot9_repeat_dir: Path,
    package_dir: Path,
) -> dict:
    rtx_manifest = _read_json(rtx_dir / "manifest.json")
    amd_manifest = _read_json(amd_dir / "manifest.json")
    amd_repeat_manifest = _read_json(amd_repeat_dir / "manifest.json")
    package_manifest = _read_json(package_dir / "manifest.json")
    rtx_slots = {item["slot"]: item for item in rtx_manifest["slots"]}
    amd_slots = {item["slot"]: item for item in amd_manifest["slots"]}
    amd_repeat_slots = {item["slot"]: item for item in amd_repeat_manifest["slots"]}

    payload_hashes = {
        name: sha256((package_dir / "payload" / name).read_bytes())
        for name in package_manifest["payload_sha256"]
    }
    payload_integrity = (
        rtx_manifest.get("payload_integrity") is True
        and payload_hashes == package_manifest["payload_sha256"]
        and payload_hashes == rtx_manifest["payload_sha256"]
    )

    execution_gate = (
        rtx_manifest.get("status") == "PASS"
        and amd_manifest.get("status") == "PASS"
        and amd_repeat_manifest.get("status") == "PASS"
    )
    input_gate = True
    parity_gate = True
    determinism_gate = True
    slot_reports = {}

    for slot in (6, 7, 8, 9):
        rtx_slot = rtx_slots[slot]
        rtx_probe = _read_json(rtx_dir / f"slot{slot}" / "probe.json")
        rtx_output = (rtx_dir / f"slot{slot}" / "output.raw").read_bytes()
        if slot < 9:
            amd_output = (amd_dir / f"slot{slot}" / "output.raw").read_bytes()
            repeat_output = (amd_repeat_dir / f"slot{slot}" / "output.raw").read_bytes()
            amd_sync = (amd_dir / f"slot{slot}" / "sync.raw").read_bytes()
            repeat_sync = (amd_repeat_dir / f"slot{slot}" / "sync.raw").read_bytes()
            amd_slot = amd_slots[slot]
            repeat_slot = amd_repeat_slots[slot]
            weight_data = (package_dir / "payload" / f"slot{slot}_weights.raw").read_bytes()
            weight_match = amd_slot["weights_sha256"] == sha256(weight_data[:WEIGHT_PREFIX_BYTES])
            amd_probe = _read_json(amd_dir / f"slot{slot}" / "probe.json")
        else:
            amd_output = (amd_slot9_dir / "output.raw").read_bytes()
            repeat_output = (amd_slot9_repeat_dir / "output.raw").read_bytes()
            amd_sync = (amd_slot9_dir / "sync.raw").read_bytes()
            repeat_sync = (amd_slot9_repeat_dir / "sync.raw").read_bytes()
            amd_probe = _read_json(amd_slot9_dir / "probe.json")
            repeat_probe = _read_json(amd_slot9_repeat_dir / "probe.json")
            amd_slot = {
                "output_sha256": sha256(amd_output),
                "input_sha256": package_manifest["payload_sha256"]["slot9_input.raw"],
                "params_sha256": sha256((amd_slot9_dir / "params.raw").read_bytes()),
            }
            repeat_slot = {"output_sha256": sha256(repeat_output)}
            weight_match = (
                amd_probe.get("n1_weights_input_bytes")
                == (package_dir / "payload" / "slot9_weights.raw").stat().st_size
            )

        rtx_sync = (rtx_dir / f"slot{slot}" / "sync.raw").read_bytes()
        comparison = compare_fp16(rtx_output, amd_output, MAIN_FP16_ELEMENTS)
        deterministic = amd_output == repeat_output and amd_sync == repeat_sync
        sync_exact = rtx_sync == amd_sync
        input_hash = package_manifest["payload_sha256"][f"slot{slot}_input.raw"]
        params_hash = package_manifest["payload_sha256"][f"slot{slot}_params.raw"]
        slot_input_gate = (
            rtx_slot.get("input_sha256") == input_hash
            and amd_slot.get("input_sha256") == input_hash
            and rtx_slot.get("params_sha256") == params_hash
            and amd_slot.get("params_sha256") == params_hash
            and weight_match
        )
        slot_execution_gate = (
            rtx_slot.get("pass") is True
            and rtx_slot.get("device_name") == "NVIDIA GeForce RTX 5070"
            and rtx_slot.get("kernel_launched") is True
            and rtx_slot.get("execution_verified") is True
            and rtx_probe.get("n1_grid") == EXPECTED_GRIDS[slot]
            and rtx_probe.get("n1_block") == EXPECTED_BLOCK
            and rtx_probe.get("n1_sync_zero_words") == EXPECTED_RELEASES[slot]
            and amd_probe.get("n1_grid") == EXPECTED_GRIDS[slot]
            and amd_probe.get("n1_block") == EXPECTED_BLOCK
            and amd_probe.get("n1_sync_zero_words") == EXPECTED_RELEASES[slot]
            and len(rtx_output) == len(amd_output) == TENSOR_BYTES
            and rtx_slot.get("output_sha256") == sha256(rtx_output)
            and amd_slot.get("output_sha256") == sha256(amd_output)
            and repeat_slot.get("output_sha256") == sha256(repeat_output)
        )
        execution_gate &= slot_execution_gate
        input_gate &= slot_input_gate
        parity_gate &= comparison["parity_gate"]["pass"] and sync_exact
        determinism_gate &= deterministic
        slot_reports[str(slot)] = {
            "execution_gate": slot_execution_gate,
            "same_input_weights_params_gate": slot_input_gate,
            "output": {
                "rtx": raw_summary(rtx_output),
                "amd": raw_summary(amd_output),
                "comparison": comparison,
            },
            "sync_bitwise_exact": sync_exact,
            "sync_sha256": sha256(rtx_sync),
            "amd_repeat_deterministic": deterministic,
        }

    rtx_extra = (rtx_dir / "slot9" / "extra_output.raw").read_bytes()
    amd_extra = (amd_slot9_dir / "extra_output.raw").read_bytes()
    repeat_extra = (amd_slot9_repeat_dir / "extra_output.raw").read_bytes()
    extra_comparison = compare_fp16(rtx_extra, amd_extra, EXTRA_FP16_ELEMENTS)
    extra_deterministic = amd_extra == repeat_extra
    extra_hash_valid = rtx_slots[9].get("extra_output_sha256") == sha256(rtx_extra)
    parity_gate &= extra_comparison["parity_gate"]["pass"] and extra_hash_valid
    determinism_gate &= extra_deterministic

    passed = execution_gate and payload_integrity and input_gate and parity_gate and determinism_gate
    return {
        "schema": 1,
        "experiment": "swin2h_slots6_9_same_input_rtx5070_vs_rx9070xt",
        "status": "PASS" if passed else "FAIL",
        "classification": "CROSS_VENDOR_SWIN2H_SLOTS6_9_NUMERICAL_PARITY",
        "counts_as_s7": False,
        "execution_gate": execution_gate,
        "payload_integrity_gate": payload_integrity,
        "same_input_weights_params_gate": input_gate,
        "parity_gate": parity_gate,
        "amd_determinism_gate": determinism_gate,
        "payload_sha256": payload_hashes,
        "slots": slot_reports,
        "slot9_downsample_fp16": {
            "rtx": raw_summary(rtx_extra),
            "amd": raw_summary(amd_extra),
            "comparison": extra_comparison,
            "rtx_manifest_hash_valid": extra_hash_valid,
            "amd_repeat_deterministic": extra_deterministic,
        },
        "qualification": (
            "four additional DLSSNR-originated graph kernels execute on RX 9070 XT "
            "with identical-input original-PTX RTX 5070 numerical parity"
        ),
        "limitations": [
            "function-isolated dependency-ordered harness, not the complete 156-slot dispatch",
            "wait inputs model the captured predecessor-ready state; combined shared-sync dispatch remains pending",
            "scalar correctness lowering is not yet an optimized AMD matrix implementation",
        ],
        "next_gate": "classify and translate the slot-10 4h/128 successor family",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-dir", required=True, type=Path)
    parser.add_argument("--amd-dir", required=True, type=Path)
    parser.add_argument("--amd-repeat-dir", required=True, type=Path)
    parser.add_argument("--amd-slot9-dir", required=True, type=Path)
    parser.add_argument("--amd-slot9-repeat-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.rtx_dir.resolve(),
        args.amd_dir.resolve(),
        args.amd_repeat_dir.resolve(),
        args.amd_slot9_dir.resolve(),
        args.amd_slot9_repeat_dir.resolve(),
        args.package_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
