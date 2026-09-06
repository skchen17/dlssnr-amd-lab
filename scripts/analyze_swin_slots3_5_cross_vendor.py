#!/usr/bin/env python3
"""Compare identical-input RTX and RX 9070 XT outputs for graph slots 3-5."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path


TENSOR_BYTES = 1_966_080
DS_HALF_ELEMENTS = 96 * 160 * 32
EXPECTED_RELEASES = {3: 1025, 4: 984, 5: 0}
EXPECTED_GRIDS = {3: [41, 25, 1], 4: [41, 24, 1], 5: [40, 25, 1]}


def _load_e4m3_comparator():
    path = Path(__file__).with_name("analyze_n0_cross_vendor.py")
    spec = importlib.util.spec_from_file_location("_e4m3_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E4M3 comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_E4M3 = _load_e4m3_comparator()
compare_e4m3 = _E4M3.compare
raw_summary = _E4M3.raw_summary


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


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
    mean_r, mean_a = sum_r / finite, sum_a / finite
    var_r = max(0.0, sum_rr / finite - mean_r * mean_r)
    var_a = max(0.0, sum_aa / finite - mean_a * mean_a)
    covariance = sum_ra / finite - mean_r * mean_a
    correlation = covariance / math.sqrt(var_r * var_a) if var_r and var_a else 0.0
    rmse = math.sqrt(squared_error / finite)
    nrmse = rmse / math.sqrt(var_r) if var_r else math.inf
    exact_or_adjacent = (exact + adjacent) / elements
    gate = correlation >= 0.99 and nrmse <= 0.1 and exact_or_adjacent >= 0.95
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
        "reference_tail_nonzero_bytes": sum(value != 0 for value in reference[byte_count:]),
        "candidate_tail_nonzero_bytes": sum(value != 0 for value in candidate[byte_count:]),
        "parity_gate": {
            "correlation_min": 0.99,
            "nrmse_max": 0.1,
            "exact_or_adjacent_min": 0.95,
            "pass": gate,
        },
    }


def analyze(rtx_dir: Path, amd_dir: Path, repeat_dir: Path, package_dir: Path) -> dict:
    rtx_manifest = json.loads((rtx_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    amd_manifest = json.loads((amd_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    amd_verification = json.loads((amd_dir / "verification.json").read_text(encoding="utf-8-sig"))
    package_manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    rtx_slots = {item["slot"]: item for item in rtx_manifest["slots"]}
    amd_slots = {item["slot"]: item for item in amd_manifest["slots"]}

    payload_hashes = {
        name: sha256((package_dir / "payload" / name).read_bytes())
        for name in package_manifest["payload_sha256"]
    }
    payload_integrity = (
        rtx_manifest.get("payload_integrity") is True
        and all(payload_hashes[name] == package_manifest["payload_sha256"][name]
                for name in payload_hashes)
        and all(payload_hashes[name] == rtx_manifest["payload_sha256"][name]
                for name in payload_hashes)
    )
    slot_reports = {}
    execution_gate = (
        rtx_manifest.get("status") == "PASS"
        and amd_manifest.get("status") == "PASS"
        and amd_verification.get("status") == "PASS"
    )
    parity_gate = True
    determinism_gate = True
    for slot in (3, 4, 5):
        rtx_probe = json.loads((rtx_dir / f"slot{slot}" / "probe.json").read_text(encoding="utf-8-sig"))
        rtx_output = (rtx_dir / f"slot{slot}" / "output.raw").read_bytes()
        amd_output = (amd_dir / f"slot{slot}" / "output.raw").read_bytes()
        repeat_output = (repeat_dir / f"slot{slot}" / "output.raw").read_bytes()
        rtx_sync = (rtx_dir / f"slot{slot}" / "sync.raw").read_bytes()
        amd_sync = (amd_dir / f"slot{slot}" / "sync.raw").read_bytes()
        repeat_sync = (repeat_dir / f"slot{slot}" / "sync.raw").read_bytes()
        comparison = compare_e4m3(rtx_output, amd_output)
        deterministic = amd_output == repeat_output and amd_sync == repeat_sync
        sync_exact = rtx_sync == amd_sync
        execution_gate &= (
            rtx_probe.get("pass") is True
            and rtx_probe.get("device_name") == "NVIDIA GeForce RTX 5070"
            and rtx_probe.get("kernel_launched") is True
            and rtx_probe.get("execution_verified") is True
            and rtx_probe.get("n1_grid") == EXPECTED_GRIDS[slot]
            and rtx_probe.get("n1_sync_zero_words") == EXPECTED_RELEASES[slot]
            and len(rtx_output) == len(amd_output) == TENSOR_BYTES
            and rtx_slots[slot].get("output_sha256") == sha256(rtx_output)
            and amd_slots[slot].get("output_sha256") == sha256(amd_output)
        )
        parity_gate &= comparison["parity_gate"]["pass"] and sync_exact
        determinism_gate &= deterministic
        slot_reports[str(slot)] = {
            "output": {
                "rtx": raw_summary(rtx_output),
                "amd": raw_summary(amd_output),
                "comparison": comparison,
            },
            "sync_bitwise_exact": sync_exact,
            "sync_sha256": sha256(rtx_sync),
            "amd_repeat_deterministic": deterministic,
        }

    rtx_extra = (rtx_dir / "slot5" / "extra_output.raw").read_bytes()
    amd_extra = (amd_dir / "slot5" / "extra_output.raw").read_bytes()
    repeat_extra = (repeat_dir / "slot5" / "extra_output.raw").read_bytes()
    extra_comparison = compare_fp16(rtx_extra, amd_extra, DS_HALF_ELEMENTS)
    extra_deterministic = amd_extra == repeat_extra
    extra_tail_zero = (
        extra_comparison["reference_tail_nonzero_bytes"] == 0
        and extra_comparison["candidate_tail_nonzero_bytes"] == 0
    )
    parity_gate &= extra_comparison["parity_gate"]["pass"] and extra_tail_zero
    determinism_gate &= extra_deterministic
    passed = execution_gate and payload_integrity and parity_gate and determinism_gate
    return {
        "schema": 1,
        "experiment": "swin_slots3_5_same_input_rtx5070_vs_rx9070xt",
        "status": "PASS" if passed else "FAIL",
        "classification": "CROSS_VENDOR_SWIN_SLOTS3_5_NUMERICAL_PARITY",
        "counts_as_s7": False,
        "execution_gate": execution_gate,
        "payload_integrity_gate": payload_integrity,
        "parity_gate": parity_gate,
        "amd_determinism_gate": determinism_gate,
        "payload_sha256": payload_hashes,
        "slots": slot_reports,
        "slot5_downsample_fp16": {
            "rtx": raw_summary(rtx_extra),
            "amd": raw_summary(amd_extra),
            "comparison": extra_comparison,
            "amd_repeat_deterministic": extra_deterministic,
        },
        "qualification": (
            "three additional DLSSNR-originated graph kernels execute on RX 9070 XT "
            "with identical-input original-PTX RTX 5070 numerical parity"
        ),
        "limitations": [
            "function-isolated dependency-ordered harness, not the complete 156-slot dispatch",
            "wait inputs model the captured predecessor-ready state; combined shared-sync dispatch remains pending",
            "scalar correctness lowering is not yet an optimized AMD matrix implementation",
        ],
        "next_gate": "classify and translate the slot-6 successor family, then expand dependency coverage",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-dir", required=True, type=Path)
    parser.add_argument("--amd-dir", required=True, type=Path)
    parser.add_argument("--repeat-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.rtx_dir.resolve(), args.amd_dir.resolve(),
                     args.repeat_dir.resolve(), args.package_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
