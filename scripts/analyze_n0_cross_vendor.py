#!/usr/bin/env python3
"""Compare identical-input RTX and RX 9070 XT N0 E4M3 tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def e4m3(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    magnitude = code & 0x7F
    exponent, mantissa = magnitude >> 3, magnitude & 7
    if magnitude == 0x7F:
        return math.nan
    if exponent == 0:
        value = math.ldexp(float(mantissa), -9)
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return sign * value


DECODE = tuple(e4m3(code) for code in range(256))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def raw_summary(data: bytes) -> dict:
    return {
        "bytes": len(data),
        "nonzero_bytes": sum(value != 0 for value in data),
        "sha256": sha256(data),
    }


def compare(reference: bytes, candidate: bytes) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("tensor byte sizes differ")
    n = len(reference)
    exact = adjacent = sign_mismatch = nan_pairs = hamming = 0
    finite = 0
    sum_r = sum_c = sum_rr = sum_cc = sum_rc = 0.0
    absolute_error = squared_error = 0.0
    delta_bins = {"0": 0, "1": 0, "2_3": 0, "4_7": 0, "8_15": 0, "16_31": 0, "32_plus": 0}
    first_mismatches = []
    for index, (r_code, c_code) in enumerate(zip(reference, candidate)):
        if r_code == c_code:
            exact += 1
        elif len(first_mismatches) < 24:
            first_mismatches.append({"offset": index, "rtx": r_code, "amd": c_code})
        same_sign = (r_code >> 7) == (c_code >> 7)
        delta = abs((r_code & 0x7F) - (c_code & 0x7F))
        if r_code != c_code and same_sign and delta == 1:
            adjacent += 1
        if not same_sign:
            sign_mismatch += 1
        if delta == 0:
            delta_bins["0"] += 1
        elif delta == 1:
            delta_bins["1"] += 1
        elif delta <= 3:
            delta_bins["2_3"] += 1
        elif delta <= 7:
            delta_bins["4_7"] += 1
        elif delta <= 15:
            delta_bins["8_15"] += 1
        elif delta <= 31:
            delta_bins["16_31"] += 1
        else:
            delta_bins["32_plus"] += 1
        hamming += (r_code ^ c_code).bit_count()
        r_value, c_value = DECODE[r_code], DECODE[c_code]
        if not math.isfinite(r_value) or not math.isfinite(c_value):
            nan_pairs += 1
            continue
        finite += 1
        error = c_value - r_value
        absolute_error += abs(error)
        squared_error += error * error
        sum_r += r_value
        sum_c += c_value
        sum_rr += r_value * r_value
        sum_cc += c_value * c_value
        sum_rc += r_value * c_value
    mean_r = sum_r / finite
    mean_c = sum_c / finite
    variance_r = max(0.0, sum_rr / finite - mean_r * mean_r)
    variance_c = max(0.0, sum_cc / finite - mean_c * mean_c)
    covariance = sum_rc / finite - mean_r * mean_c
    correlation = covariance / math.sqrt(variance_r * variance_c) if variance_r and variance_c else 0.0
    rmse = math.sqrt(squared_error / finite)
    nrmse = rmse / math.sqrt(variance_r) if variance_r else math.inf
    exact_or_adjacent = exact + adjacent
    gate = correlation >= 0.99 and nrmse <= 0.1 and exact_or_adjacent / n >= 0.95
    return {
        "elements": n,
        "exact": exact,
        "exact_fraction": exact / n,
        "adjacent_same_sign": adjacent,
        "exact_or_adjacent_fraction": exact_or_adjacent / n,
        "sign_mismatches": sign_mismatch,
        "sign_mismatch_fraction": sign_mismatch / n,
        "mean_hamming_bits_per_element": hamming / n,
        "magnitude_code_delta_bins": delta_bins,
        "finite_pairs": finite,
        "nonfinite_pairs": nan_pairs,
        "rtx_mean": mean_r,
        "rtx_stddev": math.sqrt(variance_r),
        "amd_mean": mean_c,
        "amd_stddev": math.sqrt(variance_c),
        "mae": absolute_error / finite,
        "rmse": rmse,
        "nrmse_vs_rtx_stddev": nrmse,
        "pearson_correlation": correlation,
        "parity_gate": {
            "correlation_min": 0.99,
            "nrmse_max": 0.1,
            "exact_or_adjacent_min": 0.95,
            "pass": gate,
        },
        "first_mismatches": first_mismatches,
    }


def byte_mismatches(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("byte sizes differ")
    return sum(a != b for a, b in zip(left, right))


def analyze(root: Path) -> dict:
    rtx_dir = root / "results/20260831_130611_rtx5070_n0_same_input"
    amd_dir = root / "results/20260831_135000_amd_n0_full_grid"
    rtx_manifest = json.loads((rtx_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    tensors = {}
    for name, expected_bytes in (("scratch", 7_864_320), ("output", 1_966_080)):
        rtx = (rtx_dir / f"{name}.raw").read_bytes()
        amd = (amd_dir / f"{name}.raw").read_bytes()
        tensors[name] = {
            "rtx": raw_summary(rtx),
            "amd": raw_summary(amd),
            "comparison": compare(rtx, amd),
            "expected_bytes": expected_bytes,
        }
    repeat_dirs = [
        amd_dir,
        root / "results/20260831_131500_amd_n0_repeat1",
        root / "results/20260831_131501_amd_n0_repeat2",
    ]
    repeat_hashes = {
        name: [sha256((directory / f"{name}.raw").read_bytes()) for directory in repeat_dirs]
        for name in ("scratch", "output")
    }


def analyze_corrected(root: Path) -> dict:
    rtx_dir = root / "results/20260831_130611_rtx5070_n0_same_input"
    amd_dir = root / "results/20260831_140500_amd_n0_full_grid_corrected"
    repeat_dir = root / "results/20260831_140501_amd_n0_full_grid_corrected_repeat"
    payload_dir = root / "deliverables/n0_full_reference_20260831_130219/payload"
    rtx_manifest = json.loads((rtx_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    amd_probe = json.loads((amd_dir / "probe.json").read_text(encoding="utf-8"))
    tensors = {}
    deterministic = True
    for name, expected_bytes in (("scratch", 7_864_320), ("output", 1_966_080)):
        rtx = (rtx_dir / f"{name}.raw").read_bytes()
        amd = (amd_dir / f"{name}.raw").read_bytes()
        repeated = (repeat_dir / f"{name}.raw").read_bytes()
        deterministic &= amd == repeated
        tensors[name] = {
            "rtx": raw_summary(rtx),
            "amd": raw_summary(amd),
            "comparison": compare(rtx, amd),
            "expected_bytes": expected_bytes,
            "amd_repeat_sha256": sha256(repeated),
        }
    payload_hashes = {
        name: sha256((payload_dir / name).read_bytes())
        for name in ("input_rgba16f.raw", "weights.raw", "params.raw")
    }
    input_integrity = all(
        payload_hashes[name] == rtx_manifest["payload_sha256"][name]
        for name in payload_hashes
    )
    execution_ok = (
        rtx_manifest.get("status") == "PASS"
        and rtx_manifest.get("device_name") == "NVIDIA GeForce RTX 5070"
        and rtx_manifest.get("kernel_launched") is True
        and amd_probe.get("pass") is True
        and amd_probe.get("device_name") == "AMD Radeon RX 9070 XT [ZLUDA]"
        and amd_probe.get("module_loaded") is True
        and amd_probe.get("function_resolved") is True
        and amd_probe.get("kernel_launched") is True
        and amd_probe.get("execution_verified") is True
        and amd_probe.get("n0_grid") == [80, 48, 1]
        and all(item["rtx"]["bytes"] == item["expected_bytes"] for item in tensors.values())
        and all(item["amd"]["bytes"] == item["expected_bytes"] for item in tensors.values())
    )
    parity_ok = all(item["comparison"]["parity_gate"]["pass"] for item in tensors.values())
    passed = execution_ok and input_integrity and deterministic and parity_ok
    return {
        "schema": 1,
        "experiment": "n0_corrected_same_input_rtx5070_vs_rx9070xt",
        "status": "PASS" if passed else "FAIL",
        "classification": "CROSS_VENDOR_N0_NUMERICAL_PARITY",
        "counts_as_s6": passed,
        "execution_gate": execution_ok,
        "input_integrity_gate": input_integrity,
        "parity_gate": parity_ok,
        "amd_deterministic_two_runs": deterministic,
        "payload_sha256": payload_hashes,
        "lowering_correction": (
            "m16n8k16 f16 A-fragment source lane/register/half mapping corrected "
            "to the PTX ISA fragment layout"
        ),
        "tensors": tensors,
        "qualification": (
            "first DLSSNR-originated learned neural kernel executed on RX 9070 XT "
            "with same-input RTX numerical parity"
        ),
        "limitations": [
            "standalone isolated N0 harness, not yet wired into the 156-slot Feature-18 graph",
            "controlled synthetic RGBA16F input rather than a complete rendered frame",
            "scalar correctness lowering is not yet an efficient AMD matrix implementation",
        ],
        "next_gate": "translate the next dependent neural function and integrate resource/slot dispatch",
    }
    fp8_dir = root / "results/20260831_132500_amd_fp8_mma_lowered_ptx"
    fp8_ref = root / "results/20260831_112253_rtx_m16n8k32_e4m3_mma/_mma_reference_result_20260831_112253"
    remaining_dir = root / "results/20260831_133100_amd_remaining_lowered_ptx"
    remaining_ref = root / "results/20260831_114007_rtx_remaining_n0_primitives/_remaining_reference_result_20260831_114007"
    amd_f16 = (remaining_dir / "f16_d.raw").read_bytes()
    rtx_f16 = (remaining_ref / "f16_d.raw").read_bytes()
    per_case_f16 = [
        byte_mismatches(amd_f16[index * 256:(index + 1) * 256], rtx_f16[index * 256:(index + 1) * 256])
        for index in range(8)
    ]
    execution_ok = (
        rtx_manifest.get("status") == "PASS"
        and rtx_manifest.get("device_name") == "NVIDIA GeForce RTX 5070"
        and rtx_manifest.get("kernel_launched") is True
        and all(item["rtx"]["bytes"] == item["expected_bytes"] for item in tensors.values())
        and all(item["amd"]["bytes"] == item["expected_bytes"] for item in tensors.values())
    )
    parity_ok = all(item["comparison"]["parity_gate"]["pass"] for item in tensors.values())
    deterministic = all(len(set(values)) == 1 for values in repeat_hashes.values())
    return {
        "schema": 1,
        "experiment": "n0_same_input_rtx5070_vs_rx9070xt",
        "status": "PASS" if execution_ok and parity_ok and deterministic else "FAIL",
        "classification": "CROSS_VENDOR_NUMERICAL_PARITY_FAIL_PRE_MMA_CHECKPOINT_PENDING",
        "counts_as_s6": False,
        "execution_gate": execution_ok,
        "parity_gate": parity_ok,
        "amd_deterministic_three_runs": deterministic,
        "amd_repeat_sha256": repeat_hashes,
        "tensors": tensors,
        "lowered_ptx_micro_oracles": {
            "fp8_mma_output_byte_mismatches": byte_mismatches(
                (fp8_dir / "mma_d.raw").read_bytes(), (fp8_ref / "mma_d.raw").read_bytes()
            ),
            "movmatrix_output_byte_mismatches": byte_mismatches(
                (remaining_dir / "mov_output.raw").read_bytes(),
                (remaining_ref / "mov_output.raw").read_bytes(),
            ),
            "f16_mma_output_byte_mismatches_by_case": per_case_f16,
            "f16_functional_cases_exact": all(per_case_f16[index] == 0 for index in (0, 1, 2, 7)),
            "f16_overflow_stress_cases_scoped": [3, 4, 5, 6],
        },
        "next_gate": (
            "Compare the 3,072-byte first-CTA A/B fragment checkpoint immediately before "
            "the first f16 MMA on RTX and AMD. This separates texture/preprocessing from "
            "matrix-lowering divergence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corrected", action="store_true")
    args = parser.parse_args()
    report = analyze_corrected(args.root.resolve()) if args.corrected else analyze(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
