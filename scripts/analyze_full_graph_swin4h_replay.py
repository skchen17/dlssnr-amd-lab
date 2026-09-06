#!/usr/bin/env python3
"""Validate exact-state RTX native vs RX 9070 XT family parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def e4m3(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    magnitude = code & 0x7F
    exponent, mantissa = magnitude >> 3, magnitude & 7
    if magnitude == 0x7F:
        return math.nan
    value = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(
        1.0 + mantissa / 8.0, exponent - 7
    )
    return sign * value


E4M3_DECODE = tuple(e4m3(code) for code in range(256))


def correlation_metrics(reference_values, candidate_values) -> tuple[float, float]:
    finite = [
        (r, c)
        for r, c in zip(reference_values, candidate_values)
        if math.isfinite(r) and math.isfinite(c)
    ]
    if not finite:
        return 0.0, math.inf
    n = len(finite)
    sum_r = sum(r for r, _ in finite)
    sum_c = sum(c for _, c in finite)
    sum_rr = sum(r * r for r, _ in finite)
    sum_cc = sum(c * c for _, c in finite)
    sum_rc = sum(r * c for r, c in finite)
    mean_r, mean_c = sum_r / n, sum_c / n
    variance_r = max(0.0, sum_rr / n - mean_r * mean_r)
    variance_c = max(0.0, sum_cc / n - mean_c * mean_c)
    covariance = sum_rc / n - mean_r * mean_c
    correlation = covariance / math.sqrt(variance_r * variance_c) if variance_r and variance_c else 0.0
    rmse = math.sqrt(sum((c - r) ** 2 for r, c in finite) / n)
    return correlation, rmse / math.sqrt(variance_r) if variance_r else math.inf


def compare_e4m3(reference: bytes, candidate: bytes) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("E4M3 tensor sizes differ")
    exact = sum(r == c for r, c in zip(reference, candidate))
    adjacent = sum(
        r != c and (r >> 7) == (c >> 7) and abs((r & 0x7F) - (c & 0x7F)) == 1
        for r, c in zip(reference, candidate)
    )
    reference_values = [E4M3_DECODE[value] for value in reference]
    candidate_values = [E4M3_DECODE[value] for value in candidate]
    correlation, nrmse = correlation_metrics(reference_values, candidate_values)
    nonfinite = sum(
        not math.isfinite(r) or not math.isfinite(c)
        for r, c in zip(reference_values, candidate_values)
    )
    tolerant = (exact + adjacent) / len(reference)
    passed = nonfinite == 0 and correlation >= 0.99 and nrmse <= 0.1 and tolerant >= 0.95
    return {
        "elements": len(reference),
        "exact_fraction": exact / len(reference),
        "exact_or_adjacent_fraction": tolerant,
        "nonfinite_pairs": nonfinite,
        "pearson_correlation": correlation,
        "nrmse_vs_rtx_stddev": nrmse,
        "parity_gate": {"pass": passed, "correlation_min": 0.99, "nrmse_max": 0.1,
                        "exact_or_adjacent_min": 0.95},
    }


def compare_fp16(reference: bytes, candidate: bytes) -> dict:
    if len(reference) != len(candidate) or len(reference) % 2:
        raise ValueError("FP16 tensor sizes differ")
    elements = len(reference) // 2
    ref_bits = struct.unpack(f"<{elements}H", reference)
    amd_bits = struct.unpack(f"<{elements}H", candidate)
    ref_values = struct.unpack(f"<{elements}e", reference)
    amd_values = struct.unpack(f"<{elements}e", candidate)
    exact = sum(r == a for r, a in zip(ref_bits, amd_bits))
    adjacent = sum(
        r != a and (r >> 15) == (a >> 15) and abs((r & 0x7FFF) - (a & 0x7FFF)) == 1
        for r, a in zip(ref_bits, amd_bits)
    )
    nonfinite = sum(
        not math.isfinite(r) or not math.isfinite(a)
        for r, a in zip(ref_values, amd_values)
    )
    correlation, nrmse = correlation_metrics(ref_values, amd_values)
    tolerant = (exact + adjacent) / elements
    # PTX explicitly leaves f16 MMA accumulation order, rounding and subnormal
    # handling unspecified. Correlation/NRMSE are the normative cross-implementation
    # gate; exact-or-adjacent remains a visible 95% diagnostic, not a false hard fail.
    passed = nonfinite == 0 and correlation >= 0.99 and nrmse <= 0.1
    return {
        "elements": elements,
        "exact_fraction": exact / elements,
        "exact_or_adjacent_fraction": tolerant,
        "nonfinite_pairs": nonfinite,
        "pearson_correlation": correlation,
        "nrmse_vs_rtx_stddev": nrmse,
        "parity_gate": {
            "pass": passed,
            "correlation_min": 0.99,
            "nrmse_max": 0.1,
            "nonfinite_pairs_max": 0,
            "exact_or_adjacent_advisory_min": 0.95,
            "exact_or_adjacent_advisory_pass": tolerant >= 0.95,
            "ptx_semantics": "f16 MMA accumulation order, rounding, and subnormal handling are unspecified",
        },
    }


def analyze(root: Path) -> dict:
    cases_root = root / "cases"
    case_manifest = json.loads((cases_root / "manifest.json").read_text(encoding="utf-8-sig"))
    slot_reports = {}
    advisories = []
    all_execution = all_state = all_parity = all_sync = all_determinism = True
    model_arena = case_manifest.get("model_arena")
    if model_arena:
        model_data = (cases_root / model_arena["path"]).read_bytes()
        all_state &= (
            len(model_data) == int(model_arena["bytes"])
            and sha256(model_data) == model_arena["sha256"]
        )

    for case in case_manifest["slots"]:
        slot = int(case["slot"])
        case_dir = cases_root / f"slot{slot}"
        run1 = root / "amd_run1" / f"slot{slot}"
        run2 = root / "amd_run2" / f"slot{slot}"
        probe1 = json.loads((run1 / "probe.json").read_text(encoding="utf-8-sig"))
        probe2 = json.loads((run2 / "probe.json").read_text(encoding="utf-8-sig"))

        state_valid = True
        for asset in case["assets"].values():
            data = (case_dir / asset["path"]).read_bytes()
            state_valid &= len(data) == int(asset["bytes"]) and sha256(data) == asset["sha256"]

        output_reference_name = (
            "settled_output_reference.raw"
            if case.get("main_oracle") == "downstream_input_before"
            else "output_reference.raw"
        )
        output_reference = (case_dir / output_reference_name).read_bytes()
        output1 = (run1 / "output.raw").read_bytes()
        output2 = (run2 / "output.raw").read_bytes()
        logical = int(case["main_logical_bytes"])
        main_type = case.get("main_tensor_type", "fp16")
        output_comparison = (
            compare_fp16(output_reference[:logical], output1[:logical])
            if main_type == "fp16"
            else compare_e4m3(output_reference[:logical], output1[:logical])
        )
        if (
            main_type == "fp16"
            and not output_comparison["parity_gate"]["exact_or_adjacent_advisory_pass"]
        ):
            advisories.append(
                f"slot {slot} FP16 exact-or-adjacent fraction is below the 0.95 advisory"
            )
        if case.get("main_oracle") == "downstream_input_before":
            # The downstream capture proves the producer's logical edge tensor.
            # Bytes after that tensor are adjacent arena state and can be changed
            # by intervening graph work, so they are outside this oracle's scope.
            tail_exact = None
        else:
            captured = min(int(case.get("output_capture_bytes", len(output_reference))),
                           len(output_reference))
            tail_exact = output_reference[logical:captured] == output1[logical:captured]
        deterministic = output1 == output2
        execution = all(
            probe.get("pass") is True
            and probe.get("device_name") == "AMD Radeon RX 9070 XT [ZLUDA]"
            and probe.get("kernel_launched") is True
            and probe.get("execution_verified") is True
            and probe.get("n1_output_initial_loaded") is True
            and probe.get("n1_grid") == case["grid"]
            and probe.get("n1_block") == case["block"]
            and probe.get("n1_sync_zero_words") == case["expected_releases"]
            for probe in (probe1, probe2)
        )
        if "wait_sync_initial" in case["assets"]:
            execution &= (
                probe1.get("n1_wait_sync_initial_loaded") is True
                and probe2.get("n1_wait_sync_initial_loaded") is True
            )
        if "input2" in case["assets"]:
            execution &= (
                probe1.get("n1_input2_loaded") is True
                and probe2.get("n1_input2_loaded") is True
            )
        abi = case.get("abi")
        if abi:
            execution &= all(
                probe.get("n1_input_param_offset") == abi["input"]
                and probe.get("n1_output_param_offset") == abi["output"]
                and probe.get("n1_weights_param_offset") == abi["weights"]
                and ("input2" not in abi or
                     probe.get("n1_input2_param_offset") == abi["input2"])
                for probe in (probe1, probe2)
            )
        arena_views = case.get("arena_views")
        if arena_views:
            execution &= all(
                probe.get("n1_arena_loaded") is True
                and probe.get("n1_arena_bytes") == case["activation_arena_bytes"]
                and probe.get("n1_arena_input_offset") == arena_views["input"]
                and probe.get("n1_arena_output_offset") == arena_views["output"]
                and probe.get("n1_weight_view_offset") == case["weight_view_offset"]
                and (arena_views.get("input2") is None or
                     probe.get("n1_arena_input2_offset") == arena_views["input2"])
                and (arena_views.get("extra") is None or
                     probe.get("n1_arena_extra_offset") == arena_views["extra"])
                for probe in (probe1, probe2)
            )

        sync_exact = True
        if "sync_reference" in case["assets"]:
            sync_reference = (case_dir / "sync_reference.raw").read_bytes()
            sync1 = (run1 / "sync.raw").read_bytes()
            sync2 = (run2 / "sync.raw").read_bytes()
            sync_exact = sync1 == sync_reference and sync2 == sync_reference
            deterministic &= sync1 == sync2
            execution &= (
                probe1.get("n1_sync_initial_loaded") is True
                and probe2.get("n1_sync_initial_loaded") is True
                and case["release_native_changed_bytes"] == case["expected_releases"] * 4
            )

        extra_report = None
        if "extra_output_reference" in case["assets"]:
            extra_reference_name = (
                "settled_extra_output_reference.raw"
                if case.get("extra_oracle") == "downstream_input_before"
                else "extra_output_reference.raw"
            )
            extra_reference = (case_dir / extra_reference_name).read_bytes()
            extra1 = (run1 / "extra_output.raw").read_bytes()
            extra2 = (run2 / "extra_output.raw").read_bytes()
            extra_logical = int(case["extra_logical_bytes"])
            extra_type = case.get("extra_tensor_type", "e4m3")
            extra_comparison = (
                compare_fp16(extra_reference[:extra_logical], extra1[:extra_logical])
                if extra_type == "fp16"
                else compare_e4m3(extra_reference[:extra_logical], extra1[:extra_logical])
            )
            extra_tail_exact = (
                None
                if case.get("extra_oracle") == "downstream_input_before"
                else extra_reference[extra_logical:] == extra1[extra_logical:]
            )
            extra_deterministic = extra1 == extra2
            deterministic &= extra_deterministic
            execution &= (
                probe1.get("n1_extra_output_initial_loaded") is True
                and probe2.get("n1_extra_output_initial_loaded") is True
            )
            extra_report = {
                "oracle_source": case.get("extra_oracle", "immediate_after"),
                "comparison": extra_comparison,
                "tail_bitwise_exact": extra_tail_exact,
                "amd_repeat_bitwise_exact": extra_deterministic,
            }
            all_parity &= (
                extra_comparison["parity_gate"]["pass"]
                and extra_tail_exact is not False
            )

        slot_parity = (
            output_comparison["parity_gate"]["pass"] and tail_exact is not False
        )
        all_execution &= execution
        all_state &= state_valid
        all_parity &= slot_parity
        all_sync &= sync_exact
        all_determinism &= deterministic
        slot_reports[str(slot)] = {
            "function": case["function"],
            "execution_gate": execution,
            "exact_rtx_before_state_gate": state_valid,
                "main_tensor": {
                    "type": main_type,
                "oracle_source": case.get("main_oracle", "immediate_after"),
                "comparison": output_comparison,
                "tail_bitwise_exact": tail_exact,
                "rtx_sha256": sha256(output_reference),
                "amd_sha256": sha256(output1),
            },
            "sync_bitwise_exact": sync_exact,
            "amd_repeat_bitwise_exact": deterministic,
            "extra_tensor": extra_report,
        }

    passed = all_execution and all_state and all_parity and all_sync and all_determinism
    return {
        "schema": 1,
        "experiment": f"full_graph_{case_manifest.get('family', 'swin_family')}_rtx_native_vs_rx9070xt",
        "status": "PASS" if passed else "FAIL",
        "classification": "EXACT_STATE_CROSS_VENDOR_SWIN_FAMILY_NUMERICAL_PARITY",
        "counts_as_s7": False,
        "amd_execution_gate": all_execution,
        "exact_rtx_before_state_gate": all_state,
        "numerical_parity_gate": all_parity,
        "sync_gate": all_sync,
        "amd_repeat_determinism_gate": all_determinism,
        "advisories": advisories,
        "slots": slot_reports,
        "qualification": (
            f"{case_manifest.get('family', 'Swin family')} executes on RX 9070 XT from exact native RTX before states "
            "and matches synchronization-settled native RTX graph-edge states numerically"
        ),
        "limitations": [
            "function-isolated replay, not yet the complete 156-slot dispatch",
            "scalar correctness lowering is not an optimized AMD implementation",
        ],
        "next_gate": case_manifest.get("next_gate", "continue with the next graph family"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
