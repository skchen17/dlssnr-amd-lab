#!/usr/bin/env python3
"""Numerically validate the stateful 156-launch RX 9070 XT graph replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.analyze_full_graph_swin4h_replay import compare_e4m3, compare_fp16
except ModuleNotFoundError:
    from analyze_full_graph_swin4h_replay import compare_e4m3, compare_fp16


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def compare_rgba16f_rgb(reference: bytes, candidate: bytes) -> dict:
    """Compare RGBA16F images normatively on RGB, excluding constant alpha."""
    if len(reference) != len(candidate) or len(reference) % 8:
        raise ValueError("RGBA16F image sizes differ")
    ref_rgb = bytearray()
    candidate_rgb = bytearray()
    alpha_exact = 0
    pixels = len(reference) // 8
    for offset in range(0, len(reference), 8):
        ref_rgb.extend(reference[offset:offset + 6])
        candidate_rgb.extend(candidate[offset:offset + 6])
        alpha_exact += reference[offset + 6:offset + 8] == candidate[offset + 6:offset + 8]
    comparison = compare_fp16(bytes(ref_rgb), bytes(candidate_rgb))
    comparison["normative_channels"] = "RGB"
    comparison["alpha_excluded_from_normative_metrics"] = True
    comparison["alpha_exact_fraction"] = alpha_exact / pixels
    return comparison


def analyze(plan_path: Path, execution_root: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    execution = json.loads((execution_root / "execution.json").read_text(encoding="utf-8-sig"))
    if execution.get('color_input', {}).get('external'):
        raise ValueError("captured zero-input checkpoints cannot validate an external-color execution")
    state_gate = True
    for key in ("activation_arena_initial", "model_arena"):
        spec = plan[key]
        path = plan_path.parent / spec["path"]
        data = path.read_bytes()
        state_gate &= len(data) == int(spec["bytes"]) and sha256(data) == spec["sha256"]

    execution_gate = (
        execution.get("status") == "PASS"
        and execution.get("pass") is True
        and execution.get("device") == "AMD Radeon RX 9070 XT [ZLUDA]"
        and execution.get("single_context") is True
        and execution.get("shared_activation_arena") is True
        and execution.get("shared_model_arena") is True
        and execution.get("rtx_intermediate_state_injection") is False
        and execution.get("all_slots_executed") is True
        and len(execution.get("runs", [])) == 2
        and all(len(run.get("slots", [])) == 156 for run in execution.get("runs", []))
    )
    repeat_gate = execution.get("repeat_final_bitwise_exact") is True
    numerical_gate = True
    boundary_reports = {}
    advisories = []

    for slot_text, checkpoint in plan["checkpoints"].items():
        slot = int(slot_text)
        reference = (plan_path.parent / checkpoint["reference"]["path"]).read_bytes()
        logical_bytes = int(checkpoint.get("logical_bytes", len(reference)))
        if len(reference) < logical_bytes:
            raise ValueError(f"slot {slot} reference is shorter than its logical tensor")
        reference = reference[:logical_bytes]
        candidates = [(execution_root / f"run{index}" / f"slot{slot}.raw").read_bytes()
                      for index in (1, 2)]
        if any(len(candidate) != logical_bytes for candidate in candidates):
            raise ValueError(f"slot {slot} checkpoint size differs from its logical tensor")
        if checkpoint["tensor_type"] == "sync_u32":
            comparison = {
                "bytes": len(reference),
                "exact_fraction": sum(a == b for a, b in zip(reference, candidates[0])) / len(reference),
                "parity_gate": {"pass": candidates[0] == reference, "criterion": "bitwise exact"},
            }
        elif checkpoint["tensor_type"] == "rgba16f":
            comparison = compare_rgba16f_rgb(reference, candidates[0])
        elif checkpoint["tensor_type"] == "fp16":
            comparison = compare_fp16(reference, candidates[0])
        else:
            comparison = compare_e4m3(reference, candidates[0])
            advisory = comparison["exact_or_adjacent_fraction"] >= 0.95
            comparison["parity_gate"] = {
                "pass": comparison["nonfinite_pairs"] == 0
                        and comparison["pearson_correlation"] >= 0.99
                        and comparison["nrmse_vs_rtx_stddev"] <= 0.1,
                "correlation_min": 0.99,
                "nrmse_max": 0.1,
                "nonfinite_pairs_max": 0,
                "exact_or_adjacent_advisory_min": 0.95,
                "exact_or_adjacent_advisory_pass": advisory,
            }
            if not advisory:
                advisories.append(
                    f"slot {slot} integrated E4M3 exact-or-adjacent fraction "
                    f"{comparison['exact_or_adjacent_fraction']:.9f} is below 0.95")
        passed = comparison["parity_gate"]["pass"]
        deterministic = candidates[0] == candidates[1]
        numerical_gate &= passed
        repeat_gate &= deterministic
        boundary_reports[slot_text] = {
            "tensor_type": checkpoint["tensor_type"],
            "oracle_source": checkpoint["oracle_source"],
            "comparison": comparison,
            "amd_repeat_bitwise_exact": deterministic,
            "rtx_sha256": sha256(reference),
            "amd_sha256": sha256(candidates[0]),
        }

    final_reference = (plan_path.parent / "references" / "slot155_rgba16f.raw").read_bytes()
    final_outputs = [(execution_root / f"run{index}" / "final_rgba16f.raw").read_bytes()
                     for index in (1, 2)]
    final_comparison = compare_rgba16f_rgb(final_reference, final_outputs[0])
    copy_gate = True
    for index in (1, 2):
        post = (execution_root / f"run{index}" / "slot154.raw").read_bytes()
        copy_gate &= post == final_outputs[index - 1]
    final_gate = final_comparison["parity_gate"]["pass"] and copy_gate
    numerical_gate &= final_gate
    repeat_gate &= final_outputs[0] == final_outputs[1]
    passed = execution_gate and state_gate and numerical_gate and repeat_gate and final_gate
    failed_boundaries = sorted(
        int(slot) for slot, item in boundary_reports.items()
        if not item["comparison"]["parity_gate"]["pass"]
    )
    first_failure = failed_boundaries[0] if failed_boundaries else None
    if execution.get("rtx_intermediate_state_injection"):
        next_gate = "use this injection result only for causal localization; rerun the fix without RTX state"
    elif first_failure is not None:
        next_gate = f"correct earliest strict boundary slot {first_failure} without RTX state injection"
    else:
        next_gate = "export and inspect the integrated RX RGBA16F image"
    texture_limitation = (
        "The slot-154 D3D12 texture is recreated from a supplied RGBA16F snapshot "
        "with the captured point/border sampler model."
        if plan.get("post_texture") else
        "Texture instructions lowered to zero are an explicit approximation, not "
        "a validated model of the captured slot-154 SRV."
    )
    return {
        "schema": 1,
        "experiment": "rx9070xt_full_graph_integrated_rtx_parity",
        "status": "PASS" if passed else "FAIL",
        "classification": "SINGLE_CONTEXT_STATEFUL_156_LAUNCH_CROSS_VENDOR_PARITY",
        "counts_as_s7": passed,
        "execution_gate": execution_gate,
        "frame_start_state_gate": state_gate,
        "boundary_numerical_gate": numerical_gate,
        "amd_repeat_determinism_gate": repeat_gate,
        "slot155_gpu_copy_gate": copy_gate,
        "final_rgba16f_image_gate": final_gate,
        "first_strict_failing_boundary": first_failure,
        "final_comparison": final_comparison,
        "final_rtx_sha256": sha256(final_reference),
        "final_amd_sha256": sha256(final_outputs[0]),
        "boundaries": boundary_reports,
        "advisories": advisories,
        "evidence": {
            "device": execution.get("device"),
            "launches_per_run": [len(run.get("slots", [])) for run in execution.get("runs", [])],
            "runs": len(execution.get("runs", [])),
            "single_context": execution.get("single_context"),
            "rtx_intermediate_state_injection": execution.get("rtx_intermediate_state_injection"),
        },
        "limitations": [
            texture_limitation,
            "PTX resource handles are explicitly adapted to linear device pointers where ZLUDA lacks CUDA surface objects.",
            "RGBA16F parity is evaluated on RGB only; constant alpha is reported separately and cannot satisfy the image gate.",
        ],
        "next_gate": next_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.plan.resolve(), args.execution.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
