#!/usr/bin/env python3
"""Analyze exact-state ViT 1D graph-edge replay on RX 9070 XT."""

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


def analyze(root: Path) -> dict:
    cases_root = root / "cases"
    manifest = json.loads((cases_root / "manifest.json").read_text(encoding="utf-8-sig"))
    model = (cases_root / manifest["model_arena"]["path"]).read_bytes()
    state_gate = (
        len(model) == int(manifest["model_arena"]["bytes"])
        and sha256(model) == manifest["model_arena"]["sha256"]
    )
    execution_gate = numerical_gate = determinism_gate = True
    reports = {}
    advisories = []
    settled_edges = 0
    exact_edges = 0

    for case in manifest["slots"]:
        slot = int(case["slot"])
        case_dir = cases_root / f"slot{slot}"
        run1 = root / "amd_run1" / f"slot{slot}"
        run2 = root / "amd_run2" / f"slot{slot}"
        probe1 = json.loads((run1 / "probe.json").read_text(encoding="utf-8-sig"))
        probe2 = json.loads((run2 / "probe.json").read_text(encoding="utf-8-sig"))
        arena1 = (run1 / "arena_after.raw").read_bytes()
        arena2 = (run2 / "arena_after.raw").read_bytes()
        arena_initial = (case_dir / "activation_arena_initial.raw").read_bytes()
        params = (case_dir / "params.raw").read_bytes()
        case_state = (
            len(arena_initial) == int(case["activation_arena"]["bytes"])
            and sha256(arena_initial) == case["activation_arena"]["sha256"]
            and len(params) == int(case["params"]["bytes"])
            and sha256(params) == case["params"]["sha256"]
        )
        state_gate &= case_state
        execution = all(
            probe.get("pass") is True
            and probe.get("device_name") == "AMD Radeon RX 9070 XT [ZLUDA]"
            and probe.get("kernel_launched") is True
            and probe.get("execution_verified") is True
            and probe.get("n1_arena_output_written") is True
            and probe.get("n1_arena_param_view_count") ==
                len(case["activation_param_views"])
            and probe.get("n1_grid") == case["grid"]
            and probe.get("n1_block") == case["block"]
            for probe in (probe1, probe2)
        )
        execution_gate &= execution
        deterministic = arena1 == arena2
        determinism_gate &= deterministic
        output_reports = []
        for output in case["outputs"]:
            start = int(output["arena_offset"])
            size = int(output["logical_bytes"])
            candidate = arena1[start:start + size]
            candidate2 = arena2[start:start + size]
            reference = (case_dir / output["asset"]["path"]).read_bytes()
            comparison = (
                compare_fp16(reference, candidate)
                if output["tensor_type"] == "fp16"
                else compare_e4m3(reference, candidate)
            )
            settled = output["oracle_source"].startswith(f"slot{slot + 1}_")
            if settled:
                settled_edges += 1
                exact_edges += candidate == reference
                numerical_gate &= comparison["parity_gate"]["pass"]
            else:
                advisories.append(
                    f"slot {slot} parameter +{output['param_offset']} has only an asynchronous immediate-after diagnostic"
                )
            output_reports.append({
                "param_offset": output["param_offset"],
                "tensor_type": output["tensor_type"],
                "oracle_source": output["oracle_source"],
                "normative_settled_edge": settled,
                "comparison": comparison,
                "amd_repeat_bitwise_exact": candidate == candidate2,
                "rtx_sha256": sha256(reference),
                "amd_sha256": sha256(candidate),
            })
        reports[str(slot)] = {
            "function": case["function"],
            "execution_gate": execution,
            "exact_state_gate": case_state,
            "full_arena_repeat_bitwise_exact": deterministic,
            "outputs": output_reports,
        }

    passed = execution_gate and state_gate and numerical_gate and determinism_gate
    return {
        "schema": 1,
        "experiment": "vit1d_slots57_98_exact_state_rtx_vs_rx9070xt",
        "status": "PASS" if passed else "FAIL",
        "classification": "SETTLED_GRAPH_EDGE_CROSS_VENDOR_NUMERICAL_PARITY",
        "counts_as_s7": False,
        "execution_gate": execution_gate,
        "exact_state_gate": state_gate,
        "settled_numerical_gate": numerical_gate,
        "amd_repeat_determinism_gate": determinism_gate,
        "settled_edge_count": settled_edges,
        "bitwise_exact_settled_edge_count": exact_edges,
        "advisories": advisories,
        "slots": reports,
        "limitations": [
            "function-isolated exact-state replay, not an integrated full frame",
            "side accumulator outputs lacking a pointer-identical consumer are diagnostic only",
        ],
        "next_gate": "settle side accumulators through decoder consumers, then slots 99-154",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
