#!/usr/bin/env python3
"""Summarize same-capture full-graph RTX-state injection sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze(native_path: Path, injection_paths: dict[int, Path]) -> dict:
    native = _read(native_path)
    injections = {slot: _read(path) for slot, path in injection_paths.items()}
    if native.get("evidence", {}).get("rtx_intermediate_state_injection") is not False:
        raise ValueError("native analysis is not an injection-free execution")
    if native.get("execution_gate") is not True:
        raise ValueError("native execution gate did not pass")
    for slot, report in injections.items():
        if report.get("evidence", {}).get("rtx_intermediate_state_injection") is not True:
            raise ValueError(f"slot {slot} analysis is not an injection execution")
        if report.get("counts_as_s7") is not False:
            raise ValueError(f"slot {slot} injection must not count as S7")
        if report.get("amd_repeat_determinism_gate") is not True:
            raise ValueError(f"slot {slot} injection is not repeat deterministic")

    boundary_slots = sorted(int(slot) for slot in native["boundaries"])
    rows = {}
    for slot in boundary_slots:
        key = str(slot)
        baseline = native["boundaries"][key]
        base_nrmse = baseline["comparison"].get("nrmse_vs_rtx_stddev")
        row = {
            "tensor_type": baseline["tensor_type"],
            "rtx_sha256": baseline["rtx_sha256"],
            "native_nrmse": base_nrmse,
            "native_pass": baseline["comparison"]["parity_gate"]["pass"],
            "injections": {},
        }
        for injected_after, report in injections.items():
            candidate = report["boundaries"][key]
            if candidate["rtx_sha256"] != baseline["rtx_sha256"]:
                raise ValueError(
                    f"slot {slot} RTX reference differs in injection-after-{injected_after}"
                )
            nrmse = candidate["comparison"].get("nrmse_vs_rtx_stddev")
            row["injections"][str(injected_after)] = {
                "nrmse": nrmse,
                "delta_vs_native": None if nrmse is None or base_nrmse is None else nrmse - base_nrmse,
                "pass": candidate["comparison"]["parity_gate"]["pass"],
            }
        rows[key] = row

    failed = [slot for slot in boundary_slots if not rows[str(slot)]["native_pass"]]
    final = {
        "native_nrmse": native["final_comparison"]["nrmse_vs_rtx_stddev"],
        "native_pass": native["final_rgba16f_image_gate"],
        "injections": {
            str(slot): {
                "nrmse": report["final_comparison"]["nrmse_vs_rtx_stddev"],
                "pass": report["final_rgba16f_image_gate"],
            }
            for slot, report in injections.items()
        },
    }
    inject5_closes_2h = 5 in injections and all(
        rows[str(slot)]["injections"]["5"]["pass"] for slot in (6, 7, 8)
    )
    return {
        "schema": 1,
        "experiment": "full_graph_same_capture_rtx_state_injection_sensitivity",
        "status": "PASS",
        "classification": "CAUSAL_DIAGNOSTIC_WITH_RTX_STATE_INJECTION_NOT_S7",
        "counts_as_s7": False,
        "native_first_strict_failure": failed[0] if failed else None,
        "boundaries": rows,
        "final": final,
        "findings": {
            "inject_after_slot5_makes_slots6_8_pass": inject5_closes_2h,
            "slot6_8_failure_is_upstream_state_sensitive": inject5_closes_2h,
            "injection_results_are_never_deployment_evidence": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--injection", action="append", default=[],
                        help="SLOT=analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    injections = {}
    for value in args.injection:
        slot_text, separator, path_text = value.partition("=")
        if not separator:
            parser.error("--injection must be SLOT=analysis.json")
        slot = int(slot_text)
        if slot in injections:
            parser.error(f"duplicate injection slot {slot}")
        injections[slot] = Path(path_text).resolve(strict=True)
    report = analyze(args.native.resolve(strict=True), injections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "native_first_strict_failure": report["native_first_strict_failure"],
        "inject_after_slot5_makes_slots6_8_pass":
            report["findings"]["inject_after_slot5_makes_slots6_8_pass"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
