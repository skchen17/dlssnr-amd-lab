#!/usr/bin/env python3
"""Compare RTX and RX traces for the N0 r2329 reduction tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.analyze_n0_norm_path_trace import analyze as analyze_generic
    from scripts.instrument_n0_reduction_path_trace import STAGES, TRACE_BYTES
except ModuleNotFoundError:
    from analyze_n0_norm_path_trace import analyze as analyze_generic
    from instrument_n0_reduction_path_trace import STAGES, TRACE_BYTES


def analyze(rtx_path: Path, amd_path: Path) -> dict:
    report = analyze_generic(rtx_path, amd_path, stages_layout=STAGES, trace_bytes=TRACE_BYTES)
    report["experiment"] = "n0_r2329_reduction_path_rtx_vs_rx9070xt"
    report["classification"] = "N0_HIDDEN_PRECISION_FUSION_BOUNDARY_LOCALIZATION"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--rtx",required=True,type=Path); parser.add_argument("--amd",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); args=parser.parse_args()
    report=analyze(args.rtx,args.amd);args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"first_divergent_stage":report["first_divergent_stage"],"first_divergent_stage_name":report["first_divergent_stage_name"]},separators=(",",":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
