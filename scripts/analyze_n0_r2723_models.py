#!/usr/bin/env python3
"""Enumerate FP16/fused rounding models for the zero-input N0 r2723 reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.analyze_n0_r2329_models import evaluate
    from scripts.instrument_n0_mma178_b_path_trace import TRACE_BYTES
except ModuleNotFoundError:
    from analyze_n0_r2329_models import evaluate
    from instrument_n0_mma178_b_path_trace import TRACE_BYTES


MMA_INDICES = (144, 145, 146, 147)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mma-trace", required=True, type=Path)
    parser.add_argument("--rtx-path", required=True, type=Path)
    parser.add_argument("--amd-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.mma_trace.read_bytes(), args.rtx_path.read_bytes(), args.amd_path.read_bytes(),
        mma_indices=MMA_INDICES,
        path_stage=2,
        path_trace_bytes=TRACE_BYTES,
        experiment="n0_r2723_rounding_boundary_enumeration",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "exact_rtx_models": report["exact_rtx_models"],
        "exact_amd_models": report["exact_amd_models"],
        "best_models": [{key: item[key] for key in ("name", "word_mismatches_vs_rtx", "word_mismatches_vs_amd")}
                        for item in report["best_models"][:5]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
