#!/usr/bin/env python3
"""Verify an exact-state post-block store trace against direct output and oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.analyze_full_graph_integrated import compare_rgba16f_rgb
    from scripts.analyze_postblock_store_trace import reconstruct_rgba16f
except ModuleNotFoundError:
    from analyze_full_graph_integrated import compare_rgba16f_rgb
    from analyze_postblock_store_trace import reconstruct_rgba16f


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True,
                        help="640x384 padded linear output")
    parser.add_argument("--reference", type=Path, required=True,
                        help="640x360 complete RTX oracle")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = args.trace.read_bytes()
    reconstructed, layout = reconstruct_rgba16f(trace)
    reference = args.reference.read_bytes()
    direct = args.direct.read_bytes()[:len(reference)]
    visible = reconstructed[:len(reference)]
    report = {
        "schema": 1,
        "experiment": "postblock_exact_state_pre_store_vs_complete_oracle",
        "status": "PASS" if visible == direct else "FAIL",
        "classification": "POSTBLOCK_DIVERGENCE_PRECEDES_SURFACE_LOWERING",
        "trace_sha256": sha256(trace),
        "trace_nonzero_bytes": sum(value != 0 for value in trace),
        "layout": layout,
        "reconstructed_visible_sha256": sha256(visible),
        "direct_visible_sha256": sha256(direct),
        "reconstructed_equals_direct": visible == direct,
        "reference_sha256": sha256(reference),
        "bitwise_exact_to_reference": visible == reference,
        "comparison": compare_rgba16f_rgb(reference, visible),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
