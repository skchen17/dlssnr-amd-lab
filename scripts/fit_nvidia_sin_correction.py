#!/usr/bin/env python3
"""Fit a segmented correction from the RX sine SFU to the RTX 5070 SFU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.fit_nvidia_cos_correction import fit
except ModuleNotFoundError:
    from fit_nvidia_cos_correction import fit


def fit_sine(rtx: Path, amd: Path, segment_bits: int, degree: int) -> dict:
    model = fit(rtx, amd, segment_bits, degree, [])
    model["experiment"] = (
        f"nvidia_sm120_sin_rx_correction_{model['segments']}_segment_degree{degree}"
    )
    model["classification"] = "RTX5070_PRODUCTION_DOMAIN_EMPIRICAL_SIN_SFU_CORRECTION"
    model["target_site"] = "first_n0_sine_r228_r226"
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-bits", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()
    model = fit_sine(args.rtx, args.amd, args.segment_bits, args.degree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in (
        "status", "segments", "degree", "sample_count", "fp32_exact_fraction",
        "fp16_mismatch_samples")}, separators=(",", ":")))
    return 0 if model["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
