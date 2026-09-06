#!/usr/bin/env python3
"""Compare RTX/RX immediate post-definition Box-Muller traces."""

from __future__ import annotations

import hashlib
import struct

try:
    from scripts.instrument_n0_box_muller_cta_trace import SAMPLES, STAGES, TRACE_BYTES
except ModuleNotFoundError:
    from instrument_n0_box_muller_cta_trace import SAMPLES, STAGES, TRACE_BYTES


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def analyze_bytes(rtx: bytes, amd: bytes, stages=STAGES) -> dict:
    trace_bytes = SAMPLES * len(stages) * 4
    if len(rtx) != trace_bytes or len(amd) != trace_bytes:
        raise ValueError(f"trace size must be {trace_bytes} bytes")
    reports = []; first_stage = None
    for stage, (name, _, _, kind) in enumerate(stages):
        mismatches = []; base = stage * SAMPLES * 4
        for sample in range(SAMPLES):
            offset = base + sample * 4; rb = rtx[offset:offset+4]; ab = amd[offset:offset+4]
            if rb != ab:
                item = {"sample": sample, "rtx_hex": rb.hex().upper(), "amd_hex": ab.hex().upper()}
                if kind == "f32":
                    item.update(rtx_value=struct.unpack("<f", rb)[0], amd_value=struct.unpack("<f", ab)[0])
                else:
                    item.update(rtx_value=int.from_bytes(rb, "little"), amd_value=int.from_bytes(ab, "little"))
                mismatches.append(item)
        if mismatches and first_stage is None:
            first_stage = name
        reports.append({"stage": stage, "name": name, "kind": kind,
                        "mismatch_samples": len(mismatches), "first_mismatches": mismatches[:16]})
    return {"schema": 1, "experiment": "n0_box_muller_cta_rtx_vs_rx9070xt",
            "classification": "FIRST_APPROX_MATH_DIVERGENCE_LOCALIZATION",
            "rtx_sha256": sha256(rtx), "amd_sha256": sha256(amd),
            "sample_count": SAMPLES, "stage_count": len(stages),
            "first_mismatch_stage": first_stage, "per_stage": reports}
