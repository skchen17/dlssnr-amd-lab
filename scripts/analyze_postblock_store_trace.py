#!/usr/bin/env python3
"""Compare RTX/RX post-block values immediately before both surface stores."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


RECORD_BYTES = 32
SITES = 2
THREADS_PER_SITE = 127_008


def reconstruct_rgba16f(trace: bytes, *, threads_per_site: int = THREADS_PER_SITE,
                        sites: int = SITES, record_bytes: int = RECORD_BYTES,
                        width: int = 640, height: int = 384) -> tuple[bytes, dict]:
    expected = threads_per_site * sites * record_bytes
    if len(trace) != expected:
        raise ValueError(f"post-block trace size differs: expected {expected}")
    output = bytearray(width * height * 8)
    visited: set[tuple[int, int]] = set()
    active = 0
    for site in range(sites):
        base = site * threads_per_site * record_bytes
        for index in range(threads_per_site):
            offset = base + index * record_bytes
            if struct.unpack_from("<I", trace, offset + 24)[0] == 0:
                continue
            x, y = struct.unpack_from("<II", trace, offset + 16)
            if x >= width or y >= height:
                raise ValueError(f"surface coordinate out of range: {x},{y}")
            if (x, y) in visited:
                raise ValueError(f"duplicate surface coordinate: {x},{y}")
            values = struct.unpack_from("<4f", trace, offset)
            struct.pack_into("<4e", output, (y * width + x) * 8, *values)
            visited.add((x, y))
            active += 1
    return bytes(output), {
        "active_records": active,
        "unique_coordinates": len(visited),
        "expected_pixels": width * height,
        "complete_surface": len(visited) == width * height,
    }


def ordered_float_bits(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


def analyze_bytes(rtx: bytes, amd: bytes, *, threads_per_site: int = THREADS_PER_SITE,
                  sites: int = SITES, record_bytes: int = RECORD_BYTES) -> dict:
    expected = threads_per_site * sites * record_bytes
    if len(rtx) != expected or len(amd) != expected:
        raise ValueError(f"post-block trace size differs: expected {expected}")
    first = None
    site_reports = []
    all_bitwise = True
    all_numeric = True
    for site in range(sites):
        active_rtx = active_amd = marker_mismatches = coordinate_mismatches = 0
        stats = [{
            "n": 0, "exact": 0, "nonfinite": 0, "sum_r": 0.0, "sum_a": 0.0,
            "sum_rr": 0.0, "sum_aa": 0.0, "sum_ra": 0.0, "sum_diff2": 0.0,
            "max_ulp": 0,
        } for _ in range(4)]
        examples = []
        base = site * threads_per_site * record_bytes
        for index in range(threads_per_site):
            offset = base + index * record_bytes
            rtx_marker = struct.unpack_from("<I", rtx, offset + 24)[0]
            amd_marker = struct.unpack_from("<I", amd, offset + 24)[0]
            active_rtx += rtx_marker != 0
            active_amd += amd_marker != 0
            if rtx_marker != amd_marker:
                marker_mismatches += 1
                all_bitwise = False
                if first is None:
                    first = {"site": site, "record_index": index, "kind": "marker",
                             "rtx": rtx_marker, "amd": amd_marker}
                continue
            if rtx_marker == 0:
                continue
            rtx_x, rtx_y = struct.unpack_from("<II", rtx, offset + 16)
            amd_x, amd_y = struct.unpack_from("<II", amd, offset + 16)
            if (rtx_x, rtx_y) != (amd_x, amd_y):
                coordinate_mismatches += 1
                all_bitwise = False
                if first is None:
                    first = {"site": site, "record_index": index, "kind": "coordinate",
                             "rtx": [rtx_x, rtx_y], "amd": [amd_x, amd_y]}
            rtx_bits = struct.unpack_from("<4I", rtx, offset)
            amd_bits = struct.unpack_from("<4I", amd, offset)
            rtx_values = struct.unpack_from("<4f", rtx, offset)
            amd_values = struct.unpack_from("<4f", amd, offset)
            for component, (rb, ab, rv, av) in enumerate(
                zip(rtx_bits, amd_bits, rtx_values, amd_values)
            ):
                stat = stats[component]
                stat["n"] += 1
                stat["exact"] += rb == ab
                if rb != ab:
                    all_bitwise = False
                    ulp = abs(ordered_float_bits(rb) - ordered_float_bits(ab))
                    stat["max_ulp"] = max(stat["max_ulp"], ulp)
                    example = {
                        "site": site,
                        "record_index": index,
                        "cta": [(index // 32) % 81, (index // 32) // 81, 0],
                        "lane": index % 32,
                        "coordinate": [rtx_x, rtx_y],
                        "component": component,
                        "rtx_bits": f"0x{rb:08X}",
                        "amd_bits": f"0x{ab:08X}",
                        "rtx_value": rv,
                        "amd_value": av,
                        "ulp_distance": ulp,
                    }
                    if len(examples) < 16:
                        examples.append(example)
                    if first is None:
                        first = example | {"kind": "value"}
                if not math.isfinite(rv) or not math.isfinite(av):
                    stat["nonfinite"] += 1
                    continue
                stat["sum_r"] += rv
                stat["sum_a"] += av
                stat["sum_rr"] += rv * rv
                stat["sum_aa"] += av * av
                stat["sum_ra"] += rv * av
                stat["sum_diff2"] += (av - rv) ** 2

        component_reports = []
        for component, stat in enumerate(stats):
            n = stat["n"] - stat["nonfinite"]
            if n:
                mean_r = stat["sum_r"] / n
                mean_a = stat["sum_a"] / n
                variance_r = max(0.0, stat["sum_rr"] / n - mean_r * mean_r)
                variance_a = max(0.0, stat["sum_aa"] / n - mean_a * mean_a)
                covariance = stat["sum_ra"] / n - mean_r * mean_a
                correlation = covariance / math.sqrt(variance_r * variance_a) \
                    if variance_r and variance_a else (1.0 if stat["exact"] == stat["n"] else 0.0)
                rmse = math.sqrt(stat["sum_diff2"] / n)
                nrmse = rmse / math.sqrt(variance_r) if variance_r else (0.0 if rmse == 0 else math.inf)
            else:
                correlation, rmse, nrmse = 0.0, math.inf, math.inf
            passed = stat["nonfinite"] == 0 and correlation >= 0.99 and nrmse <= 0.1
            all_numeric &= passed
            component_reports.append({
                "component": component,
                "samples": stat["n"],
                "exact_fraction": stat["exact"] / stat["n"] if stat["n"] else 0.0,
                "nonfinite_pairs": stat["nonfinite"],
                "pearson_correlation": correlation,
                "rmse": rmse,
                "nrmse_vs_rtx_stddev": nrmse,
                "max_ulp_distance": stat["max_ulp"],
                "parity_gate": passed,
            })
        site_reports.append({
            "site": site,
            "active_records_rtx": active_rtx,
            "active_records_amd": active_amd,
            "marker_mismatches": marker_mismatches,
            "coordinate_mismatches": coordinate_mismatches,
            "components": component_reports,
            "mismatch_examples": examples,
        })
        all_numeric &= marker_mismatches == 0 and coordinate_mismatches == 0
    return {
        "schema": 1,
        "experiment": "postblock_pre_surface_store_rtx_vs_rx9070xt",
        "status": "PASS" if all_numeric else "FAIL",
        "classification": "POSTBLOCK_RGB_FIRST_DIVERGENCE_LOCALIZATION",
        "trace_layout_gate": True,
        "pre_surface_bitwise_gate": all_bitwise,
        "pre_surface_numerical_gate": all_numeric,
        "first_mismatch": first,
        "sites": site_reports,
        "interpretation": (
            "transformed RTX/RX pre-surface values differ; localize earlier post-block arithmetic"
            if not all_numeric else
            "transformed RTX/RX pre-surface values pass; any remaining transformed-output "
            "difference is after these f32 values, while original-capture parity still requires "
            "an independent resource-lowering control"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--amd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_bytes(args.rtx.read_bytes(), args.amd.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
