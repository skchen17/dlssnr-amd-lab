#!/usr/bin/env python3
"""Compare the RTX copy-kernel input/output snapshots bytewise and numerically."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


FORMATS = {
    10: ("R16G16B16A16_FLOAT", "e", 2),
    2: ("R32G32B32A32_FLOAT", "f", 4),
}


def analyze(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    base = metadata_path.parent
    input_path = base / metadata["input_file"]
    output_path = base / metadata["output_file"]
    input_data = input_path.read_bytes()
    output_data = output_path.read_bytes()
    fmt = int(metadata["format"])
    if fmt not in FORMATS:
        raise ValueError(f"unsupported DXGI format {fmt}")
    format_name, scalar_code, scalar_bytes = FORMATS[fmt]
    expected_bytes = int(metadata["width"]) * int(metadata["height"]) * 4 * scalar_bytes
    if len(input_data) != expected_bytes or len(output_data) != expected_bytes:
        raise ValueError(
            f"snapshot size mismatch: expected={expected_bytes}, "
            f"input={len(input_data)}, output={len(output_data)}"
        )

    byte_mismatches = sum(a != b for a, b in zip(input_data, output_data))
    scalar_count = expected_bytes // scalar_bytes
    input_values = struct.iter_unpack("<" + scalar_code, input_data)
    output_values = struct.iter_unpack("<" + scalar_code, output_data)
    scalar_mismatches = 0
    finite_pairs = 0
    nonfinite_pairs = 0
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    for index, (a_tuple, b_tuple) in enumerate(zip(input_values, output_values)):
        a, b = a_tuple[0], b_tuple[0]
        if math.isfinite(a) and math.isfinite(b):
            error = abs(a - b)
            finite_pairs += 1
            sum_abs += error
            sum_sq += error * error
            max_abs = max(max_abs, error)
            if a != b:
                scalar_mismatches += 1
        else:
            nonfinite_pairs += 1
            offset = index * scalar_bytes
            if input_data[offset : offset + scalar_bytes] != output_data[offset : offset + scalar_bytes]:
                # The authoritative non-finite comparison remains the raw bits.
                scalar_mismatches += 1
    mean_abs = sum_abs / finite_pairs if finite_pairs else None
    mse = sum_sq / finite_pairs if finite_pairs else None
    psnr_1 = "inf" if mse == 0 else (10.0 * math.log10(1.0 / mse) if mse else None)
    bitwise_equal = input_data == output_data
    status = "PASS" if metadata.get("status") == "PASS" and bitwise_equal else "FAIL"
    return {
        "schema": 1,
        "experiment": "copy_snapshot_numeric_comparison",
        "status": status,
        "source_snapshot_status": metadata.get("status"),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "dxgi_format": fmt,
        "format_name": format_name,
        "expected_bytes_per_file": expected_bytes,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "output_sha256": hashlib.sha256(output_data).hexdigest(),
        "bitwise_equal": bitwise_equal,
        "byte_mismatches": byte_mismatches,
        "scalar_count": scalar_count,
        "scalar_mismatches": scalar_mismatches,
        "finite_pairs": finite_pairs,
        "nonfinite_pairs": nonfinite_pairs,
        "max_abs_error": max_abs if finite_pairs else None,
        "mean_abs_error": mean_abs,
        "mse": mse,
        "psnr_peak_1_db": psnr_1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.metadata)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
