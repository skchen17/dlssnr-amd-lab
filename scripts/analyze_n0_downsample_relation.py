#!/usr/bin/env python3
"""Test whether captured N0 output is the PTX 2x2 FP16 average of scratch FP8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


CHANNELS = 32


def e4m3_to_float(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    magnitude = code & 0x7F
    exponent, mantissa = magnitude >> 3, magnitude & 7
    if exponent == 0:
        value = math.ldexp(float(mantissa), -9)
    elif magnitude == 0x7F:
        value = math.nan
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return sign * value


def round_nearest_even_positive(value: float) -> int:
    base = math.floor(value)
    fraction = value - base
    if fraction > 0.5 or (fraction == 0.5 and base & 1):
        base += 1
    return base


def float_to_e4m3(value: float) -> int:
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return 0x7F
    magnitude = abs(value)
    if magnitude >= 448.0:
        return sign | 0x7E
    if magnitude < 0.015625:
        mantissa = round_nearest_even_positive(magnitude * 512.0)
        return sign | (0x08 if mantissa >= 8 else mantissa)
    fraction, exponent_plus_one = math.frexp(magnitude)
    encoded_exponent = exponent_plus_one - 1 + 7
    mantissa = round_nearest_even_positive((fraction * 2.0 - 1.0) * 8.0)
    if mantissa == 8:
        mantissa = 0
        encoded_exponent += 1
    if encoded_exponent > 15 or (encoded_exponent == 15 and mantissa >= 7):
        return sign | 0x7E
    return sign | (encoded_exponent << 3) | mantissa


def half(value: float) -> float:
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def ptx_average(codes: tuple[int, int, int, int]) -> int:
    values = [half(e4m3_to_float(code)) for code in codes]
    first = half(values[0] + values[1])
    second = half(values[2] + values[3])
    total = half(first + second)
    return float_to_e4m3(half(total * half(0.25)))


def cd_coord(lane: int, element: int) -> tuple[int, int]:
    group, thread = lane >> 2, lane & 3
    return group + (8 if element >= 2 else 0), thread * 2 + (element & 1)


def detile_scratch(scratch: bytes, height: int, width: int, transpose_spatial: bool = False) -> bytes:
    """Decode 4x4x32 tiles stored as four m16n8 D fragments."""
    logical = bytearray(len(scratch))
    tile_rows, tile_columns = height // 4, width // 4
    # Byte order emitted by rs369..rs376 after the four n8 MMA calls.
    position_to_fragment = (
        (0, 0), (0, 1), (1, 0), (1, 1),
        (0, 2), (0, 3), (1, 2), (1, 3),
        (2, 0), (2, 1), (3, 0), (3, 1),
        (2, 2), (2, 3), (3, 2), (3, 3),
    )
    for tile_y in range(tile_rows):
        for tile_x in range(tile_columns):
            tile_base = (tile_y * tile_columns + tile_x) * 512
            for lane in range(32):
                lane_base = tile_base + lane * 16
                for byte_position, (n_tile, element) in enumerate(position_to_fragment):
                    matrix_row, matrix_column = cd_coord(lane, element)
                    local_y, local_x = divmod(matrix_row, 4)
                    if transpose_spatial:
                        local_y, local_x = local_x, local_y
                    channel = n_tile * 8 + matrix_column
                    logical_index = (
                        ((tile_y * 4 + local_y) * width + tile_x * 4 + local_x) * CHANNELS
                        + channel
                    )
                    logical[logical_index] = scratch[lane_base + byte_position]
    return bytes(logical)


def unpack_output(output: bytes, height: int, width: int) -> bytes:
    """Decode PTX's two contiguous HxWx16 channel planes into HWC32."""
    logical = bytearray(len(output))
    plane_bytes = height * width * 16
    for plane in range(2):
        for y in range(height):
            for x in range(width):
                source = plane * plane_bytes + (y * width + x) * 16
                destination = (y * width + x) * CHANNELS + plane * 16
                logical[destination : destination + 16] = output[source : source + 16]
    return bytes(logical)


def compare(logical_scratch: bytes, logical_output: bytes, height: int, width: int) -> dict:
    output_height, output_width = height // 2, width // 2
    predicted = bytearray(len(logical_output))
    mismatches = 0
    absolute_code_error = 0
    first_mismatches = []
    for y in range(output_height):
        for x in range(output_width):
            top_left = ((2 * y) * width + 2 * x) * CHANNELS
            top_right = top_left + CHANNELS
            bottom_left = top_left + width * CHANNELS
            bottom_right = bottom_left + CHANNELS
            out_base = (y * output_width + x) * CHANNELS
            for channel in range(CHANNELS):
                code = ptx_average((
                    logical_scratch[top_left + channel],
                    logical_scratch[top_right + channel],
                    logical_scratch[bottom_left + channel],
                    logical_scratch[bottom_right + channel],
                ))
                index = out_base + channel
                predicted[index] = code
                actual = logical_output[index]
                if code != actual:
                    mismatches += 1
                    absolute_code_error += abs((code & 0x7F) - (actual & 0x7F))
                    if len(first_mismatches) < 16:
                        first_mismatches.append({
                            "index": index,
                            "y": y,
                            "x": x,
                            "channel": channel,
                            "predicted": code,
                            "actual": actual,
                        })
    return {
        "matches": len(logical_output) - mismatches,
        "mismatches": mismatches,
        "match_fraction": (len(logical_output) - mismatches) / len(logical_output),
        "mean_absolute_magnitude_code_error_on_mismatch": (
            absolute_code_error / mismatches if mismatches else 0.0
        ),
        "predicted_sha256": hashlib.sha256(predicted).hexdigest(),
        "first_mismatches": first_mismatches,
    }


def analyze(scratch: bytes, output: bytes, height: int, width: int) -> dict:
    expected_scratch = height * width * CHANNELS
    expected_output = (height // 2) * (width // 2) * CHANNELS
    if len(scratch) != expected_scratch or len(output) != expected_output:
        raise ValueError("raw tensor size does not match HWC envelope")
    logical_output = unpack_output(output, height // 2, width // 2)
    candidates = []
    for transpose in (False, True):
        logical_scratch = detile_scratch(scratch, height, width, transpose)
        comparison = compare(logical_scratch, logical_output, height, width)
        comparison["spatial_row_mapping"] = "column_major_4x4" if transpose else "row_major_4x4"
        candidates.append(comparison)
    best = max(candidates, key=lambda item: item["matches"])
    return {
        "schema": 1,
        "experiment": "n0_scratch_output_downsample_relation",
        "status": "PASS" if best["matches"] > expected_output * 0.9 else "FAIL",
        "classification": "CAPTURED_DATAFLOW_HYPOTHESIS",
        "counts_as_s6": False,
        "shape": {"scratch": [height, width, CHANNELS], "output": [height // 2, width // 2, CHANNELS]},
        "scratch_storage": "[H/4,W/4,lane32,byte16] m16n8 fragment tiles",
        "output_storage": "[channel_plane2,H/2,W/2,channel16]",
        "ptx_operation": "pairwise f16 add, f16 add, f16 multiply by 0.25, satfinite e4m3 encode",
        "elements": expected_output,
        "matches": best["matches"],
        "mismatches": best["mismatches"],
        "match_fraction": best["match_fraction"],
        "mean_absolute_magnitude_code_error_on_mismatch": best["mean_absolute_magnitude_code_error_on_mismatch"],
        "spatial_row_mapping": best["spatial_row_mapping"],
        "candidate_match_fractions": {
            item["spatial_row_mapping"]: item["match_fraction"] for item in candidates
        },
        "scratch_sha256": hashlib.sha256(scratch).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "predicted_logical_sha256": best["predicted_sha256"],
        "logical_output_sha256": hashlib.sha256(logical_output).hexdigest(),
        "first_mismatches": best["first_mismatches"],
        "interpretation_limit": (
            "Scratch contains already-quantized E4M3 values, whereas PTX averages the original "
            "FP16 registers before quantization; residual threshold differences are expected."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = analyze(args.scratch.read_bytes(), args.output.read_bytes(), args.height, args.width)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
