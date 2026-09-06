#!/usr/bin/env python3
"""Reconstruct N0's PTX downsample epilogue directly from captured tiled scratch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


DEPENDENCY = Path(__file__).resolve().parent / "analyze_n0_downsample_relation.py"
SPEC = importlib.util.spec_from_file_location("n0_downsample_relation", DEPENDENCY)
RELATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(RELATION)


def pair(scratch: bytes, tile: int, lane: int, position: int) -> tuple[int, int]:
    offset = tile * 512 + lane * 16 + position
    return scratch[offset], scratch[offset + 1]


def average_pairs(values: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return tuple(
        RELATION.ptx_average(tuple(value[component] for value in values))
        for component in range(2)
    )


def warp_average(
    scratch: bytes, tile_a: int, tile_b: int, first_position: int, second_position: int
) -> list[tuple[int, int]]:
    selected = []
    for lane in range(32):
        a0 = pair(scratch, tile_a, lane, first_position)
        b0 = pair(scratch, tile_b, lane, first_position)
        a1 = pair(scratch, tile_a, lane, second_position)
        b1 = pair(scratch, tile_b, lane, second_position)
        first, second, third, fourth = (a0, b0, a1, b1) if not lane & 4 else (b0, a0, b1, a1)
        selected.append(
            (first, second, third, fourth)
            if not lane & 16
            else (third, fourth, first, second)
        )
    result = []
    for lane in range(32):
        source0 = (lane & 19) | ((lane >> 1) & 4) | ((lane << 1) & 8)
        result.append(average_pairs((
            selected[source0][0],
            selected[source0 ^ 4][1],
            selected[source0 ^ 16][2],
            selected[source0 ^ 20][3],
        )))
    return result


def reconstruct(scratch: bytes, height: int, width: int) -> bytes:
    if height % 8 or width % 8 or len(scratch) != height * width * 32:
        raise ValueError("scratch must be an HxWx32 envelope with dimensions divisible by 8")
    output_height, output_width = height // 2, width // 2
    output = bytearray(output_height * output_width * 32)
    tile_columns = width // 4
    plane_bytes = output_height * output_width * 16
    for cta_y in range(height // 8):
        for cta_x in range(width // 8):
            tile_a = (cta_y * 2) * tile_columns + cta_x * 2
            tile_b = tile_a + 1
            tile_c = tile_a + tile_columns
            tile_d = tile_c + 1
            top_low0 = warp_average(scratch, tile_a, tile_b, 0, 4)
            top_low1 = warp_average(scratch, tile_a, tile_b, 2, 6)
            bottom_low0 = warp_average(scratch, tile_c, tile_d, 0, 4)
            bottom_low1 = warp_average(scratch, tile_c, tile_d, 2, 6)
            top_high0 = warp_average(scratch, tile_a, tile_b, 8, 12)
            top_high1 = warp_average(scratch, tile_a, tile_b, 10, 14)
            bottom_high0 = warp_average(scratch, tile_c, tile_d, 8, 12)
            bottom_high1 = warp_average(scratch, tile_c, tile_d, 10, 14)
            groups = (
                (0, top_low0, top_low1),
                (2, bottom_low0, bottom_low1),
                (0, top_high0, top_high1),
                (2, bottom_high0, bottom_high1),
            )
            for group_index, (y_delta, values0, values1) in enumerate(groups):
                plane_offset = plane_bytes if group_index >= 2 else 0
                for lane in range(32):
                    y = cta_y * 4 + y_delta + lane // 16
                    x = cta_x * 4 + (lane // 4) % 4
                    channel_group = lane % 4
                    destination = plane_offset + (y * output_width + x) * 16 + channel_group * 4
                    output[destination : destination + 2] = bytes(values0[lane])
                    output[destination + 2 : destination + 4] = bytes(values1[lane])
    return bytes(output)


def analyze(scratch: bytes, output: bytes, height: int, width: int) -> dict:
    predicted = reconstruct(scratch, height, width)
    if len(predicted) != len(output):
        raise ValueError("output size does not match reconstructed envelope")
    positions = [index for index, values in enumerate(zip(predicted, output)) if values[0] != values[1]]
    matches = len(output) - len(positions)
    adjacent = sum(
        (predicted[index] >> 7) == (output[index] >> 7)
        and abs((predicted[index] & 0x7F) - (output[index] & 0x7F)) == 1
        for index in positions
    )
    exact_or_adjacent = matches + adjacent
    tile_row_bytes = (width // 4) * 512
    leading_zero_tile_rows = 0
    while (
        (leading_zero_tile_rows + 1) * tile_row_bytes <= len(scratch)
        and not any(scratch[leading_zero_tile_rows * tile_row_bytes : (leading_zero_tile_rows + 1) * tile_row_bytes])
    ):
        leading_zero_tile_rows += 1
    affected_output_rows = leading_zero_tile_rows * 2
    plane_bytes = (height // 2) * (width // 2) * 16
    row_bytes = (width // 2) * 16
    interior_indices = [
        index for index in range(len(output))
        if (index % plane_bytes) // row_bytes >= affected_output_rows
    ]
    interior_matches = sum(predicted[index] == output[index] for index in interior_indices)
    interior_exact_or_adjacent = sum(
        predicted[index] == output[index]
        or (
            (predicted[index] >> 7) == (output[index] >> 7)
            and abs((predicted[index] & 0x7F) - (output[index] & 0x7F)) == 1
        )
        for index in interior_indices
    )
    overall_tolerant_fraction = exact_or_adjacent / len(output)
    interior_tolerant_fraction = interior_exact_or_adjacent / len(interior_indices)
    return {
        "schema": 1,
        "experiment": "n0_tiled_downsample_epilogue",
        "status": "PASS" if overall_tolerant_fraction > 0.9 and interior_tolerant_fraction > 0.95 else "FAIL",
        "classification": "CAPTURED_PTX_DATAFLOW_RECONSTRUCTION",
        "counts_as_s6": False,
        "scratch_storage": "[H/4,W/4,lane32,byte16]",
        "output_storage": "[channel_plane2,H/2,W/2,channel16]",
        "elements": len(output),
        "matches": matches,
        "mismatches": len(positions),
        "match_fraction": matches / len(output),
        "adjacent_same_sign_e4m3_codes": adjacent,
        "exact_or_adjacent_fraction": overall_tolerant_fraction,
        "leading_zero_scratch_tile_rows": leading_zero_tile_rows,
        "capture_affected_output_rows_per_plane": affected_output_rows,
        "interior_elements": len(interior_indices),
        "interior_exact_matches": interior_matches,
        "interior_exact_fraction": interior_matches / len(interior_indices),
        "interior_exact_or_adjacent_fraction": interior_tolerant_fraction,
        "first_mismatch_offsets": positions[:32],
        "scratch_sha256": hashlib.sha256(scratch).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "predicted_sha256": hashlib.sha256(predicted).hexdigest(),
        "interpretation_limit": (
            "The capture stores quantized E4M3 scratch while PTX averages pre-quantization FP16 "
            "registers; one same-sign E4M3 code of tolerance is the declared quantization gate. "
            "The leading all-zero scratch tile rows are reported separately rather than hidden."
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
