#!/usr/bin/env python3
"""Fit candidate square-reduction rounding models to clean RTX lane-sum traces."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


LANES = 32
STAGE_BYTES = LANES * 4
BRANCHES = (
    {"name": "a0", "inputs": (0, 2, 1, 3), "lane": 10},
    {"name": "a1", "inputs": (22, 24, 23, 25), "lane": 32},
)


def hround(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def hbits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def unpack_stage(raw: bytes, stage: int) -> list[tuple[float, float]]:
    base = stage * STAGE_BYTES
    return [struct.unpack_from("<2e", raw, base + lane * 4) for lane in range(LANES)]


def strict(x: tuple[float, ...]) -> float:
    squares = [hround(value * value) for value in x]
    return hround(hround(squares[0] + squares[1]) + hround(squares[2] + squares[3]))


def exact_once(x: tuple[float, ...]) -> float:
    return hround((x[0] * x[0] + x[1] * x[1]) + (x[2] * x[2] + x[3] * x[3]))


def f32_tree_once(x: tuple[float, ...]) -> float:
    products = [f32(value * value) for value in x]
    pair0 = f32(products[0] + products[1])
    pair1 = f32(products[2] + products[3])
    return hround(f32(pair0 + pair1))


def half_squares_sum_once(x: tuple[float, ...]) -> float:
    squares = [hround(value * value) for value in x]
    return hround((squares[0] + squares[1]) + (squares[2] + squares[3]))


def fused_pairs_half_lane(x: tuple[float, ...]) -> float:
    return hround(hround(x[0] * x[0] + x[1] * x[1]) + hround(x[2] * x[2] + x[3] * x[3]))


def hfma_pairs_half_lane(x: tuple[float, ...], separate: int) -> float:
    pairs = []
    for left, right in ((x[0], x[1]), (x[2], x[3])):
        if separate == 0:
            pairs.append(hround(left * left + hround(right * right)))
        else:
            pairs.append(hround(hround(left * left) + right * right))
    return hround(pairs[0] + pairs[1])


MODELS = {
    "strict_f16_each_op": strict,
    "exact_four_squares_round_once": exact_once,
    "f32_tree_round_once_to_f16": f32_tree_once,
    "f16_squares_fused_add_tree": half_squares_sum_once,
    "fused_pairs_then_f16_lane": fused_pairs_half_lane,
    "hfma_pair_right_square_separate": lambda x: hfma_pairs_half_lane(x, 0),
    "hfma_pair_left_square_separate": lambda x: hfma_pairs_half_lane(x, 1),
}


def analyze(trace: bytes) -> dict:
    if len(trace) % STAGE_BYTES or len(trace) < 33 * STAGE_BYTES:
        raise ValueError("invalid fusion sweep trace length")
    branches = []
    totals = {name: {"matches": 0, "samples": 0, "mismatches": []} for name in MODELS}
    for branch in BRANCHES:
        input_stages = [unpack_stage(trace, index) for index in branch["inputs"]]
        expected = unpack_stage(trace, branch["lane"])
        models = {name: {"matches": 0, "samples": 0, "mismatches": []} for name in MODELS}
        for lane in range(LANES):
            for half in range(2):
                values = tuple(stage[lane][half] for stage in input_stages)
                target = expected[lane][half]
                if not all(math.isfinite(value) for value in (*values, target)):
                    continue
                for name, model in MODELS.items():
                    predicted = model(values)
                    record = models[name]
                    record["samples"] += 1
                    totals[name]["samples"] += 1
                    if hbits(predicted) == hbits(target):
                        record["matches"] += 1
                        totals[name]["matches"] += 1
                    elif len(record["mismatches"]) < 12:
                        mismatch = {
                            "lane": lane,
                            "half": half,
                            "inputs_hex": [f"{hbits(value):04X}" for value in values],
                            "expected_hex": f"{hbits(target):04X}",
                            "predicted_hex": f"{hbits(predicted):04X}",
                        }
                        record["mismatches"].append(mismatch)
                        if len(totals[name]["mismatches"]) < 12:
                            totals[name]["mismatches"].append({"branch": branch["name"], **mismatch})
        for record in models.values():
            record["bitwise_gate"] = record["matches"] == record["samples"]
        branches.append({"name": branch["name"], "input_stage_indices": list(branch["inputs"]),
                         "lane_stage_index": branch["lane"], "models": models})
    for record in totals.values():
        record["bitwise_gate"] = record["matches"] == record["samples"]
    winners = [name for name, record in totals.items() if record["bitwise_gate"]]
    return {
        "schema": 1,
        "experiment": "n0_mma176_a_hidden_fusion_semantics_fit",
        "status": "PASS" if winners else "UNRESOLVED",
        "classification": "CLEAN_RTX_LANE_SUM_MODEL_SELECTION",
        "sample_count": sum(record["samples"] for record in totals.values()) // len(totals),
        "bitwise_exact_models": winners,
        "models": totals,
        "branches": branches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.trace.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"],
                      "bitwise_exact_models": report["bitwise_exact_models"],
                      "matches": {name: item["matches"] for name, item in report["models"].items()}},
                     separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
