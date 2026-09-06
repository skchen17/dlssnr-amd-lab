"""Validate one reconstructed family across encoder/decoder weights and windows.

No new NVIDIA capture is required. All GPU intermediates are freshly calculated;
captured inputs are isolated-operator controls, not whole-network acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def checked(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"asset hash mismatch: {path}")
    return data


def compare(reference: bytes, output: bytes) -> dict:
    a, b = np.frombuffer(reference, dtype=np.uint8), np.frombuffer(output, dtype=np.uint8)
    if not a.size or a.size != b.size:
        raise ValueError("E4 buffers must have the same nonzero length")
    codes = np.arange(256, dtype=np.int32)
    exponent, mantissa = (codes & 127) >> 3, codes & 7
    lut = np.where(exponent == 0, mantissa * 2.**-9,
                   (1 + mantissa / 8) * 2.**(exponent - 7)) * np.where(codes & 128, -1, 1)
    nonfinite = int(np.count_nonzero((a & 127) == 127) + np.count_nonzero((b & 127) == 127))
    if nonfinite:
        return {"pass": False, "nonfinite": nonfinite}
    x, y = lut[a], lut[b]
    error = x - y
    std = float(np.std(x))
    exact = bool(np.array_equal(a, b))
    corr = float(np.corrcoef(x, y)[0, 1]) if std and np.std(y) else float(exact)
    rmse = float(np.sqrt(np.mean(error**2)))
    nrmse = rmse / std if std else (0.0 if not rmse else None)
    return {"pass": corr >= .99 and nrmse is not None and nrmse <= .1,
            "correlation": corr, "nrmse": nrmse, "byte_mismatches": int(np.count_nonzero(a != b)),
            "mae": float(np.mean(abs(error))), "max_error": float(np.max(abs(error))), "nonfinite": 0}


def run(output: Path, executable: Path) -> dict:
    payload = Path("deliverables/swin_slots3_5_reference_20260831_153128/payload")
    encoder = Path("results/20260831_153542_rtx5070_swin_slots3_5/_swin_slots3_5_reference_result_20260831_153542")
    decoder = Path("results/20260831_230000_decoder_full_graph_exact_state/cases")
    enc_manifest = json.loads((encoder / "manifest.json").read_text(encoding="utf-8-sig"))
    dec_manifest = json.loads((decoder / "manifest.json").read_text())
    model = checked(decoder / "model_arena.raw", dec_manifest["model_arena"]["sha256"])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for slot in (3, 4, 151, 152):
        case_dir = output / f"slot{slot}"
        case_dir.mkdir()
        if slot < 5:
            hashes = enc_manifest["payload_sha256"]
            inputs = checked(payload / f"slot{slot}_input.raw", hashes[f"slot{slot}_input.raw"])
            weights = checked(payload / f"slot{slot}_weights.raw", hashes[f"slot{slot}_weights.raw"])[:20672]
            params = checked(payload / f"slot{slot}_params.raw", hashes[f"slot{slot}_params.raw"])
            reference_hash = next(s for s in enc_manifest["slots"] if s["slot"] == slot)["output_sha256"]
            reference_path = encoder / f"slot{slot}/output.raw"
        else:
            spec = next(s for s in dec_manifest["slots"] if s["slot"] == slot)
            source = decoder / f"slot{slot}"
            arena = checked(source / spec["activation_arena"]["path"], spec["activation_arena"]["sha256"])
            params = checked(source / spec["params"]["path"], spec["params"]["sha256"])
            height, width = struct.unpack_from("<2i", params, 24)
            offset = next(v["arena_offset"] for v in spec["activation_param_views"] if v["param_offset"] == 0)
            inputs = arena[offset:offset + width * height * 32]
            offset = spec["weight_param_views"][0]["weight_offset"]
            weights = model[offset:offset + 20672]
            asset = spec["outputs"][0]["asset"]
            reference_path, reference_hash = source / asset["path"], asset["sha256"]
        height, width, ox, oy = struct.unpack_from("<4i", params, 24)
        reference = checked(reference_path, reference_hash)
        (case_dir / "input.e4").write_bytes(inputs)
        (case_dir / "weights.raw").write_bytes(weights)
        subprocess.run([str(executable.resolve()), str(case_dir / "input.e4"), str(case_dir / "weights.raw"),
                        str(width), str(height), str(ox), str(oy), str(case_dir)], check=True)
        inference = json.loads((case_dir / "manifest.json").read_text())
        actual = (case_dir / "output.e4").read_bytes()
        records.append({"slot": slot, "geometry": [width, height, ox, oy],
                        "input_sha256": digest(inputs), "weights_sha256": digest(weights),
                        "reference": str(reference_path.resolve()), "reference_sha256": reference_hash,
                        "output_sha256": digest(actual), "inference": inference,
                        "same_input_comparison": compare(reference, actual)})
    report = {"experiment": "swin1h_native_family_four_weight_window_cases",
              "status": "PASS" if all(r["same_input_comparison"]["pass"] for r in records) else "NUMERICAL_GATE_FAIL",
              "slots": records, "same_list_per_operator": True,
              "captured_operator_inputs": True, "intermediate_bitwise_parity_required": False,
              "native_graph_complete": False, "counts_as_s7": False, "game_runtime_ready": False}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=Path("build/swin1h_d3d12.exe"))
    args = parser.parse_args()
    result = run(args.output, args.executable)
    print(result["status"])
    for item in result["slots"]:
        print(item["slot"], item["same_input_comparison"])
    raise SystemExit(0 if result["status"] == "PASS" else 1)
