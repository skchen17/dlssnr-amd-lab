"""Reproducible end-to-end diagnostic using AMD-generated upstream activations.

The upstream runner still uses ZLUDA; this does not claim native-graph completion.
Unlike isolated head tests, no RTX activation tensor feeds the output head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

FRAME1_SHA256 = "8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def compare_rgb(reference: Path, candidate: Path) -> dict:
    ref = np.fromfile(reference, dtype="<f2")
    out = np.fromfile(candidate, dtype="<f2")
    if ref.size != 640 * 360 * 4 or out.size != ref.size:
        raise ValueError("expected two 640x360 RGBA16F frames")
    ref = ref.reshape(-1, 4).astype(np.float64)
    out = out.reshape(-1, 4).astype(np.float64)
    a, b = ref[:, :3], out[:, :3]
    nonfinite = int(np.count_nonzero(~np.isfinite(ref)) + np.count_nonzero(~np.isfinite(out)))
    if nonfinite:
        return {"pass": False, "nonfinite_values": nonfinite}
    error = a - b
    mse = float(np.mean(error * error))
    deviation = float(np.std(a))
    correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if deviation and np.std(b) else 0.0
    nrmse = float(np.sqrt(mse) / deviation) if deviation else (0.0 if mse == 0 else None)
    alpha_exact = bool(np.array_equal(ref[:, 3], out[:, 3]))
    return {
        "normative_channels": "RGB", "nonfinite_values": nonfinite,
        "correlation": correlation, "nrmse": nrmse,
        "mae": float(np.mean(abs(error))), "max_error": float(np.max(abs(error))),
        "unit_range_psnr_db": float(-10 * np.log10(mse)) if mse else None,
        "reference_rgb_max": float(np.max(a)), "reference_rgb_stddev": deviation,
        "alpha_exact": alpha_exact,
        "pass": correlation >= 0.99 and nrmse is not None and nrmse <= 0.1 and alpha_exact,
        "note": "Unit-range PSNR alone is not acceptance for this near-black capture.",
    }


def validate(upstream: Path, reference: Path, model: Path, executable: Path, output: Path) -> dict:
    execution = json.loads((upstream / "execution.json").read_text(encoding="utf-8-sig"))
    if execution.get('color_input', {}).get('external'):
        raise ValueError("external color requires a matching RTX reference and base; zero-input validator is inapplicable")
    if digest(reference) != FRAME1_SHA256:
        raise ValueError("reference is not the accepted queue-complete frame1 output")
    if not execution.get("pass") or execution.get("rtx_intermediate_state_injection") or execution.get("diagnostic_injections"):
        raise ValueError("requires successful injection-free upstream execution")
    if len(execution.get("runs", [])) < 2:
        raise ValueError("two upstream repeats are required")
    model_hash = digest(model)
    upstream_model = execution.get("model_sha256")
    if not upstream_model:
        # Historical reports predate explicit model hashes; the upstream plan
        # carries the hash already verified by that runner before execution.
        upstream_model = json.loads(Path(execution["plan"]).read_text(encoding="utf-8-sig"))["model_arena"]["sha256"]
    if model_hash != upstream_model:
        raise ValueError("head and upstream model hashes differ")
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for run in execution["runs"]:
        state = run.get("pre_head_activation")
        if [s["slot"] for s in run.get("slots", []) if s.get("executed")] != list(range(156)):
            raise ValueError("upstream report must contain all 156 executed slots")
        if run.get("diagnostic_injections"):
            raise ValueError("run contains injected intermediates")
        if not state or state.get("producer") != "AMD_slots_0_153":
            raise ValueError("missing AMD pre-head export; run with --export-pre-head")
        activation = Path(state["path"])
        if digest(activation) != state["sha256"] or activation.stat().st_size != state["bytes"]:
            raise ValueError("pre-head export hash/size mismatch")
        index = int(run["run"])
        frame = output / f"run{index}_rgba16f.raw"
        manifest = output / f"run{index}_inference.json"
        subprocess.run([str(executable.resolve()), str(activation.resolve()), str(model.resolve()), "-", str(frame.resolve()), str(manifest.resolve())], check=True)
        inference = json.loads(manifest.read_text())
        runs.append({"run": index, "activation_sha256": state["sha256"],
                     "output_sha256": digest(frame), "inference": inference,
                     "same_frame_rtx_comparison": compare_rgb(reference, frame)})
    repeat = len({r["output_sha256"] for r in runs}) == 1
    image_gate = all(r["same_frame_rtx_comparison"]["pass"] for r in runs)
    report = {
        "experiment": "amd_upstream_to_reconstructed_d3d12_head",
        "status": "PASS" if repeat and image_gate else "IMAGE_GATE_FAIL",
        "reference": str(reference.resolve()), "reference_sha256": FRAME1_SHA256,
        "upstream_execution": str((upstream / "execution.json").resolve()),
        "model_sha256": model_hash,
        "rtx_intermediate_state_injection": False, "repeat_exact": repeat,
        "image_gate": image_gate, "runs": runs,
        "upstream_backend": "ZLUDA_WITH_NATIVE_SWIN_DXIL" if execution.get("native_swin_enabled") else "ZLUDA_PTX_TRANSLATION",
        "diagnostic_cpu_handoff": bool(execution.get("diagnostic_cpu_handoff")),
        "head_backend": "D3D12_DXIL",
        "native_graph_complete": False, "same_list_full_graph": False,
        "counts_as_s7": False, "game_runtime_ready": False,
        "scope": "Offline zero-input single-frame diagnostic; not real-scene quality or temporal validation.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/20260904_690000_deferred_frame1_output_rtx5070/copy_input.raw"))
    parser.add_argument("--model", type=Path, default=Path("local_models/decoded_310_8/model_arena.raw"))
    parser.add_argument("--executable", type=Path, default=Path("build/output_head_infer_d3d12.exe"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.upstream, args.reference, args.model, args.executable, args.output)
    print(json.dumps({k: report[k] for k in ("status", "repeat_exact", "image_gate", "native_graph_complete")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
