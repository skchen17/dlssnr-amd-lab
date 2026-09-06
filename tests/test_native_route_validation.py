import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


family = load("run_swin1h_native_validation")
frame = load("validate_reconstructed_head_full_frame")


def test_e4_exact_and_nonfinite():
    assert family.compare(bytes([0, 56, 184, 64]), bytes([0, 56, 184, 64]))["pass"]
    assert family.compare(bytes(4), bytes(4))["pass"]
    assert not family.compare(bytes([127]), bytes([127]))["pass"]
    assert not family.compare(bytes([0, 1]), bytes([56, 64]))["pass"]


def test_e4_shape_and_hash_validation(tmp_path):
    with pytest.raises(ValueError):
        family.compare(b"", b"")
    with pytest.raises(ValueError):
        family.compare(b"a", b"ab")
    asset = tmp_path / "weights.raw"
    asset.write_bytes(b"weights")
    assert family.checked(asset, family.digest(b"weights")) == b"weights"
    with pytest.raises(ValueError, match="hash mismatch"):
        family.checked(asset, family.digest(b"other"))


def frames(tmp_path):
    a = np.zeros((360 * 640, 4), dtype="<f2")
    a[:, :3] = np.linspace(.0001, .005, a.shape[0])[:, None]
    a[:, 3] = 1
    ref, out = tmp_path / "ref.raw", tmp_path / "out.raw"
    a.tofile(ref)
    return a, ref, out


def test_rgb_gate_ignores_alpha_dominance_and_unit_psnr(tmp_path):
    a, ref, out = frames(tmp_path)
    a.tofile(out)
    assert frame.compare_rgb(ref, out)["pass"]
    a[:, :3] = 0
    a.tofile(out)
    metrics = frame.compare_rgb(ref, out)
    assert metrics["unit_range_psnr_db"] > 35
    assert not metrics["pass"]


def test_rgb_gate_rejects_alpha_nan_and_wrong_shape(tmp_path):
    a, ref, out = frames(tmp_path)
    a[0, 3] = 0
    a.tofile(out)
    assert not frame.compare_rgb(ref, out)["pass"]
    a[0, 0] = np.nan
    a.tofile(out)
    assert frame.compare_rgb(ref, out)["nonfinite_values"] == 1
    out.write_bytes(b"\0\0")
    with pytest.raises(ValueError):
        frame.compare_rgb(ref, out)


def test_mismatched_upstream_weights_fail_before_gpu(tmp_path, monkeypatch):
    model, ref = tmp_path / "model.raw", tmp_path / "ref.raw"
    model.write_bytes(b"new weights")
    ref.write_bytes(b"reference")
    monkeypatch.setattr(frame, "FRAME1_SHA256", frame.digest(ref))
    execution = {"pass": True, "runs": [{}, {}], "model_sha256": family.digest(b"old weights")}
    (tmp_path / "execution.json").write_text(json.dumps(execution))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="model hashes differ"):
        frame.validate(tmp_path, ref, model, tmp_path / "not_invoked.exe", output)
    assert not output.exists()
