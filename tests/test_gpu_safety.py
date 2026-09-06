import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_safety


def _write_state(path, **overrides):
    state = {
        "halted": True,
        "non_profiler_gpu_tests_halted": False,
    }
    state.update(overrides)
    path.write_text(json.dumps(state), encoding="utf-8")


def test_profiler_remains_blocked_when_limited_gpu_tests_resume(tmp_path, monkeypatch):
    halt = tmp_path / "halt.json"
    _write_state(halt)
    monkeypatch.setattr(gpu_safety, "HALT_FILE", halt)

    gpu_safety.require_gpu_tests_enabled("minimal gate")
    with pytest.raises(RuntimeError, match="RGP capture blocked"):
        gpu_safety.require_gpu_tests_enabled("RGP capture", profiler=True)


def test_non_profiler_gate_can_be_halted_independently(tmp_path, monkeypatch):
    halt = tmp_path / "halt.json"
    _write_state(halt, halted=False, non_profiler_gpu_tests_halted=True)
    monkeypatch.setattr(gpu_safety, "HALT_FILE", halt)

    gpu_safety.require_gpu_tests_enabled("RGP capture", profiler=True)
    with pytest.raises(RuntimeError, match="minimal gate blocked"):
        gpu_safety.require_gpu_tests_enabled("minimal gate")


def test_missing_state_file_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_safety, "HALT_FILE", tmp_path / "missing.json")
    gpu_safety.require_gpu_tests_enabled("minimal gate")
    gpu_safety.require_gpu_tests_enabled("RGP capture", profiler=True)
