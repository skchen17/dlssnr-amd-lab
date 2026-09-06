import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("ffx_observer_analysis", Path(__file__).parents[1] / "scripts/analyze_ffx_observer_selftest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_strict_json_rejects_nan(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"value":NaN}\n')
    with pytest.raises(ValueError, match="nonstandard"):
        module.read_jsonl(path)


def test_strict_json_accepts_null_rejects_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"value":null}\n\n')
    assert module.read_jsonl(path) == [{"value": None}]
    path.write_text('{"value":')
    with pytest.raises(json.JSONDecodeError):
        module.read_jsonl(path)


def test_incomplete_evidence_fails(tmp_path):
    (tmp_path / "manifest.json").write_text('{"status":"PASS"}')
    (tmp_path / "decoded.jsonl").write_text('')
    (tmp_path / "header_only.jsonl").write_text('')
    (tmp_path / "gpu_output.raw").write_bytes(b'')
    report = module.analyze(tmp_path)
    assert report["status"] == "FAIL"
    assert not report["checks"]["gpu_output_independent_pattern"]
    assert report["game_abi_verified"] is False
