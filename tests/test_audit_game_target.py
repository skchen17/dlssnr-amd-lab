import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("audit_game_target", Path(__file__).parents[1] / "scripts/audit_game_target.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rejects_game_and_external_output_before_reading_executable(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    for output in (game / "output", tmp_path / "other"):
        with pytest.raises(ValueError, match="output must be under lab results"):
            module.audit(game, output)
        assert not output.exists()


def test_missing_game_executable_is_not_launched(tmp_path):
    output = Path(__file__).parents[1] / "results/unused_audit_test"
    with pytest.raises(ValueError, match="GoWR.exe is absent"):
        module.audit(tmp_path, output)
    assert not output.exists()
