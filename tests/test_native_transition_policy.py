import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_transition_policy import active_transition_policy,transition_policy


def test_default_preserves_capture_compatible_layout_path():
    value=active_transition_policy()
    assert not value.resident and not value.capture_outviews


def test_resident_policy_is_scoped_and_explicit():
    with transition_policy(resident=True) as value:
        assert value.resident and not value.capture_outviews
        assert active_transition_policy()==value
    assert not active_transition_policy().resident


def test_debug_view_requires_resident_path():
    with pytest.raises(ValueError):
        with transition_policy(capture_outviews=True):pass
    with pytest.raises(ValueError):
        with transition_policy(resident=1):pass
