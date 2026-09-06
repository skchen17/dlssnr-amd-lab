import sys
from pathlib import Path
import pytest

pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_scale_transitions import abba_schedule,geometry


def test_two_group_abba_schedule_is_balanced():
    labels=abba_schedule()
    assert labels==['reference','native','native','reference']*2
    assert labels.count('reference')==labels.count('native')==4


def test_scale_geometry_matches_padded_whole_frame():
    assert geometry('4k','encoder',32)==(1920,1088)
    assert geometry('4k','encoder',256)==(240,136)
    assert geometry('4k','decoder',256)==(240,136)
    assert geometry('4k','decoder',32)==(1920,1088)
