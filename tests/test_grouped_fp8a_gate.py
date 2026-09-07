import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from validate_grouped_wide_ffn import FP8A_TOLERANCE


def test_fp8a_tolerance_is_bounded_below_one_half_ulp_near_one():
    assert FP8A_TOLERANCE == {'max_absolute_error': 0.001, 'nrmse': 0.0005}
    assert FP8A_TOLERANCE['max_absolute_error'] < 2 ** -9
