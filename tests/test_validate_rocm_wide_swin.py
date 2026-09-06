import sys
from pathlib import Path
import pytest
pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_rocm_wide_swin import validate_selection


@pytest.mark.parametrize('slots,chain', [([], False), ([7, 7], False), ([8, 7], False),
                                        ([999], False), ([7, 8, 11], True), ([133, 135], True)])
def test_bad_selection_rejected_before_gpu(slots, chain):
    with pytest.raises(ValueError):
        validate_selection(slots, chain)


@pytest.mark.parametrize('slots,chain', [([7, 8, 11, 12], False), (list(range(133, 139)), True), ([147, 148], True)])
def test_valid_selection(slots, chain):
    validate_selection(slots, chain)
