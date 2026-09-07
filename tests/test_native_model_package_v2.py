import pytest

from scripts.native_model_package_v2 import build, inspect


def test_v2_package_has_deterministic_verified_sections():
    strict = bytes(range(251)) * 3
    fp8 = bytes(reversed(range(251))) * 2
    strict_segments = [('pre', 0, 251), ('backbone', 251, len(strict) - 251)]
    fp8_segments = [('c256', 0, 251), ('remaining', 251, len(fp8) - 251)]
    first = build(strict, fp8_weights=fp8, temporal_contract_id='temporal-unresolved-v0',
                  strict_segments=strict_segments, fp8_segments=fp8_segments)
    assert first == build(strict, fp8_weights=fp8, temporal_contract_id='temporal-unresolved-v0',
                          strict_segments=strict_segments, fp8_segments=fp8_segments)
    result = inspect(first)
    assert result['schema'] == 2
    assert result['sections']['strict_fp16']['bytes'] == len(strict)
    assert result['sections']['gfx1201_fp8']['bytes'] == len(fp8)
    assert [x['name'] for x in result['sections']['strict_fp16']['segments']] == ['pre', 'backbone']
    assert [x['name'] for x in result['sections']['gfx1201_fp8']['segments']] == ['c256', 'remaining']


def test_v2_package_rejects_corruption_and_empty_contract():
    raw = bytearray(build(b'strict weights', fp8_weights=None,
                          temporal_contract_id='temporal-unresolved-v0'))
    raw[-1] ^= 1
    with pytest.raises(ValueError, match='hash'):
        inspect(bytes(raw))
    with pytest.raises(ValueError):
        build(b'x', fp8_weights=None, temporal_contract_id='')


def test_v2_package_rejects_segment_gaps_and_index_corruption():
    with pytest.raises(ValueError, match='contiguously'):
        build(b'abcdef', fp8_weights=None, temporal_contract_id='temporal-unresolved-v0',
              strict_segments=[('a', 0, 2), ('b', 3, 3)])
    raw = bytearray(build(b'abcdef', fp8_weights=None,
                          temporal_contract_id='temporal-unresolved-v0'))
    raw[-1] ^= 1
    with pytest.raises(ValueError, match='index'):
        inspect(bytes(raw))
