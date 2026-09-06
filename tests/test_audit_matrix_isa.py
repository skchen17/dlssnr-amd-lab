import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_matrix_isa import audit


def test_requires_every_candidate_family_and_wmma():
    sections = []
    from audit_matrix_isa import FAMILIES
    for marker in FAMILIES.values():
        name='X'+(''.join(marker) if isinstance(marker,tuple) else marker)+'Y'
        sections.append(f'; -- Begin function {name}\n v_wmma_f32_16x16x16_f16 v[0:7],v[0:3],v[0:3],0\n'
                        '.amdhsa_next_free_vgpr 12\n.amdhsa_private_segment_fixed_size 0\n'
                        '.amdhsa_group_segment_fixed_size 32\n; -- End function')
    result = audit('\n'.join(sections))
    assert result['all_candidate_variants_contain_wmma']
    assert set(result['families']) == set(FAMILIES)


def test_rejects_missing_family():
    with pytest.raises(ValueError, match='missing WMMA evidence'):
        audit('')
