import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_transition_isa import audit


def test_all_transition_kernels_require_static_resource_metadata():
    sections=[]
    for marker in ('pool_permute_skip_kernel','quantize_pack_kernel','unpack_permute_kernel','expand_skip_pack_kernel'):
        sections.append(f'; -- Begin function _ZL20{marker}v\n.amdhsa_next_free_vgpr 12\n.amdhsa_next_free_sgpr 8\n.amdhsa_private_segment_fixed_size 0\n.amdhsa_group_segment_fixed_size 0\n; -- End function')
    report=audit('\n'.join(sections))
    assert report['all_kernels_no_scratch'] and report['all_kernels_no_lds']
    assert len(report['kernels'])==4


def test_missing_kernel_is_rejected():
    with pytest.raises(ValueError,match='expected one'):
        audit('')
