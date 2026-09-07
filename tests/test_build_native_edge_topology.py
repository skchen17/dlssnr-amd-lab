import ctypes as ct

import pytest

from scripts.build_native_edge_topology import build
from scripts.native_cpp_nr_plan import ApproxHead, ApproxPre, approximate_edge_descriptors


NAMES = {
    'pre.input_project': ('fp16_le', [16, 32]),
    'pre.swin.block.ffn.expand': ('e4m3fn', [32, 128]),
    'pre.swin.block.ffn.contract': ('e4m3fn', [1, 128, 32]),
    'pre.swin.block.ffn.residual_scale': ('fp16_le', [32]),
    'pre.swin.a_index': ('int32_le', [64, 32]),
    'pre.swin.residual_index': ('int32_le', [64, 32]),
    'pre.swin.block.ffn.permutation': ('int32_le', [32]),
    'pre.swin.block.ffn.inverse_permutation': ('int32_le', [32]),
    'pre.swin.block.attention.qkv': ('e4m3fn', [32, 96]),
    'pre.swin.block.attention.q_scale': ('fp16_le', [1]),
    'pre.swin.block.attention.permutation': ('int32_le', [32]),
    'pre.swin.block.attention.position_bias': ('fp16_le', [1, 64, 64]),
    'pre.swin.block.attention.project': ('e4m3fn', [32, 32]),
    'pre.swin.block.attention.attention_scale': ('fp16_le', [32]),
    'head.main_scale': ('fp16_le', [32]),
    'head.skip_scale': ('fp16_le', [32]),
    'head.tail': ('fp16_le', [32, 4]),
    'head.block.expand': ('e4m3fn', [4, 32, 32]),
    'head.block.contract': ('e4m3fn', [4, 32, 32]),
    'head.block.ffn_scale': ('fp16_le', [32]),
    'head.block.a_index': ('int32_le', [64, 32]),
    'head.block.residual_index': ('int32_le', [64, 32]),
    'head.block.permutation': ('int32_le', [32]),
    'head.block.qkv': ('e4m3fn', [32, 96]),
    'head.block.q_scale': ('fp16_le', []),
    'head.block.position_bias': ('fp16_le', [64, 64]),
    'head.block.project': ('e4m3fn', [32, 32]),
    'head.block.attention_scale': ('fp16_le', [32]),
}


def fixture():
    records = []
    for index, (name, (dtype, shape)) in enumerate(NAMES.items()):
        records.append({'name': name, 'dtype': dtype, 'logical_shape': shape,
                        'offset': index * 65536})
    weights = {'target_arch': 'gfx1201',
               'status': 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED',
               'records': records}
    model = {'settings': {'color_scale': .125,
                          'conditioning': [0., 1., 1., -1., -1.]}}
    return weights, model


def test_edge_topology_is_exact_1080p_single_color_reset_contract():
    weights, model = fixture()
    result = build(weights, model)
    assert result['status'] == 'FULL_NATIVE_SINGLE_COLOR_EDGES_NOT_RUNTIME_ACCEPTED'
    assert result['temporal_contract_verified'] is False
    assert result['pre']['windows'] == 34560
    assert result['head']['windows'] == 34945
    pre, head = approximate_edge_descriptors(result)
    assert pre.struct_size == ct.sizeof(ApproxPre) == 160
    assert head.struct_size == ct.sizeof(ApproxHead) == 136


def test_edge_topology_rejects_unreviewed_size_and_conditioning():
    weights, model = fixture()
    with pytest.raises(ValueError, match='1080p'):
        build(weights, model, 1280, 720)
    model['settings']['conditioning'][0] = 1.
    with pytest.raises(ValueError, match='conditioning'):
        build(weights, model)
