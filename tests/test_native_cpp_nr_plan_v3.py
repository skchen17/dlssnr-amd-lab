import json
from pathlib import Path

from scripts.build_native_stage_topology import build
from scripts.build_native_split512_topology import build as build_split512
from scripts.build_native_vit_topology import build as build_vit
from scripts.build_native_bottleneck_topology import build as build_bottleneck
from scripts.build_native_transition_topology import build as build_transitions
from scripts.native_cpp_nr_plan import (approximate_bottleneck_descriptor,
                                        approximate_scale_transition_descriptor_array,
                                        approximate_split512_descriptor_array,
                                        approximate_stage_descriptor_array,
                                        approximate_vit_descriptor_array,
                                        validate_v3_layouts)
from scripts.native_nr_arena import build_arena_plan


def test_ctypes_abi_v3_layout_is_fixed():
    assert validate_v3_layouts() == {
        'ShapeDescV3': 52,
        'BindingsV3': 152,
        'ExternalSyncV3': 40,
        'ApproxStageBlock': 240,
        'ApproxSplit512Block': 248,
        'ApproxVitBlock': 152,
        'ApproxBottleneck': 112,
        'ApproxScaleTransition': 96,
        'PerformanceStats': 48,
        'ModelPackageStats': 272,
    }


def test_real_wide_topology_converts_to_owned_c_descriptor_array():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    origins = json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']
    topology = build(manifest, build_arena_plan(1920, 1080, 'approx_fp8'), origins)
    descriptors = approximate_stage_descriptor_array(topology)
    assert len(descriptors) == 44
    assert {row.channels for row in descriptors} == {32, 64, 128, 256}
    assert all(row.ffn_inverse_permutation_weight_offset > 0 for row in descriptors)


def test_real_split512_topology_converts_to_owned_c_descriptor_array():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    origins = json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']
    topology = build_split512(manifest, build_arena_plan(1920, 1080, 'approx_fp8'), origins)
    descriptors = approximate_split512_descriptor_array(topology)
    assert len(descriptors) == 16
    assert [row.record_number for row in descriptors] == \
        list(range(23, 31)) + list(range(40, 48))
    assert all(row.qkv_weight_offset > 0 for row in descriptors)


def test_real_vit_topology_converts_to_owned_c_descriptor_array():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    topology = build_vit(manifest, build_arena_plan(1920, 1080, 'approx_fp8'))
    descriptors = approximate_vit_descriptor_array(topology)
    assert len(descriptors) == 8
    assert [row.record_number for row in descriptors] == list(range(31, 39))
    assert all(row.tokens == 640 for row in descriptors)


def test_real_bottleneck_topology_converts_to_owned_c_descriptor():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    topology = build_bottleneck(manifest, build_arena_plan(1920, 1080, 'approx_fp8'))
    descriptor = approximate_bottleneck_descriptor(topology)
    assert descriptor.tokens == 640
    assert descriptor.feature_width == 60
    assert descriptor.feature_height == 36


def test_real_scale_transitions_convert_to_owned_c_descriptor_array():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    arena = build_arena_plan(1920, 1080, 'approx_fp8')
    origins = json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']
    topology = build_transitions(manifest, arena, build(manifest, arena, origins))
    descriptors = approximate_scale_transition_descriptor_array(topology)
    assert len(descriptors) == 8
    assert [row.anchor_record for row in descriptors] == [4, 8, 14, 22, 48, 56, 62, 66]
    assert [row.direction for row in descriptors] == [1, 1, 1, 1, 2, 2, 2, 2]
