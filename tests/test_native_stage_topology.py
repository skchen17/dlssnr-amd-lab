import copy
import json
from pathlib import Path

import pytest

from scripts.build_native_stage_topology import build
from scripts.native_nr_arena import build_arena_plan


def manifest():
    return json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())


def origins():
    return json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']


def test_real_private_manifest_maps_all_c32_to_c256_blocks_without_claiming_completion():
    result = build(manifest(), build_arena_plan(1920, 1080, 'approx_fp8'), origins())
    assert result['family_counts'] == {'c32': 8, 'c64': 8, 'c128': 12, 'c256': 16}
    assert result['block_count'] == 44
    assert result['graph_kernel_nodes_if_recorded'] == 300
    assert not result['complete_native_topology']
    assert [row['record_number'] for row in result['blocks']] == sorted(
        row['record_number'] for row in result['blocks'])
    assert len({row['record_number'] for row in result['blocks']}) == 44
    assert all(row['struct_size'] == 240 for row in result['blocks'])
    assert all(row['qkv_projection_fp16_offset'] > 0 for row in result['blocks'])
    assert {row['windows'] for row in result['blocks'] if row['channels'] == 64} == \
        {2160, 2196, 2220, 2257}
    assert {row['windows'] for row in result['blocks'] if row['channels'] == 32} == \
        {8640, 8712, 8760, 8833}


def test_wrong_qkv_layout_is_rejected():
    source = manifest()
    qkv = next(row for row in source['records']
               if row['name'].endswith('attention.qkv') and row['logical_shape'][0] == 64)
    qkv['storage_layout'] = 'row_major'
    with pytest.raises(ValueError, match='invalid QKV'):
        build(source, build_arena_plan(1920, 1080, 'approx_fp8'), origins())
