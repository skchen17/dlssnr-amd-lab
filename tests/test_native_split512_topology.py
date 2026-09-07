import copy
import json
from pathlib import Path

import pytest

from scripts.build_native_split512_topology import build
from scripts.native_nr_arena import build_arena_plan


def inputs():
    weights = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    origins = json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']
    return weights, build_arena_plan(1920, 1080, 'approx_fp8'), origins


def test_real_private_cache_maps_all_split512_blocks():
    result = build(*inputs())
    assert result['block_count'] == 16
    assert result['record_numbers'] == list(range(23, 31)) + list(range(40, 48))
    assert result['graph_kernel_nodes_if_recorded'] == 112
    assert {row['windows'] for row in result['blocks']} == {40}
    assert all(row['struct_size'] == 248 for row in result['blocks'])
    assert result['complete_native_topology'] is False


def test_invalid_qkv_storage_is_rejected():
    weights, arena, origins = inputs()
    row = next(row for row in weights['records']
               if row['name'] == 'encoder512.23.block.attention.qkv')
    row['storage_layout'] = 'row_major'
    with pytest.raises(ValueError, match='column-major'):
        build(weights, arena, origins)
