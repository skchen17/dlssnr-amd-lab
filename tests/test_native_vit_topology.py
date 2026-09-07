import json
from pathlib import Path

from scripts.build_native_vit_topology import build
from scripts.native_nr_arena import build_arena_plan


ROOT = Path(__file__).resolve().parents[1]


def test_real_cache_maps_eight_vit_blocks():
    weights = json.loads((ROOT / 'local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    result = build(weights, build_arena_plan(1920, 1080, 'approx_fp8'))
    assert result['block_count'] == 8
    assert result['tokens'] == 640
    assert result['graph_kernel_nodes_if_recorded'] == 40
    assert [row['record_number'] for row in result['blocks']] == list(range(31, 39))
    assert result['blocks'][0]['input_offset'] == result['blocks'][1]['next_offset']
    assert result['blocks'][0]['next_offset'] == result['blocks'][1]['input_offset']
