import json
from pathlib import Path

from scripts.build_native_transition_topology import build
from scripts.build_native_stage_topology import build as build_wide
from scripts.native_nr_arena import build_arena_plan


def test_real_cache_maps_all_eight_resident_transitions():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    arena = build_arena_plan(1920, 1080, 'approx_fp8')
    origins = json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']
    result = build(manifest, arena, build_wide(manifest, arena, origins))
    assert result['transition_count'] == 8
    assert result['graph_kernel_nodes_if_recorded'] == 8
    assert result['graph_memcpy_nodes_if_recorded'] == 4
    encoder = [row for row in result['descriptors'] if row['direction'] == 1]
    decoder = [row for row in result['descriptors'] if row['direction'] == 2]
    assert [row['anchor_record'] for row in encoder] == [4, 8, 14, 22]
    assert [row['anchor_record'] for row in decoder] == [48, 56, 62, 66]
    assert encoder[0]['source_width'] == 960 and encoder[0]['target_width'] == 480
    assert decoder[-1]['source_width'] == 480 and decoder[-1]['target_width'] == 960
    assert all(row['skip_scale_weight_offset'] == 0 for row in encoder)
    assert all(row['skip_scale_weight_offset'] != 0 for row in decoder)
    assert all(row['source_offset'] != row['source_resident_offset'] for row in encoder)
