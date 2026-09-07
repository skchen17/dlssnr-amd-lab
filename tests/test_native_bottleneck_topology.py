import json
from pathlib import Path

from scripts.build_native_bottleneck_topology import build
from scripts.native_nr_arena import build_arena_plan


def test_real_cache_maps_c512_vit_boundaries():
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    result = build(manifest, build_arena_plan(1920, 1080, 'approx_fp8'))
    assert result['encoder_after_record'] == 30
    assert result['decoder_record'] == 39
    assert result['graph_kernel_nodes_if_recorded'] == 2
    assert result['graph_memcpy_nodes_if_recorded'] == 1
    assert result['descriptor']['tokens'] == 640
    assert result['descriptor']['feature_width'] == 60
    assert result['descriptor']['feature_height'] == 36
