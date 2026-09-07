import json
from pathlib import Path

from scripts.audit_native_stage_package import audit
from scripts.build_native_stage_topology import build as build_topology
from scripts.native_model_package_v2 import build as build_package
from scripts.native_nr_arena import build_arena_plan


def test_package_cache_topology_cross_check():
    cache = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    fp8 = Path('local_models/gfx1201_stage_cache_v3/weights_gfx1201.bin').read_bytes()
    origins = json.loads(Path('local_models/native_single_color_v1/model.json').read_text())['window_origins']
    topology = build_topology(cache, build_arena_plan(1920, 1080, 'approx_fp8'), origins)
    package = build_package(b'strict', fp8_weights=fp8,
                            temporal_contract_id='temporal-unresolved-v0')
    result = audit(package, cache, topology)
    assert result['mapped_stage_blocks'] == 44
    assert not result['complete_native_topology']
