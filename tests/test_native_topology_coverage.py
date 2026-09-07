import json
from pathlib import Path

from scripts.audit_native_topology_coverage import audit
from scripts.build_native_transition_topology import build as build_transitions
from scripts.build_native_stage_topology import build as build_wide
from scripts.native_nr_arena import build_arena_plan


def test_real_partial_topologies_cover_exactly_c32_through_vit():
    root = Path('results/20260907_native_nr_plan_build_v27')
    manifest = json.loads(Path('local_models/gfx1201_stage_cache_v3/manifest.json').read_text())
    arena = build_arena_plan(1920, 1080, 'approx_fp8')
    wide = json.loads((root / 'swin32_256_topology.json').read_text())
    transitions = build_transitions(manifest, arena, wide)
    result = audit(wide,
                   json.loads((root / 'split512_topology.json').read_text()),
                   json.loads((root / 'vit_topology.json').read_text()),
                   json.loads((root / 'bottleneck_topology.json').read_text()),
                   transitions)
    assert result['native_block_count'] == 69
    assert result['conceptual_launch_count_not_graph_nodes'] == 418
    assert result['theoretical_graph_memcpy_nodes'] == 5
    assert result['kernel_node_budget_evaluated'] is False
    assert result['missing_by_family'] == {
        'Pre': [0], 'Head': [70],
    }
    assert result['native_scale_transition_count'] == 8
    assert set(result['unimplemented_record_subpaths']) == {
        'pre_temporal_import_and_swin', 'head_and_history_export'}
    assert result['complete_native_topology'] is False
    assert result['gpu_executed'] is False
