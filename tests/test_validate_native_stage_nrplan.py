import argparse
import json
from pathlib import Path

from scripts.validate_native_stage_nrplan import preflight


def args(tmp_path, record=5):
    return argparse.Namespace(
        dll=Path('results/20260907_native_nr_plan_build_v30/nr_plan.dll'),
        package=Path('local_models/native_single_color_gfx1201_v3.nrmpkg'),
        cache=Path('local_models/gfx1201_stage_cache_v3/manifest.json'),
        arena=Path('results/20260907_native_nr_plan_build_v30/arena_1080p_approx.json'),
        topology=Path('results/20260907_native_nr_plan_build_v30/swin32_256_topology.json'),
        record_number=record,
        output=tmp_path / 'preflight.json', execute=False, child=False)


def test_cpu_preflight_binds_one_origin_zero_block_without_gpu(tmp_path):
    result, arena, block = preflight(args(tmp_path))
    assert result['status'] == 'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED'
    assert result['gpu_executed'] is False
    assert result['expected_graph_kernel_nodes'] == 11
    assert result['complete_native_topology'] is False
    assert block['record_number'] == 5
    assert arena['workspace_bytes'] < 1_000_000_000


def test_checked_in_safety_state_blocks_the_gpu_mode():
    state = json.loads(Path('safety/GPU_PROFILE_HALT.json').read_text(encoding='utf-8'))
    assert state['last_minimal_gate']['authorizes_next_gpu_gate'] is False


def test_qkv_poison_is_synchronized_across_runtime_streams():
    source = Path('scripts/validate_native_stage_nrplan.py').read_text(encoding='utf-8')
    assert "'q': torch.full_like(encode_e4(q_expected), 0xA1)" in source
    assert "'k': torch.full_like(encode_e4(k_expected), 0xA2)" in source
    assert "'v': torch.full_like(encode_e4(v_expected), 0xA3)" in source
    assert 'torch.cuda.synchronize()' in source
    assert "'remaining_after_submit'" in source
    assert 'nrPlanDebugGetKernelU64Arguments' in source
    assert "'kernel_argument_audit': kernel_argument_audit" in source
    assert "'exact': captured == expected_arguments" in source
    assert "'kernel_pointer_audits': pointer_audits" in source
    assert "'stage_e4_lut_audit': stage_e4_lut_audit" in source
    assert 'nr_stage_debug_pack_e4x4_c256' in source
