from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_submit_has_fixed_allocation_free_graph_contract():
    source=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    submit=source.split('nrPlanSubmit(',1)[1].split('nrPlanReset(',1)[0]
    assert 'hipGraphLaunch' in submit
    assert 'hipMemcpyAsync' in submit
    assert 'hipMalloc' not in submit
    assert 'hipStreamSynchronize' not in submit
    assert 'hipErrorNotReady' in submit


def test_frame_bindings_include_identity_and_dynamic_geometry():
    header=(ROOT/'tools/native_nr_plan/nr_plan.h').read_text(encoding='utf-8')
    for field in ('input','output','width','height','frame_id','resource_generation'):
        assert field in header

