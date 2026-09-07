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
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    assert 'bindings->width!=plan->static_width' in runtime
    assert 'bindings->height!=plan->static_height' in runtime


def test_cpp_capture_bridge_has_no_torch_operation_in_submit():
    source=(ROOT/'scripts/native_cpp_nr_plan.py').read_text(encoding='utf-8')
    submit=source.split('def submit(',1)[1].split('def close(',1)[0]
    assert 'nrPlanSubmit' in submit
    for forbidden in ('torch.empty','torch.cat','torch.split','torch.arange','copy_('):
        assert forbidden not in submit


def test_allocator_aware_capture_is_instantiated_by_native_plan():
    bridge=(ROOT/'scripts/native_cpp_nr_plan.py').read_text(encoding='utf-8')
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    assert 'torch.cuda.CUDAGraph(keep_graph=True)' in bridge
    assert 'nrPlanAdoptGraph' in bridge
    assert 'hipGraphClone' not in runtime
    assert 'hipGraphInstantiate' in runtime
    assert 'owns_graph' in runtime
    assert 'nrPlanGetGraphStats' in runtime


def test_native_plan_exposes_read_only_captured_argument_audit():
    header=(ROOT/'tools/native_nr_plan/nr_plan.h').read_text(encoding='utf-8')
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    assert 'nrPlanDebugGetKernelU64Arguments' in header
    assert 'nrPlanDebugGetOwnedAddresses' in header
    audit=runtime.split('nrPlanDebugGetKernelU64Arguments',1)[1].split(
        'nrPlanDebugGetOwnedAddresses',1)[0]
    assert 'hipGraphKernelNodeGetParams' in audit
    assert 'hipGraphLaunch' not in audit
