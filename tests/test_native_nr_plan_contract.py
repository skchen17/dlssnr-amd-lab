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
    assert 'nrPlanDebugCopyStageE4LutToDevice' in header
    audit=runtime.split('nrPlanDebugGetKernelU64Arguments',1)[1].split(
        'nrPlanDebugGetOwnedAddresses',1)[0]
    assert 'hipGraphKernelNodeGetParams' in audit
    assert 'hipGraphLaunch' not in audit


def test_resident_transition_bringup_does_not_weaken_production_topology_gate():
    header=(ROOT/'tools/native_nr_plan/nr_plan.h').read_text(encoding='utf-8')
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    production=runtime.split('nrPlanConfigureApproxScaleTransitions',1)[1].split(
        'nrPlanDebugConfigureApproxEncoderTransition',1)[0]
    debug=runtime.split('nrPlanDebugConfigureApproxEncoderTransition',1)[1].split(
        'nrPlanDebugGetFixedTransitionState',1)[0]
    assert 'count!=8' in production
    assert 'approximate_stage_blocks.size()!=44' in production
    assert 'approximate_split512_blocks.size()!=16' in production
    assert 'NR_TRANSITION_ENCODER_DOWNSAMPLE' in debug
    assert 'value.channels!=32&&value.channels!=64&&value.channels!=128' in debug
    assert 'nrPlanDebugConfigureApproxEncoderTransition' in header


def test_fixed_transition_chain_uses_owned_target_and_clears_stale_boundaries():
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    record=runtime.split('static hipError_t record_scale_transition(',1)[1].split(
        'static hipError_t record_scale_transition_at',1)[0]
    assert 'plan->fixed_transition_target' in record
    assert 'plan->fixed_transition_skip_pool' in record
    assert 'plan->fixed_resident_chain_input=plan->fixed_transition_target' in record
    assert 'if(fixed)clear_chain();' in record
    assert 'transition.channels==32||transition.channels==64||transition.channels==128' in record


def test_split_c512_fixed_sequence_uses_compact_resident_chain():
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    split=runtime.split('static hipError_t record_approximate_split512_block',1)[1].split(
        'static hipError_t record_approximate_vit_block',1)[0]
    assert 'scalars*8' in split
    assert 'fixed_resident_chain_input' in split
    assert 'nr_stage_c512_ffn_fp8(active_raw' in split
    assert 'scratch+scalars*7' in split
    assert 'fixed_resident_chain_channels=512' in split


def test_fixed_central_chain_owns_vit_and_bottleneck_storage():
    header=(ROOT/'tools/native_nr_plan/nr_plan.h').read_text(encoding='utf-8')
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    vit=runtime.split('static hipError_t record_approximate_vit_block',1)[1].split(
        'static hipError_t record_encoder_bottleneck',1)[0]
    encoder=runtime.split('static hipError_t record_encoder_bottleneck',1)[1].split(
        'static hipError_t record_decoder_bottleneck',1)[0]
    decoder=runtime.split('static hipError_t record_decoder_bottleneck',1)[1].split(
        'static hipError_t record_scale_transition',1)[0]
    assert 'fixed_vit_scratch' in vit
    assert 'activation*10' in vit
    assert 'fixed_resident_chain_input=next' in vit
    assert 'fixed_bottleneck_skip' in encoder
    assert 'fixed_resident_chain_channels=1024' in encoder
    assert 'fixed_transition_target' in decoder
    assert 'fixed_resident_chain_channels=512' in decoder
    assert 'nrPlanDebugGetFixedCentralState' in header
    assert 'nrPlanDebugCopyFixedResidentToDevice' in header


def test_fixed_sequence_host_boundary_elision_is_explicit_and_reversible():
    header=(ROOT/'tools/native_nr_plan/nr_plan.h').read_text(encoding='utf-8')
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    assert 'nrPlanDebugSetFixedHostBoundaries' in header
    assert 'fixed_sequence_host_boundaries{true}' in runtime
    helper=runtime.split('static hipError_t fixed_boundary',1)[1].split(
        'struct InstantiateTask',1)[0]
    assert 'hipStreamSynchronize' in helper
    assert '!plan->fixed_sequence_host_boundaries' in helper
    assert 'fixed_resident_recording' in runtime
    assert 'plan->fixed_sequence_host_boundaries?1:0' in runtime
    submit=runtime.split('static hipError_t submit_v3',1)[1].split(
        'NRPLAN_API hipError_t nrPlanSubmitV3',1)[0]
    assert 'plan->fixed_sequence_host_boundaries' in submit
    assert 'hipGraphLaunch' in submit


def test_reset_single_color_edges_are_native_and_dynamically_bound():
    header=(ROOT/'tools/native_nr_plan/nr_plan.h').read_text(encoding='utf-8')
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    io=(ROOT/'tools/native_io_fp8/io_fp8.hip').read_text(encoding='utf-8')
    stage=(ROOT/'tools/native_stage_fp8/stage_fp8.hip').read_text(encoding='utf-8')
    build=(ROOT/'tools/native_nr_plan/build.ps1').read_text(encoding='utf-8')
    assert 'NRApproxPreDesc' in header and 'NRApproxHeadDesc' in header
    assert 'nrPlanConfigureApproxPre' in header and 'nrPlanConfigureApproxHead' in header
    pre=runtime.split('static hipError_t record_approximate_pre',1)[1].split(
        'static hipError_t record_approximate_head',1)[0]
    head=runtime.split('static hipError_t record_approximate_head',1)[1].split(
        'static hipError_t record_approximate_stage_blocks',1)[0]
    assert 'plan->device_bindings_v3' in pre
    assert 'nr_io_pre_project_pack_v3' in pre
    assert 'nr_io_pre_pool_fp8' in pre
    assert 'nr_io_head_input_fp8' in head
    assert 'nr_io_head_tail_v3' in head
    assert 'fixed_edge_scratch_bytes%12' in runtime
    assert 'edge_values*12' in runtime
    assert 'nr_stage_head_c32_ffn_fp8' in stage
    assert 'nr_stage_c32_qkv_compact_publish_fp8' in stage
    assert 'nrPlanSetApproxEdgeCompactQkv' in header
    assert 'nrPlanSetApproxStandardCompactQkv' in header
    assert 'nrPlanSetApproxFullFusedQkv' in header
    assert 'nrPlanSetApproxFusedQkvConsumesFp16' in header
    assert 'nrPlanSetApproxElideRedundantPostPublish' in header
    assert 'approximate_standard_compact_qkv' in runtime
    assert 'approximate_full_fused_qkv' in runtime
    assert 'approximate_elide_redundant_post_publish' in runtime
    assert 'stage_qkv_full_fused_lut_fp8' in stage
    assert 'qkv_full_fused_from_fp16' in stage
    for channels in (64,128,256):
        assert f'EXPORT_COMPACT_QKV({channels})' in stage
    assert 'bindings->current_color' in io
    assert 'bindings->output_residual' in io
    assert 'io_fp8.hip' in build


def test_complete_topology_does_not_claim_unverified_temporal_deployment():
    runtime=(ROOT/'tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    refresh=runtime.split('static void refresh_complete_native_topology',1)[1].split(
        'static hipError_t allocate_edge_storage',1)[0]
    assert 'has_approximate_pre' in refresh
    assert 'has_approximate_head' in refresh
    resources=runtime.split('NRPLAN_API hipError_t nrPlanGetResourceStats',1)[1].split(
        'NRPLAN_API hipError_t nrPlanConfigureArena',1)[0]
    assert 'temporal_contract_verified' in resources
    assert ('stats.deployment_ready=stats.deployment_ready&&stats.complete_native_topology&&'
            in resources)
    assert 'stats.temporal_contract_verified' in resources
