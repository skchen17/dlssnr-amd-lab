from pathlib import Path


def test_stage_source_preserves_bounded_fusion_and_stage_specific_families():
    source = Path('tools/native_stage_fp8/stage_fp8.hip').read_text(encoding='utf-8')
    assert '__shared__ _Float16 projection[3][32][16]' in source
    assert '__shared__ _Float16 scores[16][64]' in source
    assert '__shared__ uint8_t probability[16][64]' in source
    assert '__shared__ _Float16 hidden[128][16]' in source
    assert 'EXPORT_STAGE(64)' in source
    assert 'EXPORT_STAGE(128)' in source
    assert 'EXPORT_STAGE(256)' in source
    assert 'next_resident[index] = e4(float(result))' in source
    assert 'post_resident[index]=e4(float(result))' in source
    assert 'stage_scatter_fp8' in source
    assert 'window_source_index<C>' in source
    assert 'nr_stage_e4_from_fp16_lut[__half_as_ushort(rounded)]' in source
    assert 'nr_stage_initialize_e4_lut' in source
    assert 'hipMalloc(explicit_lut, table.size())' in source
    assert 'e4_explicit(float(value), lut)' in source
    assert 'std::array<uint8_t, 65536> table' in source
    assert '__hip_cvt_float_to_fp8' not in source
    assert '__hip_cvt_float2_to_fp8x2' not in source
    assert '__shared__ _Float16 hidden[256][16]' in source
    assert 'nr_stage_c512_ffn_fp8' in source
    assert 'nr_stage_c512_qkv_norm_fp8' in source
    assert 'nr_stage_c512_attention_fp8' in source
    assert 'nr_stage_c512_project_fp8' in source
    assert 'nr_stage_c512_scatter_fp8' in source
    assert 'nr_stage_c32_ffn_fp8' in source
    assert 'nr_stage_c32_qkv_norm_fp8' in source
    assert 'nr_stage_c32_attention_fp8' in source
    assert 'nr_stage_c32_project_fp8' in source
    assert 'nr_stage_c32_scatter_fp8' in source
    assert 'stage_qkv_norm_fp16_inplace' in source
    assert 'stage_qkv_pack_e4_from_fp16' in source
    assert '_Float16* projection, const _Float16* qscale, size_t vectors' in source
    assert 'uint8_t* q, uint8_t* k, uint8_t* v,' in source
    assert 'uint8_t* target = part == 0 ? q : (part == 1 ? k : v)' in source
    assert 'if (part != 2)' in source
    assert 'v_bytes-k_bytes!=resident_stride' in source
    assert 'target[index * 32 + lane] = e4_explicit(float(value), lut)' in source
    assert 'stage_qkv_norm_fp16_inplace<C>' in source
    assert 'stage_qkv_pack_e4_from_fp16<C>' in source
    assert 'reinterpret_cast<uint16_t*>(target)' not in source


def test_nrplan_initializes_stage_e4_lut_before_graph_capture():
    source = Path('tools/native_nr_plan/nr_plan.cpp').read_text(encoding='utf-8')
    assert 'static hipError_t ensure_stage_e4_lut(NRPlan* plan)' in source
    assert 'nr_stage_initialize_e4_lut(' in source
    assert '&plan->stage_e4_lut' in source
    assert 'if(plan->stage_e4_lut)keep(hipFree(plan->stage_e4_lut))' in source
    assert 'hipError_t lut_error=ensure_stage_e4_lut(plan)' in source
