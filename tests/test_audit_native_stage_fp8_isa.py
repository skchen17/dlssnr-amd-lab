from scripts.audit_native_stage_fp8_isa import audit
import pytest


def section(name, wmma=1, vgpr=20, lds=0, word_store=False):
    return (f'; -- Begin function {name}\n'
            + 'v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[0:1], v[2:3], v[0:7]\n' * wmma
            + f'.amdhsa_next_free_vgpr {vgpr}\n'
            + '.amdhsa_next_free_sgpr 12\n'
            + f'.amdhsa_group_segment_fixed_size {lds}\n'
            + '.amdhsa_private_segment_fixed_size 0\n'
            + '.amdhsa_kernarg_size 48\n'
            + '.amdhsa_wavefront_size32 1\n'
            + ('global_store_b32 v[0:1], v2, off\n' if word_store
               else 'global_store_b16 v[0:1], v2, off\n')
            + '; -- End function\n')


def test_audit_requires_all_stage_specific_kernels():
    text = ''.join(section(f'_Z_{family}ILi{channel}E',
                           word_store=family == 'stage_qkv_pack_e4x4_part_from_fp16',
                           wmma=0 if family in ('stage_qkv_norm_fp16_inplace',
                                                'stage_qkv_pack_e4x4_part_from_fp16') else 1)
                   for channel in (64, 128, 256)
                   for family in ('stage_ffn_group_fp8', 'stage_ffn_mix_fp8',
                                  'stage_qkv_project_fp8', 'stage_qkv_norm_fp16_inplace',
                                  'stage_qkv_pack_e4x4_part_from_fp16',
                                  'stage_attention_fp8',
                                  'stage_project_fp8'))
    text += ''.join(section(f'_Z_stage_scatter_fp8ILi{channel}E', wmma=0)
                    for channel in (64, 128, 256))
    text += ''.join(section(f'_Z_{family}') for family in
                    ('stage_c512_preproject_fp8', 'stage_c512_group_fp8',
                     'stage_c512_ffn_project_fp8'))
    text += ''.join(section(f'_Z_{family}ILi512E') for family in
                    ('stage_qkv_norm_fused_fp8', 'stage_attention_fp8',
                     'stage_project_fp8'))
    text += section('_Z_stage_scatter_fp8ILi512E', wmma=0)
    text += section('_Z_stage_c32_ffn_fp8')
    text += ''.join(section(f'_Z_{family}ILi32E',
                            word_store=family == 'stage_qkv_pack_e4x4_part_from_fp16',
                            wmma=0 if family in ('stage_qkv_norm_fp16_inplace',
                                                 'stage_qkv_pack_e4x4_part_from_fp16') else 1) for family in
                    ('stage_qkv_project_fp8', 'stage_qkv_norm_fp16_inplace',
                     'stage_qkv_pack_e4x4_part_from_fp16',
                     'stage_attention_fp8', 'stage_project_fp8'))
    text += section('_Z_stage_scatter_fp8ILi32E', wmma=0)
    result = audit(text)
    assert result['all_matrix_stage_kernels_emit_fp8_wmma']
    assert result['all_standard_qkv_norm_and_pack_kernels_are_wave32']
    assert result['all_standard_qkv_pack_kernels_publish_software_e4m3x4_words']
    assert result['all_fp8_boundaries_avoid_unreliable_gfx1201_hardware_convert']
    assert len(result['kernels']) == 38


def test_audit_rejects_gfx1201_hardware_fp8_conversion():
    text = ''.join(section(f'_Z_{family}ILi{channel}E',
                           word_store=family == 'stage_qkv_pack_e4x4_part_from_fp16',
                           wmma=0 if family in ('stage_qkv_norm_fp16_inplace',
                                                'stage_qkv_pack_e4x4_part_from_fp16') else 1)
                   for channel in (64, 128, 256)
                   for family in ('stage_ffn_group_fp8', 'stage_ffn_mix_fp8',
                                  'stage_qkv_project_fp8', 'stage_qkv_norm_fp16_inplace',
                                  'stage_qkv_pack_e4x4_part_from_fp16',
                                  'stage_attention_fp8', 'stage_project_fp8'))
    text += ''.join(section(f'_Z_stage_scatter_fp8ILi{channel}E', wmma=0)
                    for channel in (64, 128, 256))
    text += ''.join(section(f'_Z_{family}') for family in
                    ('stage_c512_preproject_fp8', 'stage_c512_group_fp8',
                     'stage_c512_ffn_project_fp8'))
    text += ''.join(section(f'_Z_{family}ILi512E') for family in
                    ('stage_qkv_norm_fused_fp8', 'stage_attention_fp8',
                     'stage_project_fp8'))
    text += section('_Z_stage_scatter_fp8ILi512E', wmma=0)
    text += section('_Z_stage_c32_ffn_fp8')
    text += ''.join(section(f'_Z_{family}ILi32E',
                            word_store=family == 'stage_qkv_pack_e4x4_part_from_fp16',
                            wmma=0 if family in ('stage_qkv_norm_fp16_inplace',
                                                 'stage_qkv_pack_e4x4_part_from_fp16') else 1) for family in
                    ('stage_qkv_project_fp8', 'stage_qkv_norm_fp16_inplace',
                     'stage_qkv_pack_e4x4_part_from_fp16',
                     'stage_attention_fp8', 'stage_project_fp8'))
    text += section('_Z_stage_scatter_fp8ILi32E', wmma=0)
    text = text.replace('global_store_b16 v[0:1], v2, off\n',
                        'v_cvt_pk_fp8_f32 v2, v0, v1\n'
                        'global_store_b16 v[0:1], v2, off\n', 1)
    with pytest.raises(ValueError):
        audit(text)
