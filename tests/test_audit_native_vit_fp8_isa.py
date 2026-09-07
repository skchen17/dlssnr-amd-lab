from scripts.audit_native_vit_fp8_isa import audit


def test_audit_requires_all_five_wmma_kernels():
    def section(name, vgpr=32, lds=0):
        return (f'; -- Begin function {name}\n'
                'v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[0:1], v[2:3], v[0:7]\n'
                f'.amdhsa_next_free_vgpr {vgpr}\n.amdhsa_next_free_sgpr 16\n'
                f'.amdhsa_group_segment_fixed_size {lds}\n'
                '.amdhsa_private_segment_fixed_size 0\n; -- End function\n')
    text = ''.join(section(name, 40 + i, 512 if 'attention' in name else 0)
                   for i, name in enumerate((
                       'vit_ffn_expand_fp8', 'vit_ffn_contract_fp8',
                       'vit_qkv_norm_fp8', 'vit_global_attention_fp8',
                       'vit_project_fp8')))
    result = audit(text)
    assert result['kernels_per_block'] == 5
    assert not result['attention_materializes_nxn']
    assert len(result['kernels']) == 5
