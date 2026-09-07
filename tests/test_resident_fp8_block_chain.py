import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_dual_ffn_epilogue_publishes_fp16_and_resident_fp8():
    source = (ROOT / 'tools' / 'native_fusion_probe' / 'head_wmma.hip').read_text(encoding='utf-8')
    assert 'wide_group_mix_fp8a_dual' in source
    assert 'output[index]=value;' in source
    assert 'resident[index]=e4(float(value));' in source
    for channels in (64,128,256):
        assert f'GROUP_FFN_FP8A_DUAL_EXPORT({channels})' in source


def test_packed_swin_routes_resident_bytes_directly_to_qkv():
    packed = (ROOT / 'scripts' / 'native_packed_swin.py').read_text(encoding='utf-8')
    matrix = (ROOT / 'scripts' / 'native_matrix_fusion.py').read_text(encoding='utf-8')
    assert "resident_attention_family=f'c{self.channels}_ffn_attention_fp8a'" in packed
    assert 'wide_attention_from_fp8' in packed
    assert 'post_fp8.view(torch.float8_e4m3fn)' in matrix
    assert 'qkv=qkv.t().contiguous().t()' in matrix
    assert "qkv_fp8.stride()!=(1,c)" in matrix
    assert 'module._linear(value,module.project,seed)' in matrix


def test_supervised_chain_gate_requires_determinism_and_reports_error():
    source = (ROOT / 'scripts' / 'validate_resident_fp8_block_chain.py').read_text(encoding='utf-8')
    assert "'accuracy_track': 'approximate_deterministic'" in source
    assert "repeat['bitwise_exact']" in source
    assert "'nrmse': rmse / denominator" in source
    assert "require_gpu_tests_enabled('resident FP8 block-chain gate')" in source
    assert "except BaseException as error:" in source
