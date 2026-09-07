import pathlib


ROOT=pathlib.Path(__file__).resolve().parents[1]


def test_wide_attention_norm_keeps_library_attention_boundaries():
    source=(ROOT/'scripts/native_window_attention.py').read_text(encoding='utf-8')
    assert 'window_attention_prepared' in source
    assert "norm_family=f'c{self.channels}_attention_norm'" in source
    assert 'native_gemm(q,k.transpose(-1,-2))' in source
    assert 'native_gemm(probability,v)' in source


def test_wide_families_have_bounded_norm_only_exports():
    source=(ROOT/'tools/native_fusion_probe/head_wmma.hip').read_text(encoding='utf-8')
    assert 'nr_c256_qkv_norm_only' in source
    assert 'wide_qkv_norm<256>' in source
    assert 'nr_c512_qkv_norm_only' in source
    assert 'wide_qkv_norm<512>' in source


def test_split_c512_attention_routes_the_stage_specific_norm():
    source=(ROOT/'scripts/native_split_swin512.py').read_text(encoding='utf-8')
    assert "'c512_attention_norm' in matrix.modules" in source
    assert 'window_attention_prepared(q, k, v, self.position_bias)' in source
