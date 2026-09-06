import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_matrix_fusion import validate_config,matrix_fusion,active_matrix_fusion


def test_profiles_and_unimplemented_modules_fail_closed():
    for profile in ('reference','wmma_fp16','wmma_fp8'):
        for waves in (1,2,4):
            validate_config(profile,('head_ffn',),waves)
            validate_config(profile,('head_ffn','head_output'),waves)
            validate_config(profile,('head_ffn','head_attention','head_output'),waves)
            validate_config(profile,('head_ffn','head_attention','head_softmax','head_output'),waves)
            validate_config(profile,('head_ffn','head_attention','head_softmax','head_output','pre_input'),waves)
            validate_config(profile,('head_ffn','head_attention','head_softmax','head_output','c32_ffn'),waves)
            validate_config(profile,('head_ffn','head_attention','head_softmax','head_output','c32_ffn','c32_attention'),waves)
            validate_config(profile,('c32_ffn','c32_attention_bounded'),waves)
            validate_config(profile,('c32_ffn','c32_attention_staged'),waves)
            validate_config(profile,('c32_ffn','c32_attention_core'),waves)
            validate_config(profile,('head_ffn','head_attention','head_softmax','head_output','c64_ffn','c128_ffn'),waves)
            validate_config(profile,('c64_ffn','c64_attention','c128_ffn','c128_attention'),waves)
            validate_config(profile,('c64_ffn','c64_attention_bounded','c128_ffn','c128_attention_bounded'),waves)
            validate_config(profile,('c256_ffn',),waves)
            validate_config(profile,('c512_ffn',),waves)
    for args in [('other',('head_ffn',),1),('wmma_fp8',('vit',),1),('reference',('head_ffn',),3)]:
        with pytest.raises(ValueError):validate_config(*args)
    with pytest.raises(ValueError,match='exactly one C32'):
        validate_config('wmma_fp16',('c32_attention_bounded','c32_attention_staged'),2)


def test_reference_is_noop_and_candidates_require_dll():
    with matrix_fusion():assert active_matrix_fusion() is None
    with pytest.raises(ValueError):
        with matrix_fusion(profile='wmma_fp16'):pass


def test_wmma_fragment_coordinates_cover_logical_output():
    # D fragment is [feature,token] for W^T X^T; store transposes logically,
    # without a global transposition pass.
    destinations=[]
    for tile in range(4):
        for lane in range(32):
            for m in range(2):
                for e in range(8):
                    row=tile*16+lane%16;col=m*16+(lane//16)*8+e
                    destinations.append(row*32+col)
    assert sorted(destinations)==list(range(2048))


def test_weight_prepack_invalidates_parameter_update():
    import torch,weakref
    from native_matrix_fusion import MatrixFusion
    from native_swin_torch import RecoveredSwin32
    model=RecoveredSwin32(bytes(21808),record_kind='head32').eval()
    op=object.__new__(MatrixFusion);op.cache=weakref.WeakKeyDictionary()
    with torch.no_grad():
        before=op.prepare(model)
        assert op.prepare(model) is before
        model.expand.add_(1)
        after=op.prepare(model)
        assert after is not before and torch.equal(after[0],torch.ones_like(after[0]))
        model.train()
        with pytest.raises(ValueError):op.prepare(model)


def test_pre_rejects_incomplete_matrix_tile_before_loading_gpu_runtime():
    import torch
    from native_matrix_fusion import MatrixFusion
    op=object.__new__(MatrixFusion);op.profile='wmma_fp16'
    with pytest.raises(ValueError,match='complete WMMA tiles'):
        op.pre_project(torch.zeros(1,17,16,dtype=torch.float16),torch.zeros(16,32,dtype=torch.float16))
