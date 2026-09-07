from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_resident_chain_keeps_semantic_boundary_and_direct_consumer():
    source=(ROOT/'tools/native_fp8_chain/resident_fp8_chain.hip').read_text()
    assert 'producer_fp16_boundary' in source and 'producer_fp8_boundary' in source
    resident=source.split('consumer_fp8_wmma',1)[1]
    assert '__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12' in resident
    assert 'static_cast<const uint8_t*>(activation)' in resident
    assert 'static_cast<const _Float16*>(activation)' in resident


def test_prototype_does_not_claim_fp16_boundary_replacement():
    source=(ROOT/'scripts/validate_resident_fp8_chain.py').read_text()
    assert "confirmed E4M3 only; residual/FP16 boundaries are unchanged" in source
    assert "does not claim whole-frame promotion" in source
