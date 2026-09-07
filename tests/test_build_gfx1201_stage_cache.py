import torch

from scripts.build_gfx1201_stage_cache import encode_tensor, with_native_derived_records


def test_qkv_is_prepacked_column_major_e4m3():
    value = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    raw, dtype, layout, shape = encode_tensor('encoder.5.block.attention.qkv', value)
    expected = value.t().contiguous().to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
    assert raw == expected
    assert (dtype, layout, shape) == ('e4m3fn', 'column_major_transposed_storage', [3, 4])


def test_residual_scale_stays_fp16_and_indices_become_int32():
    raw, dtype, layout, shape = encode_tensor(
        'encoder.5.block.ffn.residual_scale', torch.tensor([1, 2], dtype=torch.float16))
    assert dtype == 'fp16_le' and len(raw) == 4 and shape == [2]
    raw, dtype, _, shape = encode_tensor(
        'encoder.5.a_index', torch.tensor([[0, 7]], dtype=torch.int64))
    assert dtype == 'int32_le' and len(raw) == 8 and shape == [1, 2]


def test_native_inverse_permutation_is_derived_without_changing_source():
    source = {'x.ffn.permutation': torch.tensor([2, 0, 3, 1], dtype=torch.int64)}
    entries = dict(with_native_derived_records(source))
    assert entries['x.ffn.inverse_permutation'].tolist() == [1, 3, 0, 2]
    assert source['x.ffn.permutation'].tolist() == [2, 0, 3, 1]
