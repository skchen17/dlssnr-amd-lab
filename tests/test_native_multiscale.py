import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_multiscale import (DOWNSAMPLE_RECORDS, RecoveredDownsampleSwin,
                               average_pool2x2, pack_image, unpack_image, pack_planar16, unpack_planar16,
                               pack_outview, unpack_outview, outview_channels, RecoveredOutviewSwin)


@pytest.mark.parametrize('width,height,channels', [(20, 12, 512), (12, 8, 1024), (8, 8, 32)])
def test_image_layout_roundtrip(width, height, channels):
    x = torch.arange(height*width*channels).reshape(height, width, channels).float()
    assert torch.equal(unpack_image(pack_image(x), width, height, channels), x)


def test_pool_uses_global_neighbours_and_has_gradients():
    x = torch.arange(64).reshape(8, 8, 1).half().requires_grad_()
    result = average_pool2x2(x)
    assert result[0, 0].item() == 4.5 and result[-1, -1].item() == 58.5
    result.float().sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, .25))


def test_planar16_roundtrip_is_not_packed_cell_layout():
    x = torch.arange(8*12*64).reshape(8, 12, 64).float()
    packed = pack_planar16(x)
    assert torch.equal(unpack_planar16(packed, 12, 8, 64), x)
    assert not torch.equal(packed, pack_image(x))


def test_outview_uses_mma_pair_order_not_plain_channel_order():
    x = torch.arange(8*12*64).reshape(8, 12, 64).float().requires_grad_()
    stored = pack_outview(x)
    assert stored[:16].tolist() == [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
    assert torch.equal(unpack_outview(stored, 12, 8, 64), x)
    assert not torch.equal(stored, pack_planar16(x))
    stored.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_outview_rejects_wrong_or_empty_storage():
    for w,h,c in [(0,4,32),(4,4,15),(4,4,32)]:
        with pytest.raises(ValueError):
            unpack_outview(torch.zeros(16), w,h,c)


@pytest.mark.parametrize('kind', list(DOWNSAMPLE_RECORDS))
def test_downsample_has_distinct_skip_and_next_scale(kind):
    model = RecoveredDownsampleSwin(bytes(DOWNSAMPLE_RECORDS[kind]), record_kind=kind)
    c = model.channels
    output = model(torch.zeros(8*8*c).half(), 8, 8, -4, -4)
    assert output['skip'].numel() == 64*c and output['downsampled'].numel() == 32*c
    assert output['resident'] is output['downsampled']
    assert output['target_size'] == (4, 4)
    assert torch.count_nonzero(output['downsampled']) == 0


@pytest.mark.parametrize('kind',list(DOWNSAMPLE_RECORDS))
def test_resident_downsample_omits_debug_outview(kind):
    model=RecoveredDownsampleSwin(bytes(DOWNSAMPLE_RECORDS[kind]),record_kind=kind)
    c=model.channels
    output=model(torch.zeros(8*8*c).half(),8,8,-4,-4,capture_layouts=False)
    assert set(output)=={'skip','resident','source_size','target_size'}


def test_malformed_geometry_rejected():
    with pytest.raises(ValueError):
        pack_image(torch.zeros(7, 8, 32))
    with pytest.raises(ValueError):
        average_pool2x2(torch.zeros(8, 7, 32).half())
