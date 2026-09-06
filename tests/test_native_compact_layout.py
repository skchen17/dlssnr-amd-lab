import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_compact_layout import compact_layout,enabled,pack,unpack
from native_multiscale import pack_image,unpack_image


@pytest.mark.parametrize('c',[32,64,128,256,512,1024])
@pytest.mark.parametrize('w,h',[(4,4),(12,20),(76,44)])
def test_compact_cell_layout_equals_original_every_element(w,h,c):
    x=torch.arange(h*w*c,dtype=torch.float32).reshape(h,w,c)
    expected=pack_image(x)
    assert torch.equal(pack(x),expected)
    assert torch.equal(unpack(expected,w,h,c),x)
    with compact_layout():
        assert torch.equal(pack_image(x),expected)
        assert torch.equal(unpack_image(expected,w,h,c),x)


def test_noncontiguous_input_and_gradients():
    x=torch.arange(12*8*32,dtype=torch.float32).reshape(12,8,32).transpose(0,1).requires_grad_()
    actual=unpack(pack(x),12,8,32)
    assert torch.equal(actual,x)
    actual.sum().backward()
    assert torch.equal(x.grad,torch.ones_like(x))


def test_context_restored_on_exception_and_bad_shapes_rejected():
    assert not enabled()
    with pytest.raises(RuntimeError):
        with compact_layout():
            assert enabled()
            raise RuntimeError('test')
    assert not enabled()
    for shape in [(3,4,32),(4,4,15),(0,4,32)]:
        with pytest.raises(ValueError):pack(torch.zeros(shape))
    with pytest.raises(ValueError):unpack(torch.zeros(10),4,4,32)
