import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from profile_native_boundary_batch import compare_bytes,case_spec


def test_exact_comparison_is_strict_and_finite():
    a=np.array([0,1,-2],dtype='<f2').tobytes()
    assert compare_bytes(a,a)['bitwise_exact']
    b=np.array([0,1.001,-2],dtype='<f2').tobytes()
    assert not compare_bytes(a,b)['bitwise_exact']
    assert compare_bytes(a,b)['different_components']==1
    with pytest.raises(ValueError):compare_bytes(a,b[:2])
    with pytest.raises(ValueError):compare_bytes(b'',b'')
    with pytest.raises(ValueError):compare_bytes(a,np.array([0,1,np.nan],dtype='<f2').tobytes())


def test_unreviewed_case_rejected():
    with pytest.raises(ValueError):case_spec('4k')


def test_schedule_validation_before_network_or_gpu_work():
    torch=pytest.importorskip('torch')
    from native_whole_frame import SingleColorWholeFrame
    # Bypass parameter construction: invalid schedule must fail before any tensor work.
    model=SingleColorWholeFrame.__new__(SingleColorWholeFrame)
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError,match='positive integer'):
            next(model.stages(None,0,boundary_batch=invalid))
    for invalid in (0,True,64,2048):
        with pytest.raises(ValueError,match='query chunk'):
            next(model.stages(None,0,query_chunk=invalid))


def test_row_projection_preserves_formula_shape_and_gradients():
    torch=pytest.importorskip('torch')
    from native_preblock import project_input_features
    torch.manual_seed(8)
    x=(torch.randn(7,13,16)*.1).half().requires_grad_()
    w=(torch.randn(16,32)*.1).half().requires_grad_()
    expected=(x.float()@w.float()).half()
    for rows in (1,16,91,65536):
        got=project_input_features(x,w,rows_per_chunk=rows)
        assert torch.equal(got,expected)
        assert got.shape==(7,13,32)
    got.float().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
    assert torch.isfinite(w.grad).all() and torch.count_nonzero(w.grad)
    with pytest.raises(ValueError):project_input_features(x,w,rows_per_chunk=0)
    with pytest.raises(ValueError):project_input_features(x.float(),w)
