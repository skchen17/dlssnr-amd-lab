import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_nr_fsr_static import compare,uniform_mean_valid


def test_uniform_population_valid():
    image=np.arange(25).reshape(5,5)
    assert np.array_equal(uniform_mean_valid(image,3),np.array([[6,7,8],[11,12,13],[16,17,18]]))


def test_identity_and_alpha_ignored():
    x=np.random.default_rng(1).random((13,14,4));y=x.copy();y[...,3]=10
    r=compare(x,y)
    assert r['exact_rgb'] and r['psnr_infinite'] and r['psnr_db'] is None
    assert r['ssim']==pytest.approx(1,abs=1e-12)


def test_constant_offset_no_clip_or_normalize():
    x=np.ones((11,11,4));y=x+1
    r=compare(x,y)
    assert r['mae']==1 and r['mse']==1 and r['psnr_db']==0
    assert r['candidate']['above_one_fraction']==1
    assert r['ssim']==pytest.approx((4+.0001)/(5+.0001))


def test_nonfinite_and_small_shape():
    x=np.ones((11,11,4));y=x.copy();y[0,0,0]=np.nan
    assert compare(x,y)['status']=='INVALID_NONFINITE'
    with pytest.raises(ValueError):compare(x[:4],x[:4])
