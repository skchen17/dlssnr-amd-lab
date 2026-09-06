"""CPU-only fixed-range static NR/FSR candidate agreement, not teacher quality."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def uniform_mean_valid(image, window=11):
    if image.ndim!=2 or type(window) is not int or window<=0 or min(image.shape)<window:
        raise ValueError('2D image and positive fitting window required')
    integral=np.pad(np.asarray(image,dtype=np.float64),((1,0),(1,0)))
    np.cumsum(integral,axis=0,out=integral)
    np.cumsum(integral,axis=1,out=integral)
    return (integral[window:,window:]-integral[:-window,window:]
            -integral[window:,:-window]+integral[:-window,:-window])/(window*window)


def describe(image):
    rgb=image[...,:3];finite=np.isfinite(rgb)
    values=rgb[finite]
    return dict(all_finite=bool(finite.all()),finite_fraction=float(finite.mean()),
                minimum=float(values.min()) if values.size else None,
                maximum=float(values.max()) if values.size else None,
                below_zero_fraction=float((rgb<0).mean()),above_one_fraction=float((rgb>1).mean()),
                outside_unit_range_fraction=float(((rgb<0)|(rgb>1)).mean()))


def compare(reference,candidate,window=11):
    if reference.shape!=candidate.shape or reference.ndim!=3 or reference.shape[-1]!=4:
        raise ValueError('matching HxWx4 RGBA arrays required')
    if min(reference.shape[:2])<window:raise ValueError('SSIM valid window does not fit')
    result=dict(reference=describe(reference),candidate=describe(candidate),
                color_channels='RGB only',data_range=1.0,clipping=False,normalization=False,
                ssim_definition='mean RGB; valid 11x11 uniform population moments; K1=.01 K2=.03',
                ssim_window=window)
    if not result['reference']['all_finite'] or not result['candidate']['all_finite']:
        result.update(status='INVALID_NONFINITE',psnr_db=None,mae=None,ssim=None)
        return result
    squared_sum=absolute_sum=0.;ssims=[]
    for channel in range(3):
        x=np.asarray(reference[...,channel],dtype=np.float64)
        y=np.asarray(candidate[...,channel],dtype=np.float64)
        delta=x-y;squared_sum+=float(np.square(delta).sum());absolute_sum+=float(np.abs(delta).sum())
        mx,my=uniform_mean_valid(x,window),uniform_mean_valid(y,window)
        vx=uniform_mean_valid(x*x,window)-mx*mx
        vy=uniform_mean_valid(y*y,window)-my*my
        # Only remove negative numerical roundoff in variance, not image values.
        np.maximum(vx,0,out=vx);np.maximum(vy,0,out=vy)
        covariance=uniform_mean_valid(x*y,window)-mx*my
        score=((2*mx*my+.01**2)*(2*covariance+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
        ssims.append(float(score.mean()))
    count=reference.shape[0]*reference.shape[1]*3;mse=squared_sum/count
    result.update(status='MEASURED_NOT_QUALITY_GATE',mse=mse,mae=absolute_sum/count,
                  psnr_db=float(-10*np.log10(mse)) if mse>0 else None,
                  exact_rgb=bool(mse==0),psnr_infinite=bool(mse==0),ssim=float(np.mean(ssims)),ssim_rgb=ssims)
    return result


def read_rgba16f(path,width,height):
    if path.stat().st_size!=width*height*8:raise ValueError('RGBA16F byte count mismatch: '+str(path))
    return np.memmap(path,dtype='<f2',mode='r',shape=(height,width,4))


def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest.hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('refuse overwrite of existing report')
    names={'nr_4k':'20260906_nr_resolution2160_v1','nr_1080_fsr_4k':'20260906_nr_fsr1080_twelve_v1',
           'nr_1440_fsr_4k':'20260906_nr_fsr1440_twelve_v1'}
    paths={k:args.root/'results'/v/'output.rgba16f' for k,v in names.items()}
    reference=read_rgba16f(paths['nr_4k'],3840,2160)
    result=dict(schema=1,scope='synthetic static RGB agreement relative to current 4K NR candidate; not RTX or game quality',
                temporal_quality_measured=False,hdr_quality_measured=False,quality_gate_passed=False,
                width=3840,height=2160,sources={k:fingerprint(v) for k,v in paths.items()},
                evaluator=fingerprint(Path(__file__)),reference=describe(reference),comparisons={})
    for key,path in paths.items():
        if key=='nr_4k':continue
        result['comparisons'][key]=compare(reference,read_rgba16f(path,3840,2160))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
