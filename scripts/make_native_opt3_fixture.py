"""Explicit CPU synthetic 4K color fixture; never a game capture or teacher sample."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def make_fixture(output):
    output.mkdir(parents=True,exist_ok=False)
    w,h=3840,2160
    x=np.arange(w)[None,:];y=np.arange(h)[:,None]
    color=np.empty((h,w,4),dtype='<f2')
    color[...,0]=(x%257)/256
    color[...,1]=(y%251)/250
    color[...,2]=((x//32+y//32)%2)*.5+.125
    color[...,3]=1
    raw=color.tobytes()
    with (output/'input.rgba16f').open('xb') as f:f.write(raw)
    manifest={'input_geometry':[w,h],'input_sha256':hashlib.sha256(raw).hexdigest().upper(),
              'input_source':'SYNTHETIC_CPU_GRADIENT_CHECKER_NOT_GAME_NOT_TEACHER',
              'recipe':'R=(x%257)/256;G=(y%251)/250;B=((x//32+y//32)%2)*.5+.125;A=1;HWC little-endian FP16',
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()}
    with (output/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    make_fixture(p.parse_args().output)
