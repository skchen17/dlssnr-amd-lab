"""Exhaustively test compiled conversion without initializing GPU."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from native_grouped_ffn import cubic_silu
from native_swin_torch import quantize_e4

p=argparse.ArgumentParser();p.add_argument('--dll',required=True)
p.add_argument('--operation',choices=('quantize','cubic_quantize'),default='quantize');a=p.parse_args()
lib=ct.CDLL(a.dll)
f=getattr(lib,'native_fusion_'+a.operation+'_host')
f.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64];f.restype=None
bits=torch.arange(65536,dtype=torch.int32).to(torch.uint16)
output=torch.empty_like(bits)
f(bits.data_ptr(),output.data_ptr(),bits.numel())
x=bits.view(torch.float16)
reference=(quantize_e4(cubic_silu(x)) if a.operation=='cubic_quantize' else quantize_e4(x)).view(torch.uint16)
valid=~torch.isnan(reference.view(torch.float16))
mismatches=int((output.view(torch.int16)[valid]!=reference.view(torch.int16)[valid]).sum())
assert mismatches==0,mismatches
assert torch.isnan(output.view(torch.float16)[~valid]).all()
assert not torch.cuda.is_initialized()
print(json.dumps({'pass':True,'operation':a.operation,'input_patterns':65536,'non_nan_reference_patterns':int(valid.sum()),'nan_reference_patterns':int((~valid).sum()),'nan_from_non_nan_input':int((~valid & ~torch.isnan(x)).sum()),'bit_mismatches':mismatches,'gpu_initialized':False}))
