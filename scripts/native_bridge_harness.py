"""Offline native shared-texture transport harness; never loaded into the game.

Tensor copy endpoints are GPU pointers allocated by the owning PyTorch runtime.
CPU fixture upload/readback is for transport verification only.
"""
import ctypes as ct
from pathlib import Path
import torch


class SharedTextureHarness:
    def __init__(self,dll,width,height,*,producer_only=False,adapter_luid=0):
        self.library=ct.CDLL(str(Path(dll).resolve()))
        self.width,self.height=width,height; self.bytes=width*height*8
        lib=self.library
        lib.nr_bridge_error.restype=ct.c_char_p
        lib.nr_bridge_create.argtypes=[ct.c_uint,ct.c_uint];lib.nr_bridge_create.restype=ct.c_void_p
        for name in ('upload','read'):
            f=getattr(lib,'nr_bridge_'+name);f.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64];f.restype=ct.c_int
        for name in ('to_tensor','from_tensor'):
            f=getattr(lib,'nr_bridge_'+name);f.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_void_p];f.restype=ct.c_int
        lib.nr_bridge_close.argtypes=[ct.c_void_p];lib.nr_bridge_close.restype=ct.c_int
        lib.nr_bridge_next.argtypes=[ct.c_void_p];lib.nr_bridge_next.restype=ct.c_int
        for name,kind in [('pitch',ct.c_uint),('imported_pointer',ct.c_uint64),('d3d_address',ct.c_uint64)]:
            f=getattr(lib,'nr_bridge_'+name);f.argtypes=[ct.c_void_p];f.restype=kind
        if producer_only:
            lib.nr_bridge_create_producer.argtypes=[ct.c_uint,ct.c_uint,ct.c_uint64];lib.nr_bridge_create_producer.restype=ct.c_void_p
            self.handle=lib.nr_bridge_create_producer(width,height,adapter_luid)
        else:self.handle=lib.nr_bridge_create(width,height)
        if not self.handle: self.check(-1)
        self.source_keepalive=None

    def check(self,code):
        if code: raise RuntimeError(self.library.nr_bridge_error().decode())

    def upload_fixture(self,raw):
        if len(raw)!=self.bytes: raise ValueError('fixture size')
        source=ct.create_string_buffer(raw)
        self.check(self.library.nr_bridge_upload(self.handle,source,self.bytes))

    def tensor(self,value):
        if (not torch.version.hip or value.device.type!='cuda' or value.device.index!=torch.cuda.current_device()
                or value.dtype!=torch.float16 or tuple(value.shape)!=(self.height,self.width,4) or not value.is_contiguous()):
            raise ValueError('owning GPU contiguous RGBA16F required')

    def to_tensor(self,target):
        self.tensor(target)
        self.check(self.library.nr_bridge_to_tensor(self.handle,target.data_ptr(),self.bytes,torch.cuda.current_stream().cuda_stream))
        target.record_stream(torch.cuda.current_stream())

    def from_tensor(self,source):
        self.tensor(source); self.source_keepalive=source
        self.check(self.library.nr_bridge_from_tensor(self.handle,source.data_ptr(),self.bytes,torch.cuda.current_stream().cuda_stream))
        source.record_stream(torch.cuda.current_stream())

    def read_fixture(self):
        result=ct.create_string_buffer(self.bytes)
        self.check(self.library.nr_bridge_read(self.handle,result,self.bytes))
        self.source_keepalive=None
        return result.raw

    def description(self):
        lib=self.library
        return {'row_pitch':lib.nr_bridge_pitch(self.handle),
                'hip_imported_pointer':lib.nr_bridge_imported_pointer(self.handle),
                'd3d12_gpu_address':lib.nr_bridge_d3d_address(self.handle),
                'same_adapter_luid_checked':True,'separate_input_output_fences':True,
                'game_transport':False,'cpu_fixture_io_only':True}

    def close(self):
        self.check(self.library.nr_bridge_close(self.handle));self.handle=None

    def next_frame(self):
        self.check(self.library.nr_bridge_next(self.handle))
