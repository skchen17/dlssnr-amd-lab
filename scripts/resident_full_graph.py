"""Persistent, single-threaded 640x360 original-weight graph executor.

This is an execution backend, not a temporal/DLSS-quality claim. No captured
activation, RTX injections, implicit resizing, or disk IO on the frame path.
The context, immutable model, modules, textures and patched ABIs live once.
"""
from __future__ import annotations

import ctypes as ct
import json
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

try:
    from . import run_full_graph_integrated as replay
except ImportError:
    import run_full_graph_integrated as replay

WIDTH, HEIGHT, FRAME_BYTES = 640, 360, 640 * 360 * 8


def validate_pixels(data: bytes) -> None:
    if len(data) != FRAME_BYTES:
        raise ValueError('expected exactly 640x360 tightly packed RGBA16F')
    if not np.isfinite(np.frombuffer(data, dtype='<f2')).all():
        raise ValueError('nonfinite RGBA16F pixels')


def checked_asset(root: Path, spec: dict) -> bytes:
    raw = replay.resolve_asset(root, spec).read_bytes()
    if len(raw) != int(spec['bytes']) or replay.sha256(raw) != spec['sha256'].upper():
        raise ValueError(f"asset size/hash mismatch: {spec['path']}")
    return raw


def load_assets(path: Path) -> tuple[dict, bytes, dict, list]:
    """Validate ALL immutable inputs before creating a device/context."""
    plan = json.loads(path.read_text(encoding='utf-8-sig'))
    slots = plan.get('slots', [])
    if len(slots) != 156 or [x['slot'] for x in slots] != list(range(156)):
        raise ValueError('expected contiguous 156-slot graph')
    if plan.get('diagnostic_injections'):
        raise ValueError('resident graph forbids diagnostic injections')
    if plan.get('activation_arena_bytes') != 29773824 or plan.get('final_output_bytes') != FRAME_BYTES:
        raise ValueError('unsupported captured geometry/arena')
    if slots[1]['special'] != 'zero_rgba16f_texture' or slots[154]['special'] != 'post_linear_surface_texture' or slots[155]['special'] != 'linear_final_copy':
        raise ValueError('live pre/post texture and final copy required')
    model = checked_asset(path.parent, plan['model_arena'])
    ptx, params = {}, []
    for slot in slots:
        raw = checked_asset(path.parent, slot['params'])
        if len(raw) != slot['param_size']:
            raise ValueError('parameter size mismatch')
        seen = set()
        for group, offset_key, limit in (
            ('activation_param_views', 'arena_offset', plan['activation_arena_bytes']),
            ('weight_param_views', 'weight_offset', len(model)),
        ):
            for view in slot[group]:
                pos, offset = int(view['param_offset']), int(view[offset_key])
                if pos < 0 or pos + 8 > len(raw) or not 0 <= offset < limit or any(p in seen for p in range(pos, pos + 8)):
                    raise ValueError('out-of-range or overlapping parameter view')
                seen.update(range(pos, pos + 8))
        if slot['special'] not in ('normal', 'zero_rgba16f_texture', 'post_linear_surface_texture', 'linear_final_copy'):
            raise ValueError('unsupported special ABI')
        if any(len(slot[k]) != 3 or any(type(v) is not int or v <= 0 for v in slot[k]) for k in ('grid', 'block')) or slot['dynamic_shared'] < 0:
            raise ValueError('invalid launch geometry')
        p = Path(slot['ptx'])
        p = str((p if p.is_absolute() else path.parent / p).resolve())
        if p not in ptx:
            ptx[p] = Path(p).read_bytes()
        if replay.sha256(ptx[p]) != slot['ptx_sha256'].upper():
            raise ValueError(f"slot {slot['slot']} PTX hash mismatch")
        if slot['slot'] in (1, 154) and b'tex.2d.' not in ptx[p]:
            raise ValueError('input texture sampling removed')
        slot['ptx'] = p
        params.append(raw)
    if struct.unpack_from('<2I', params[1], 208) != (HEIGHT, WIDTH):
        raise ValueError('preblock logical geometry mismatch')
    return plan, model, ptx, params


class ResidentGraph:
    def __init__(self, plan_path: Path, nvcuda: Path, expected_device='AMD Radeon RX 9070 XT', progress=None):
        started = time.perf_counter()
        self.plan, model, ptx, params = load_assets(plan_path)
        self.owner = threading.get_ident()
        self.busy = False
        self.failed = False
        self.closed = False
        self.frame_id = 0
        self.modules, self.arrays, self.textures, self.allocations, self.events = [], [], [], [], []
        self.dll_handles = []
        self.context = replay.CUcontext()
        self.api = None
        self.launches = []
        try:
            for folder in {nvcuda.parent, nvcuda.parent.parent}:
                if hasattr(os, 'add_dll_directory'):
                    self.dll_handles.append(os.add_dll_directory(str(folder)))
            a = self.api = replay.CudaApi(nvcuda)
            # Query + finite CPU deadline instead of an unbounded event synchronize.
            self.event_create = a._fn('cuEventCreate', [ct.POINTER(ct.c_void_p), ct.c_uint])
            self.event_record = a._fn('cuEventRecord', [ct.c_void_p, ct.c_void_p])
            self.event_query = a._fn('cuEventQuery', [ct.c_void_p])
            self.event_elapsed = a._fn('cuEventElapsedTime', [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p])
            self.event_destroy = a._fn('cuEventDestroy_v2', [ct.c_void_p])
            a.check('init', a.cuInit(0))
            device = replay.CUdevice()
            a.check('device', a.cuDeviceGet(ct.byref(device), 0))
            name = ct.create_string_buffer(256)
            a.check('device name', a.cuDeviceGetName(ct.addressof(name), len(name), device))
            self.device = name.value.decode('utf-8', 'replace')
            if expected_device.lower() not in self.device.lower():
                raise ValueError(f'device gate: {self.device}')
            a.check('context', a.cuCtxCreate(ct.byref(self.context), 0, device))
            functions = {}
            for index, (key, raw) in enumerate(ptx.items(), 1):
                module = replay.CUmodule()
                buf = ct.create_string_buffer(raw + b'\0')
                a.check('module load', a.cuModuleLoadData(ct.byref(module), ct.addressof(buf)))
                self.modules.append(module)
                for slot in self.plan['slots']:
                    lookup = (key, slot['function'])
                    if slot['ptx'] == key and lookup not in functions:
                        fn = replay.CUfunction()
                        a.check('function', a.cuModuleGetFunction(ct.byref(fn), module, slot['function'].encode()))
                        functions[lookup] = fn
                if progress:
                    progress({'module': index, 'total': len(ptx)})
            self.arena = self._allocate(self.plan['activation_arena_bytes'])
            self.model = self._allocate(len(model))
            self.output = self._allocate(FRAME_BYTES)
            replay.h_to_d(a, self.model.value, model, 'upload immutable weights ONCE')
            pre = self._texture(replay.CU_TR_ADDRESS_MODE_CLAMP, replay.CU_TR_FILTER_MODE_LINEAR)
            post = self._texture(replay.CU_TR_ADDRESS_MODE_BORDER, replay.CU_TR_FILTER_MODE_POINT)
            for slot, raw in zip(self.plan['slots'], params):
                patched = bytearray(raw)
                for v in slot['activation_param_views']:
                    struct.pack_into('<Q', patched, v['param_offset'], self.arena.value + v['arena_offset'])
                for v in slot['weight_param_views']:
                    struct.pack_into('<Q', patched, v['param_offset'], self.model.value + v['weight_offset'])
                replay.patch_special_parameters(patched, slot['special'], pre.value, self.output.value, post.value)
                buf = ct.create_string_buffer(bytes(patched), len(patched))
                ptrs = (ct.c_void_p * 1)(ct.addressof(buf))
                self.launches.append((functions[(slot['ptx'], slot['function'])], tuple(slot['grid']),
                                      tuple(slot['block']), slot['dynamic_shared'], buf, ptrs))
            for _ in range(2):
                event = ct.c_void_p()
                a.check('event create', self.event_create(ct.byref(event), 0))
                self.events.append(event)
            self.info = {'device': self.device, 'model_sha256': replay.sha256(model),
                         'module_count': len(self.modules), 'slot_count': len(self.launches),
                         'model_upload_count': 1, 'initialization_ms': (time.perf_counter() - started) * 1000,
                         'activation_init': 'zero_each_frame', 'rtx_intermediate_injection': False,
                         'temporal_inputs_bound': False, 'width': WIDTH, 'height': HEIGHT,
                         'arbitrary_network_shapes': False, 'dlss5_quality_verified': False}
        except BaseException:
            self.close()
            raise

    def _allocate(self, size):
        p = replay.CUdeviceptr()
        self.api.check('allocate', self.api.cuMemAlloc(ct.byref(p), size))
        self.allocations.append(p)
        return p

    def _texture(self, address, filtering):
        a = self.api
        arr = replay.CUarray()
        desc = replay.CUDA_ARRAY_DESCRIPTOR(WIDTH, HEIGHT, replay.CU_AD_FORMAT_HALF, 4)
        a.check('array', a.cuArrayCreate(ct.byref(arr), ct.byref(desc)))
        self.arrays.append(arr)
        resource = replay.CUDA_RESOURCE_DESC()
        resource.resType = replay.CU_RESOURCE_TYPE_ARRAY
        resource.res.array.hArray = arr
        desc = replay.CUDA_TEXTURE_DESC()
        desc.addressMode[:] = [address] * 3
        desc.filterMode = filtering
        desc.flags = replay.CU_TRSF_NORMALIZED_COORDINATES
        tex = replay.CUtexObject()
        a.check('texture', a.cuTexObjectCreate(ct.byref(tex), ct.byref(resource), ct.byref(desc), None))
        self.textures.append(tex)
        return tex

    def _upload(self, arr, data):
        host = ct.create_string_buffer(data, len(data))
        copy = replay.CUDA_MEMCPY2D()
        copy.srcMemoryType = replay.CU_MEMORYTYPE_HOST
        copy.srcHost = ct.addressof(host)
        copy.srcPitch = WIDTH * 8
        copy.dstMemoryType = replay.CU_MEMORYTYPE_ARRAY
        copy.dstArray = arr
        copy.WidthInBytes, copy.Height = WIDTH * 8, HEIGHT
        self.api.check('frame texture upload', self.api.cuMemcpy2D(ct.byref(copy)))

    def infer(self, data: bytes, *, post_data: bytes | None = None, timeout_seconds=10.0,
              diagnostic_sync_each=False) -> tuple[bytes, dict]:
        if threading.get_ident() != self.owner or self.closed or self.failed or self.busy:
            raise RuntimeError('resident graph wrong thread, closed, busy or failed')
        if not 0 < timeout_seconds <= 30:
            raise ValueError('frame deadline must be in (0,30] seconds')
        validate_pixels(data)
        post = data if post_data is None else post_data
        validate_pixels(post)
        self.busy = True
        started = time.perf_counter()
        a = self.api
        try:
            self._upload(self.arrays[0], data)
            self._upload(self.arrays[1], post)
            # GPU clear prevents stale state without a 29 MB host transfer.
            a.check('clear arena', a.cuMemsetD32(self.arena, 0, self.plan['activation_arena_bytes'] // 4))
            a.check('clear output', a.cuMemsetD32(self.output, 0, FRAME_BYTES // 4))
            a.check('event start', self.event_record(self.events[0], None))
            for fn, grid, block, shared, backing, ptrs in self.launches:
                a.check('frame launch', a.cuLaunchKernel(fn, *grid, *block, shared, None, ptrs, None))
                if diagnostic_sync_each:
                    a.check('diagnostic sync', a.cuCtxSynchronize())
            a.check('event end', self.event_record(self.events[1], None))
            while True:
                status = self.event_query(self.events[1])
                if status == 0:
                    break
                if status != 600:  # CUDA_ERROR_NOT_READY
                    a.check('event query', status)
                if time.perf_counter() - started > timeout_seconds:
                    raise TimeoutError('GPU completion deadline; poisoned worker must exit, not reuse buffers')
                time.sleep(0.001)
            elapsed = ct.c_float()
            a.check('GPU event elapsed', self.event_elapsed(ct.byref(elapsed), *self.events))
            result = replay.d_to_h(a, self.output.value, FRAME_BYTES, 'frame output')
            validate_pixels(result)
            if time.perf_counter() - started > timeout_seconds:
                raise TimeoutError('frame exceeded deadline; output not published')
            self.frame_id += 1
            return result, {'frame_id': self.frame_id, 'slot_count': len(self.launches),
                            'gpu_graph_ms': elapsed.value,
                            'host_frame_ms': (time.perf_counter() - started) * 1000,
                            'gpu_timing_scope': '156 launches; excludes upload, clears, readback',
                            'diagnostic_sync_each': diagnostic_sync_each,
                            'input_sha256': replay.sha256(data), 'post_sha256': replay.sha256(post),
                            'output_sha256': replay.sha256(result)}
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def close(self):
        if self.closed:
            return
        if threading.get_ident() != self.owner or self.busy:
            raise RuntimeError('close requires idle owner thread')
        self.closed = True
        # On uncertain completion do not destroy possibly in-flight objects.
        # Caller must terminate its dedicated worker; this is not a GPU cancel.
        if self.failed:
            return
        a = self.api
        if a:
            for e in self.events:
                self.event_destroy(e)
            for t in reversed(self.textures):
                a.cuTexObjectDestroy(t)
            for arr in reversed(self.arrays):
                a.cuArrayDestroy(arr)
            for p in reversed(self.allocations):
                a.cuMemFree(p)
            for m in reversed(self.modules):
                a.cuModuleUnload(m)
            if self.context.value:
                a.cuCtxDestroy(self.context)
        for handle in self.dll_handles:
            handle.close()
