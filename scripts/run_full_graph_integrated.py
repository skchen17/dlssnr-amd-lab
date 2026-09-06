#!/usr/bin/env python3
"""Execute the captured 156-slot graph in one ZLUDA context and one arena."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import math
import struct
import subprocess
import sys
import time
from pathlib import Path


CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUarray = ctypes.c_void_p
CUtexObject = ctypes.c_uint64

CUDA_SUCCESS = 0
CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_ARRAY = 3
CU_AD_FORMAT_HALF = 16
CU_RESOURCE_TYPE_ARRAY = 0
CU_TR_ADDRESS_MODE_CLAMP = 1
CU_TR_ADDRESS_MODE_BORDER = 3
CU_TR_FILTER_MODE_POINT = 0
CU_TR_FILTER_MODE_LINEAR = 1
CU_TRSF_NORMALIZED_COORDINATES = 2


class CUDA_ARRAY_DESCRIPTOR(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_size_t), ("Height", ctypes.c_size_t),
                ("Format", ctypes.c_uint), ("NumChannels", ctypes.c_uint)]


class CUDA_MEMCPY2D(ctypes.Structure):
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_uint), ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t), ("dstXInBytes", ctypes.c_size_t),
        ("dstY", ctypes.c_size_t), ("dstMemoryType", ctypes.c_uint),
        ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64),
        ("dstArray", ctypes.c_void_p), ("dstPitch", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t),
    ]


class _ArrayResource(ctypes.Structure):
    _fields_ = [("hArray", ctypes.c_void_p)]


class _ResourceUnion(ctypes.Union):
    _fields_ = [("array", _ArrayResource), ("reserved", ctypes.c_int * 32)]


class CUDA_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [("resType", ctypes.c_uint), ("padding", ctypes.c_uint),
                ("res", _ResourceUnion), ("flags", ctypes.c_uint)]


class CUDA_TEXTURE_DESC(ctypes.Structure):
    _fields_ = [
        ("addressMode", ctypes.c_uint * 3), ("filterMode", ctypes.c_uint),
        ("flags", ctypes.c_uint), ("maxAnisotropy", ctypes.c_uint),
        ("mipmapFilterMode", ctypes.c_uint), ("mipmapLevelBias", ctypes.c_float),
        ("minMipmapLevelClamp", ctypes.c_float), ("maxMipmapLevelClamp", ctypes.c_float),
        ("borderColor", ctypes.c_float * 4), ("reserved", ctypes.c_int * 12),
    ]


class CudaFailure(RuntimeError):
    def __init__(self, operation: str, code: int, detail: str):
        super().__init__(f"{operation} failed with CUDA {code}: {detail}")
        self.operation = operation
        self.code = code
        self.detail = detail


class CudaApi:
    def __init__(self, dll_path: Path):
        self.dll = ctypes.WinDLL(str(dll_path))
        self.cuInit = self._fn("cuInit", [ctypes.c_uint])
        self.cuDeviceGetCount = self._fn("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
        self.cuDeviceGet = self._fn("cuDeviceGet", [ctypes.POINTER(CUdevice), ctypes.c_int])
        self.cuDeviceGetName = self._fn("cuDeviceGetName", [ctypes.c_void_p, ctypes.c_int, CUdevice])
        self.cuCtxCreate = self._fn("cuCtxCreate_v2", [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice])
        self.cuCtxDestroy = self._fn("cuCtxDestroy_v2", [CUcontext])
        self.cuModuleLoadData = self._fn("cuModuleLoadData", [ctypes.POINTER(CUmodule), ctypes.c_void_p])
        self.cuModuleGetFunction = self._fn("cuModuleGetFunction", [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p])
        self.cuModuleUnload = self._fn("cuModuleUnload", [CUmodule])
        self.cuMemAlloc = self._fn("cuMemAlloc_v2", [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t])
        self.cuMemFree = self._fn("cuMemFree_v2", [CUdeviceptr])
        self.cuMemsetD32 = self._fn("cuMemsetD32_v2", [CUdeviceptr, ctypes.c_uint, ctypes.c_size_t])
        self.cuMemcpyDtoH = self._fn("cuMemcpyDtoH_v2", [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t])
        self.cuMemcpyHtoD = self._fn("cuMemcpyHtoD_v2", [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t])
        self.cuArrayCreate = self._fn("cuArrayCreate_v2", [ctypes.POINTER(CUarray), ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR)])
        self.cuArrayDestroy = self._fn("cuArrayDestroy", [CUarray])
        self.cuMemcpy2D = self._fn("cuMemcpy2D_v2", [ctypes.POINTER(CUDA_MEMCPY2D)])
        self.cuTexObjectCreate = self._fn("cuTexObjectCreate", [ctypes.POINTER(CUtexObject), ctypes.POINTER(CUDA_RESOURCE_DESC), ctypes.POINTER(CUDA_TEXTURE_DESC), ctypes.c_void_p])
        self.cuTexObjectDestroy = self._fn("cuTexObjectDestroy", [CUtexObject])
        self.cuLaunchKernel = self._fn("cuLaunchKernel", [CUfunction,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
        self.cuCtxSynchronize = self._fn("cuCtxSynchronize", [])
        self.cuGetErrorName = self._fn("cuGetErrorName", [CUresult, ctypes.POINTER(ctypes.c_char_p)])
        self.cuGetErrorString = self._fn("cuGetErrorString", [CUresult, ctypes.POINTER(ctypes.c_char_p)])

    def _fn(self, name: str, args: list):
        function = getattr(self.dll, name)
        function.argtypes = args
        function.restype = CUresult
        return function

    def describe(self, code: int) -> str:
        name = ctypes.c_char_p()
        text = ctypes.c_char_p()
        self.cuGetErrorName(code, ctypes.byref(name))
        self.cuGetErrorString(code, ctypes.byref(text))
        n = name.value.decode("utf-8", "replace") if name.value else "UNKNOWN"
        t = text.value.decode("utf-8", "replace") if text.value else "no description"
        return f"{n}: {t}"

    def check(self, operation: str, code: int) -> None:
        if code != CUDA_SUCCESS:
            raise CudaFailure(operation, int(code), self.describe(int(code)))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def resolve_asset(root: Path, spec: dict) -> Path:
    path = Path(spec["path"])
    return path if path.is_absolute() else root / path


def load_color_input(path: Path | None) -> tuple[bytes, dict]:
    """Keep this replay's captured geometry; external pixels are an explicit experiment."""
    data = bytes(640 * 360 * 8) if path is None else path.read_bytes()
    if len(data) != 640 * 360 * 8:
        raise ValueError("input must be tightly packed 640x360 RGBA16F; no implicit resizing")
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<e', data)):
        raise ValueError("input contains NaN/Inf")
    return data, {"path": str(path.resolve()) if path else None,
                  "width": 640, "height": 360, "format": "RGBA16F",
                  "bytes": len(data), "sha256": sha256(data),
                  "external": path is not None,
                  "sampler": "linear_clamp_normalized_coordinates",
                  "temporal_inputs_bound": False}


def h_to_d(api: CudaApi, dst: int, data: bytes, operation: str) -> None:
    host = ctypes.create_string_buffer(data, len(data))
    api.check(operation, api.cuMemcpyHtoD(dst, ctypes.addressof(host), len(data)))


def d_to_h(api: CudaApi, src: int, size: int, operation: str) -> bytes:
    host = ctypes.create_string_buffer(size)
    api.check(operation, api.cuMemcpyDtoH(ctypes.addressof(host), src, size))
    return host.raw


def write_progress(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_diagnostic_injections(plan: dict, root: Path) -> dict[int, tuple[dict, bytes]]:
    """Validate explicitly non-S7 RTX state injections before GPU execution."""
    result = {}
    for injection in plan.get("diagnostic_injections", []):
        slot = int(injection["after_slot"])
        if not 0 <= slot < 155 or slot in result:
            raise ValueError(f"invalid or duplicate diagnostic injection after slot {slot}")
        size = int(injection["bytes"])
        reference_offset = int(injection.get("reference_offset", 0))
        raw = resolve_asset(root, injection["reference"]).read_bytes()
        if reference_offset < 0 or len(raw) < reference_offset + size:
            raise ValueError(f"diagnostic injection after slot {slot} is too short")
        data = raw[reference_offset:reference_offset + size]
        if sha256(data) != injection["reference"]["sha256"]:
            raise ValueError(f"diagnostic injection after slot {slot} hash differs")
        result[slot] = (injection, data)
    return result


def patch_special_parameters(params: bytearray, special: str,
                             zero_texture: int, final_output: int,
                             post_texture: int = 0) -> None:
    """Patch only resources whose captured handles cannot remain valid on replay."""
    if special == "zero_rgba16f_texture":
        struct.pack_into("<Q", params, 0, zero_texture)
        for offset in (8, 16, 24, 32):
            struct.pack_into("<Q", params, offset, 0)
    elif special == "post_linear_surface":
        # The surface itself is already patched through activation_param_views.
        # Offset 56 must retain its captured scalar/resource value: clearing it
        # changes RGB by about 0.5 even after texture samples are lowered to zero.
        pass
    elif special == "post_linear_surface_texture":
        if not post_texture:
            raise ValueError("post-block texture replay requested without a texture object")
        # Captured offset 56 is a merged texture/sampler object, not a scalar.
        struct.pack_into("<Q", params, 56, post_texture)
    elif special == "linear_final_copy":
        struct.pack_into("<Q", params, 8, final_output)


NATIVE_SWIN_FUNCTION = "cc_tinlayout_fused_swin_1h_32_1_chained_fp8"


def run_native_swin(api, arena: int, model_bytes: bytes, record: dict,
                    params: bytes, executable: Path, output: Path) -> dict:
    """Diagnostic CPU handoff between APIs, not a resident runtime bridge."""
    if record["function"] != NATIVE_SWIN_FUNCTION or len(params) != 96:
        raise ValueError("unsupported native family ABI")
    height, width, ox, oy = struct.unpack_from("<4i", params, 24)
    if width % 8 or height % 8 or ox not in (-4, 0) or oy not in (-4, 0):
        raise ValueError("unsupported native geometry")
    grid = [(width - ox + 7) // 8, (height - oy + 7) // 8, 1]
    if record["grid"] != grid or record["block"] != [32, 1, 1] or record["dynamic_shared"]:
        raise ValueError("native family launch mismatch")
    views = {v["param_offset"]: v["arena_offset"] for v in record["activation_param_views"]}
    weight = next(v["weight_offset"] for v in record["weight_param_views"] if v["param_offset"] == 16)
    inputs = d_to_h(api, arena + views[0], width * height * 32, "native family input")
    output.mkdir()
    (output / "input.e4").write_bytes(inputs)
    (output / "weights.raw").write_bytes(model_bytes[weight:weight + 20672])
    subprocess.run([str(executable), str(output / "input.e4"), str(output / "weights.raw"),
                    str(width), str(height), str(ox), str(oy), str(output)], check=True)
    inference = json.loads((output / "manifest.json").read_text())
    if inference["status"] != "EXECUTION_PASS":
        raise ValueError("native family failed")
    data = (output / "output.e4").read_bytes()
    if len(data) != len(inputs):
        raise ValueError("native family output length mismatch")
    h_to_d(api, arena + views[8], data, "native family output")
    # The sequential harness has completed the whole producer before releasing
    # its tiled synchronization words. This is not a GPU marker substitute.
    h_to_d(api, arena + views[56], bytes(grid[0] * grid[1] * 4), "native family completion")
    return {"backend": "D3D12_DXIL", "diagnostic_cpu_handoff": True,
            "input_sha256": sha256(inputs), "output_sha256": sha256(data),
            "producer": "AMD_live_upstream_arena", "inference": inference}


def run(plan_path: Path, nvcuda_path: Path, output_dir: Path,
        expected_device: str, repetitions: int, export_pre_head: bool = False,
        native_swin: Path | None = None, input_rgba16f: Path | None = None,
        activation_init: str = 'captured') -> dict:
    if activation_init not in ('captured', 'zero'):
        raise ValueError('unsupported activation initialization')
    root = plan_path.parent
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if len(plan.get("slots", [])) != 156 or [int(x["slot"]) for x in plan["slots"]] != list(range(156)):
        raise ValueError("plan is not a literal contiguous 156-slot graph")
    injections = load_diagnostic_injections(plan, root)
    planned_color = plan.get('color_input')
    if planned_color:
        planned_path = resolve_asset(root, planned_color)
        if input_rgba16f and input_rgba16f.resolve() != planned_path.resolve():
            raise ValueError('input override disagrees with color plan')
        input_rgba16f = planned_path
    input_bytes, input_info = load_color_input(input_rgba16f)
    if planned_color and input_info['sha256'] != planned_color['sha256'].upper():
        raise ValueError('input hash differs from color plan')
    if input_rgba16f and injections:
        raise ValueError("external color experiments cannot inject reference intermediates")
    if input_rgba16f:
        slot1 = plan['slots'][1]
        if slot1['special'] != 'zero_rgba16f_texture':
            raise ValueError("unsupported preblock input binding")
        params = resolve_asset(root, slot1['params']).read_bytes()
        if struct.unpack_from('<2I', params, 208) != (360, 640):
            raise ValueError("preblock logical dimensions differ from supplied input")
    post_texture_spec = plan.get("post_texture")
    post_texture_bytes = None
    if post_texture_spec is not None:
        post_texture_bytes = resolve_asset(root, post_texture_spec).read_bytes()
        expected_texture_bytes = (
            int(post_texture_spec["width"]) * int(post_texture_spec["height"]) * 8)
        if len(post_texture_bytes) != expected_texture_bytes:
            raise ValueError("post texture size differs from RGBA16F dimensions")
        if sha256(post_texture_bytes) != post_texture_spec["sha256"]:
            raise ValueError("post texture hash differs from plan")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "execution.json"
    report = {
        "schema": 1,
        "experiment": "rx9070xt_full_graph_single_context_156_launch_execution",
        "status": "RUNNING",
        "pass": False,
        "plan": str(plan_path),
        "nvcuda": str(nvcuda_path),
        "expected_device": expected_device,
        "single_context": native_swin is None,
        "shared_activation_arena": True,
        "native_swin_enabled": native_swin is not None,
        "diagnostic_cpu_handoff": native_swin is not None,
        "same_list_full_graph": False,
        "native_graph_complete": False,
        "exports_pre_head_activation": export_pre_head,
        "color_input": input_info,
        "same_input_rtx_reference_available": False if input_rgba16f else None,
        "real_scene_quality_verified": False,
        "activation_initialization": "captured_frame_start" if activation_init == 'captured' else 'zero',
        "shared_model_arena": True,
        "synchronize_after_every_launch": True,
        "rtx_intermediate_state_injection": bool(injections),
        "diagnostic_injections": [
            {"after_slot": slot, "arena_offset": int(spec["arena_offset"]),
             "reference_offset": int(spec.get("reference_offset", 0)),
             "bytes": len(data), "sha256": sha256(data)}
            for slot, (spec, data) in sorted(injections.items())
        ],
        "post_texture_replay": post_texture_spec is not None,
        "post_texture": post_texture_spec,
        "modules": [],
        "runs": [],
    }
    write_progress(progress_path, report)

    arena_initial = resolve_asset(root, plan["activation_arena_initial"]).read_bytes()
    model_initial = resolve_asset(root, plan["model_arena"]).read_bytes()
    if len(arena_initial) != int(plan["activation_arena_bytes"]):
        raise ValueError("activation arena size differs from plan")
    if sha256(arena_initial) != plan["activation_arena_initial"]["sha256"]:
        raise ValueError("activation arena hash differs from plan")
    report['captured_activation_sha256'] = sha256(arena_initial)
    if activation_init == 'zero':
        arena_initial = bytes(len(arena_initial))
    report['runtime_activation_initial_sha256'] = sha256(arena_initial)
    if sha256(model_initial) != plan["model_arena"]["sha256"]:
        raise ValueError("model arena hash differs from plan")
    report["model_sha256"] = sha256(model_initial)

    dll_handles = []
    for directory in {nvcuda_path.parent, nvcuda_path.parent.parent}:
        if directory.is_dir() and hasattr(os, "add_dll_directory"):
            dll_handles.append(os.add_dll_directory(str(directory)))
    api = CudaApi(nvcuda_path)
    context = CUcontext()
    modules: dict[str, CUmodule] = {}
    functions: dict[tuple[str, str], CUfunction] = {}
    arena = CUdeviceptr()
    model = CUdeviceptr()
    final_output = CUdeviceptr()
    texture_array = CUarray()
    zero_texture = CUtexObject()
    post_texture_array = CUarray()
    post_texture = CUtexObject()
    device_name = ""
    active_slot = None
    try:
        api.check("cuInit", api.cuInit(0))
        count = ctypes.c_int()
        api.check("cuDeviceGetCount", api.cuDeviceGetCount(ctypes.byref(count)))
        if count.value < 1:
            raise RuntimeError("ZLUDA exposed no CUDA device")
        device = CUdevice()
        api.check("cuDeviceGet", api.cuDeviceGet(ctypes.byref(device), 0))
        name_buffer = ctypes.create_string_buffer(256)
        api.check("cuDeviceGetName", api.cuDeviceGetName(ctypes.addressof(name_buffer), len(name_buffer), device))
        device_name = name_buffer.value.decode("utf-8", "replace")
        report["device"] = device_name
        if expected_device.lower() not in device_name.lower():
            raise RuntimeError(f"device gate failed: expected {expected_device!r}, got {device_name!r}")
        api.check("cuCtxCreate_v2", api.cuCtxCreate(ctypes.byref(context), 0, device))
        print(f"DEVICE {device_name}", flush=True)

        unique_ptx = []
        for record in plan["slots"]:
            if native_swin and record["function"] == NATIVE_SWIN_FUNCTION:
                continue
            key = str(Path(record["ptx"]).resolve())
            if key not in unique_ptx:
                unique_ptx.append(key)
        for index, ptx_name in enumerate(unique_ptx, 1):
            ptx_path = Path(ptx_name)
            ptx_bytes = ptx_path.read_bytes()
            ptx_hash = sha256(ptx_bytes)
            for record in plan['slots']:
                if Path(record['ptx']).resolve() == ptx_path.resolve():
                    if ptx_hash != record['ptx_sha256'].upper():
                        raise ValueError(f"slot {record['slot']} PTX hash differs from plan")
                    if input_rgba16f and record['slot'] == 1 and b'tex.2d.' not in ptx_bytes:
                        raise ValueError("preblock has no texture samples; external input would be ignored")
            ptx_buffer = ctypes.create_string_buffer(ptx_bytes + b"\0")
            module = CUmodule()
            started = time.perf_counter()
            api.check(f"cuModuleLoadData {ptx_path.name}",
                      api.cuModuleLoadData(ctypes.byref(module), ctypes.addressof(ptx_buffer)))
            elapsed = (time.perf_counter() - started) * 1000.0
            modules[ptx_name] = module
            item = {"index": index, "count": len(unique_ptx), "path": ptx_name,
                    "milliseconds": elapsed, "loaded": True}
            report["modules"].append(item)
            print(f"MODULE {index}/{len(unique_ptx)} {ptx_path.name} {elapsed:.1f} ms", flush=True)
            write_progress(progress_path, report)

        for record in plan["slots"]:
            if native_swin and record["function"] == NATIVE_SWIN_FUNCTION:
                continue
            ptx_name = str(Path(record["ptx"]).resolve())
            key = (ptx_name, record["function"])
            if key not in functions:
                function = CUfunction()
                api.check(f"cuModuleGetFunction slot {record['slot']} {record['function']}",
                          api.cuModuleGetFunction(ctypes.byref(function), modules[ptx_name],
                                                  record["function"].encode("utf-8")))
                functions[key] = function

        api.check("cuMemAlloc activation", api.cuMemAlloc(ctypes.byref(arena), len(arena_initial)))
        api.check("cuMemAlloc model", api.cuMemAlloc(ctypes.byref(model), len(model_initial)))
        api.check("cuMemAlloc final", api.cuMemAlloc(ctypes.byref(final_output), int(plan["final_output_bytes"])))

        array_desc = CUDA_ARRAY_DESCRIPTOR(640, 360, CU_AD_FORMAT_HALF, 4)
        api.check("cuArrayCreate zero texture", api.cuArrayCreate(ctypes.byref(texture_array), ctypes.byref(array_desc)))
        zero_host = ctypes.create_string_buffer(input_bytes, len(input_bytes))
        copy = CUDA_MEMCPY2D()
        copy.srcMemoryType = CU_MEMORYTYPE_HOST
        copy.srcHost = ctypes.addressof(zero_host)
        copy.srcPitch = 640 * 8
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY
        copy.dstArray = texture_array
        copy.WidthInBytes = 640 * 8
        copy.Height = 360
        api.check("cuMemcpy2D zero texture", api.cuMemcpy2D(ctypes.byref(copy)))
        resource = CUDA_RESOURCE_DESC()
        resource.resType = CU_RESOURCE_TYPE_ARRAY
        resource.res.array.hArray = texture_array
        texture = CUDA_TEXTURE_DESC()
        texture.addressMode[:] = [CU_TR_ADDRESS_MODE_CLAMP] * 3
        texture.filterMode = CU_TR_FILTER_MODE_LINEAR
        texture.flags = CU_TRSF_NORMALIZED_COORDINATES
        api.check("cuTexObjectCreate zero texture",
                  api.cuTexObjectCreate(ctypes.byref(zero_texture), ctypes.byref(resource),
                                        ctypes.byref(texture), None))

        if post_texture_spec is not None and post_texture_bytes is not None:
            texture_width = int(post_texture_spec["width"])
            texture_height = int(post_texture_spec["height"])
            post_array_desc = CUDA_ARRAY_DESCRIPTOR(
                texture_width, texture_height, CU_AD_FORMAT_HALF, 4)
            api.check("cuArrayCreate post texture", api.cuArrayCreate(
                ctypes.byref(post_texture_array), ctypes.byref(post_array_desc)))
            post_host = ctypes.create_string_buffer(post_texture_bytes, len(post_texture_bytes))
            post_copy = CUDA_MEMCPY2D()
            post_copy.srcMemoryType = CU_MEMORYTYPE_HOST
            post_copy.srcHost = ctypes.addressof(post_host)
            post_copy.srcPitch = texture_width * 8
            post_copy.dstMemoryType = CU_MEMORYTYPE_ARRAY
            post_copy.dstArray = post_texture_array
            post_copy.WidthInBytes = texture_width * 8
            post_copy.Height = texture_height
            api.check("cuMemcpy2D post texture", api.cuMemcpy2D(ctypes.byref(post_copy)))
            post_resource = CUDA_RESOURCE_DESC()
            post_resource.resType = CU_RESOURCE_TYPE_ARRAY
            post_resource.res.array.hArray = post_texture_array
            post_desc = CUDA_TEXTURE_DESC()
            post_desc.addressMode[:] = [CU_TR_ADDRESS_MODE_BORDER] * 3
            post_desc.filterMode = CU_TR_FILTER_MODE_POINT
            post_desc.flags = CU_TRSF_NORMALIZED_COORDINATES
            api.check("cuTexObjectCreate post texture", api.cuTexObjectCreate(
                ctypes.byref(post_texture), ctypes.byref(post_resource),
                ctypes.byref(post_desc), None))

        for run_index in range(1, repetitions + 1):
            run_dir = output_dir / f"run{run_index}"
            run_dir.mkdir()
            h_to_d(api, arena.value, arena_initial, f"run {run_index} reset activation")
            h_to_d(api, model.value, model_initial, f"run {run_index} reset model")
            api.check(f"run {run_index} clear final",
                      api.cuMemsetD32(final_output, 0, int(plan["final_output_bytes"]) // 4))
            run_record = {"run": run_index, "status": "RUNNING", "slots": [],
                          "checkpoints": [], "diagnostic_injections": []}
            report["runs"].append(run_record)
            write_progress(progress_path, report)
            print(f"RUN {run_index}/{repetitions} START", flush=True)
            for record in plan["slots"]:
                slot = int(record["slot"])
                if export_pre_head and slot == 154:
                    # Output-only observation: never feeds RTX intermediates into
                    # the graph. It is the AMD-produced state after slots 0-153.
                    pre_head = d_to_h(api, arena.value, len(arena_initial),
                                      f"run {run_index} pre-head activation export")
                    pre_head_path = run_dir / "pre_head_activation.raw"
                    pre_head_path.write_bytes(pre_head)
                    run_record["pre_head_activation"] = {
                        "path": str(pre_head_path), "bytes": len(pre_head),
                        "sha256": sha256(pre_head), "producer": "AMD_slots_0_153",
                    }
                active_slot = slot
                params_path = resolve_asset(root, record["params"])
                params = bytearray(params_path.read_bytes())
                if len(params) != int(record["param_size"]):
                    raise ValueError(f"slot {slot} parameter size differs")
                for view in record["activation_param_views"]:
                    struct.pack_into("<Q", params, int(view["param_offset"]),
                                     arena.value + int(view["arena_offset"]))
                for view in record["weight_param_views"]:
                    struct.pack_into("<Q", params, int(view["param_offset"]),
                                     model.value + int(view["weight_offset"]))
                patch_special_parameters(
                    params, record["special"], zero_texture.value, final_output.value,
                    post_texture.value,
                )
                param_buffer = ctypes.create_string_buffer(bytes(params), len(params))
                kernel_params = (ctypes.c_void_p * 1)(ctypes.addressof(param_buffer))
                ptx_name = str(Path(record["ptx"]).resolve())
                grid = [int(x) for x in record["grid"]]
                block = [int(x) for x in record["block"]]
                started = time.perf_counter()
                native_details = {}
                if native_swin and record["function"] == NATIVE_SWIN_FUNCTION:
                    native_details = run_native_swin(api, arena.value, model_initial, record,
                        bytes(params), native_swin, run_dir / f"native_slot{slot}")
                else:
                    function = functions[(ptx_name, record["function"])]
                    api.check(f"run {run_index} slot {slot} cuLaunchKernel",
                              api.cuLaunchKernel(function, *grid, *block,
                                                 int(record["dynamic_shared"]), None,
                                                 kernel_params, None))
                api.check(f"run {run_index} slot {slot} cuCtxSynchronize", api.cuCtxSynchronize())
                elapsed = (time.perf_counter() - started) * 1000.0
                run_record["slots"].append({"slot": slot, "function": record["function"],
                                             "milliseconds": elapsed, "executed": True,
                                             **native_details})
                checkpoint = record.get("checkpoint")
                if checkpoint:
                    data = d_to_h(api, arena.value + int(checkpoint["arena_offset"]),
                                  int(checkpoint["logical_bytes"]),
                                  f"run {run_index} slot {slot} checkpoint")
                    checkpoint_path = run_dir / f"slot{slot}.raw"
                    checkpoint_path.write_bytes(data)
                    run_record["checkpoints"].append({"slot": slot, "path": str(checkpoint_path),
                                                       "bytes": len(data), "sha256": sha256(data)})
                if slot in injections:
                    injection, data = injections[slot]
                    h_to_d(api, arena.value + int(injection["arena_offset"]), data,
                           f"run {run_index} inject RTX state after slot {slot}")
                    run_record["diagnostic_injections"].append({
                        "after_slot": slot,
                        "arena_offset": int(injection["arena_offset"]),
                        "reference_offset": int(injection.get("reference_offset", 0)),
                        "bytes": len(data),
                        "sha256": sha256(data),
                    })
                    print(f"RUN {run_index} INJECT RTX AFTER SLOT {slot} ({len(data)} bytes)",
                          flush=True)
                print(f"RUN {run_index} SLOT {slot}/155 {elapsed:.1f} ms", flush=True)
                write_progress(progress_path, report)

            final = d_to_h(api, final_output.value, int(plan["final_output_bytes"]),
                           f"run {run_index} final readback")
            final_path = run_dir / "final_rgba16f.raw"
            final_path.write_bytes(final)
            run_record["final"] = {"path": str(final_path), "bytes": len(final), "sha256": sha256(final)}
            run_record["status"] = "PASS"
            print(f"RUN {run_index}/{repetitions} PASS {run_record['final']['sha256']}", flush=True)
            write_progress(progress_path, report)

        hashes = [item["final"]["sha256"] for item in report["runs"]]
        report["repeat_final_bitwise_exact"] = len(set(hashes)) == 1
        report["all_slots_executed"] = all(len(item["slots"]) == 156 for item in report["runs"])
        report["status"] = "PASS" if report["repeat_final_bitwise_exact"] and report["all_slots_executed"] else "FAIL"
        report["pass"] = report["status"] == "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["pass"] = False
        report["failure"] = {"active_slot": active_slot, "type": type(exc).__name__, "message": str(exc)}
        print(f"FAIL slot={active_slot}: {exc}", file=sys.stderr, flush=True)
    finally:
        if post_texture.value:
            api.cuTexObjectDestroy(post_texture)
        if post_texture_array.value:
            api.cuArrayDestroy(post_texture_array)
        if zero_texture.value:
            api.cuTexObjectDestroy(zero_texture)
        if texture_array.value:
            api.cuArrayDestroy(texture_array)
        for pointer in (final_output, model, arena):
            if pointer.value:
                api.cuMemFree(pointer)
        for module in reversed(list(modules.values())):
            if module.value:
                api.cuModuleUnload(module)
        if context.value:
            api.cuCtxDestroy(context)
        report["device"] = device_name
        write_progress(progress_path, report)
        for handle in dll_handles:
            handle.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--nvcuda", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-device", default="AMD Radeon RX 9070 XT")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--export-pre-head", action="store_true")
    parser.add_argument('--activation-init', choices=('captured', 'zero'), default='captured',
                        help='zero is a diagnostic ablation, not a verified temporal-state contract')
    parser.add_argument("--input-rgba16f", type=Path,
                        help="explicit 640x360 color input; other captured inputs remain unchanged")
    parser.add_argument("--native-swin", type=Path,
                        help="experimental native family executable; uses diagnostic CPU handoffs")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("at least two repetitions are required for determinism evidence")
    report = run(args.plan.resolve(), args.nvcuda.resolve(), args.output.resolve(),
                 args.expected_device, args.repetitions, args.export_pre_head,
                 args.native_swin.resolve() if args.native_swin else None,
                 args.input_rgba16f.resolve(strict=True) if args.input_rgba16f else None,
                 args.activation_init)
    print(json.dumps({"status": report["status"], "device": report.get("device"),
                      "runs": len(report["runs"]), "failure": report.get("failure")}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
