# ARCHITECTURE.md

## System overview

```
                +-----------------------------+
                | nr_host.exe (D3D12 x64)     |  synthetic DLAA contract:
                |  NGX init / create / eval   |  Color + Depth + MV + Exposure
                +--------------+--------------+
                               |
            +------------------+------------------+
            |                                     |
   NVIDIA path (reference)                AMD path (experiment)
            |                                     |
    nvngx.dll -> dlssnr.dll                module_trace + nvapi_trace shim
            |                                     |
   NVAPI CUBIN / CUDA Driver API          nvapi_amd / ZLUDA compatibility layer
            |                                     |
   NVIDIA driver dispatch                  HIP -> RDNA4 dispatch (gfx1200/1201)
```

## Components

### tools/hip_probe
Standalone HIP test suite. No DLSS involvement. Gate S1.
Tests: device count, alloc, FP32, FP16 matmul, FP8 E4M3/E5M2, WMMA (guarded by
`__gfx12__` / WMMA intrinsic availability at compile time). Emits JSON.

### tools/d3d12_hip_interop
Two separate experiments:
- **PASS-A**: independent HIP queue + shared fence/semaphore synchronization around a
  D3D12-created shareable buffer (NT handle export -> `hipImportExternalMemory`).
- **PASS-B**: investigate enqueueing HIP work inside a D3D12 command list context
  (ZLUDA FAQ claim — treated as unverified).

### tools/binary_probe
Read-only PE + CUDA-payload inspector. Enumerates imports, sections, fatbin containers,
PTX/cubin entries, SM targets, kernel names, weights regions. Outputs
`binary_manifest.json` (metadata only: size/hash/magic — never binary content).

### tools/module_trace
Injected or launcher-based tracing of LoadLibraryW/GetProcAddress to capture module load
order (nvapi64, nvcuda, nvngx*, etc.).

### tools/nvapi_trace
`nvapi64.dll` shim exporting `nvapi_QueryInterface`; decodes ordinals to function names,
logs every call with args/handles; binary payload pointers reduced to size+sha256+magic.

### tools/nr_host
Minimal standalone D3D12 host replicating the DLSS5-Feeder approach without ReShade:
own device + queue + (hidden) swap chain, synthetic inputs, `NGX_D3D12_INITIALIZE` ->
feature create -> evaluate -> readback. Flags: `--trace --frames N --input --output
--width --height`. Reference package mode for NVIDIA machines.

## Key design rules

1. Proprietary binaries referenced via env vars only; never copied into the repo.
2. Every GPU-success claim carries adapter proof (vendor 0x1002, device id, LUID).
3. Numeric validation uses defined tolerances (FP8/FP16 legal deviation), never bit-exact
   requirement, never hardcoded outputs.
4. Fault tolerance for closed-source addons: SEH wrapping, discard faulted command lists,
   re-init after repeated failures (pattern from DLSS5-Feeder).
