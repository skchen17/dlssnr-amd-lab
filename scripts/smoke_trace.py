"""Self-test for module_trace.dll and nvapi64.dll shims (no NVIDIA files needed).

Loads both shims via ctypes, exercises nvapi_QueryInterface resolution and one
trampolined call, triggers a LoadLibrary to check module_trace hooking, then
prints the resulting JSONL logs.
"""
import ctypes
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "20260830_170305")
os.environ["NVAPI_TRACE_LOG"] = os.path.join(OUT, "nvapi_trace_selftest.jsonl")
os.environ["MODULE_TRACE_LOG"] = os.path.join(OUT, "module_trace_selftest.log")
for p in (os.environ["NVAPI_TRACE_LOG"], os.environ["MODULE_TRACE_LOG"]):
    if os.path.exists(p):
        os.remove(p)

# module_trace first so it can observe later loads
mt = ctypes.WinDLL(os.path.join(REPO, "build", "module_trace.dll"))
mt.module_trace_version.restype = ctypes.c_char_p
print("module_trace_version:", mt.module_trace_version().decode())
time.sleep(0.7)  # allow init thread to patch IATs

# nvapi shim: resolve a known id, then call through the returned trampoline
nv = ctypes.WinDLL(os.path.join(REPO, "build", "nvapi64.dll"))
qi = nv.nvapi_QueryInterface
qi.restype = ctypes.c_void_p
qi.argtypes = [ctypes.c_uint32]
stub = qi(0x0150E828)  # NvAPI_Initialize
print("nvapi_QueryInterface(0x0150E828) ->", hex(stub or 0))
stub2 = qi(0x162BD2E5)  # community CuModule id (raw recorded either way)
print("nvapi_QueryInterface(0x162BD2E5) ->", hex(stub2 or 0))
if stub:
    fn = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint64)(stub)
    print("call through trampoline ->", hex(fn(0xDEADBEEF)))

# trigger module loads that module_trace should record
ctypes.WinDLL("dxgi.dll")
ctypes.WinDLL("d3d12.dll")
time.sleep(0.3)

for name, path in (("nvapi_trace_selftest.jsonl", os.environ["NVAPI_TRACE_LOG"]),
                   ("module_trace_selftest.log", os.environ["MODULE_TRACE_LOG"])):
    print(f"\n=== {name} ===")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 40:
                    print(line.rstrip())
    else:
        print("MISSING LOG")
        sys.exit(1)
print("\nSELFTEST OK")
