"""Long-enough HIP workload used only to validate Radeon GPU Profiler capture.

This is deliberately not a neural-network benchmark.  It gives the developer
service time to attach, then submits one already correctness-gated project
kernel repeatedly.  A failed capture must not be interpreted as a GPU or model
failure.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attach-seconds", type=float, default=5.0)
    parser.add_argument("--post-seconds", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=256)
    args = parser.parse_args()
    if (not 0.0 <= args.attach_seconds <= 30.0
            or not 0.0 <= args.post_seconds <= 30.0
            or not 1 <= args.iterations <= 100000):
        raise ValueError("bounded attach delay and iteration count required")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch

    if not torch.version.hip:
        raise RuntimeError("ROCm PyTorch is required")
    props = torch.cuda.get_device_properties(0)
    if "gfx1201" not in props.gcnArchName:
        raise RuntimeError("this smoke target is restricted to gfx1201")

    library = ct.CDLL(str(args.dll.resolve()))
    launch = library.native_fusion_cubic_quantize_f16
    launch.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.c_void_p]
    launch.restype = ct.c_int
    # Windows ROCm 7.2 currently produces NaNs when a directly-created FP16
    # linspace has more than 65535 elements.  Build a finite repeated ramp so a
    # profiler transport failure cannot be confused with that unrelated issue.
    ramp = torch.linspace(-8.0, 8.0, 1 << 16, device="cuda", dtype=torch.float16)
    source = ramp.repeat(16)
    output = torch.empty_like(source)
    stream = torch.cuda.current_stream().cuda_stream
    torch.cuda.synchronize()
    print("RGP_SMOKE_READY", flush=True)
    time.sleep(args.attach_seconds)

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(args.iterations):
        code = launch(source.data_ptr(), output.data_ptr(), source.numel(), stream)
        if code:
            raise RuntimeError(f"HIP launch failed: {code}")
    end.record()
    end.synchronize()
    finite = bool(torch.isfinite(output).all())
    report = {
        "checks_pass": finite,
        "device": props.name,
        "architecture": props.gcnArchName,
        "iterations": args.iterations,
        "gpu_event_total_ms": begin.elapsed_time(end),
        "capture_target_only": True,
    }
    (args.output / "target.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    # RDP copies SQTT/counter data while the client is still alive.  Exiting as
    # soon as the last dispatch completes truncates an otherwise valid trace.
    time.sleep(args.post_seconds)
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
