"""Strict isolated gate for bounded-LDS C64/C128 attention.

The safety guard runs before a child process or HIP runtime is started.  This
script intentionally validates one channel family per process so a failed gate
cannot automatically continue into the next family.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save, supervise


RECORD_BYTES = {64: 61760, 128: 197184}


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def compare(reference, candidate):
    import torch

    delta = candidate.float() - reference.float()
    different = int(torch.count_nonzero(candidate.view(torch.int16) != reference.view(torch.int16)).item())
    rmse = float(torch.sqrt(torch.mean(delta.square())).item())
    denominator = float(torch.sqrt(torch.mean(reference.float().square())).item())
    return {
        "bitwise_exact": different == 0,
        "different_components": different,
        "max_absolute_error": float(delta.abs().max().item()),
        "rmse": rmse,
        "nrmse": rmse / denominator if denominator else (0.0 if rmse == 0 else None),
    }


def exercise(args):
    import torch
    from native_execution_policy import execution_policy
    from native_fusion_policy import fusion_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin

    torch.set_num_threads(2)
    torch.manual_seed(181 + args.channels)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm required")
    props = torch.cuda.get_device_properties(0)
    if "gfx1201" not in props.gcnArchName:
        raise RuntimeError("gfx1201 required")

    records, _, _ = load_package(
        "local_models/native_single_color_v1",
        "AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3",
    )
    matches = [(key, raw) for key, raw in records.items() if len(raw) == RECORD_BYTES[args.channels]]
    if not matches:
        raise RuntimeError(f"no C{args.channels} record in frozen model package")
    record_key, raw_record = matches[0]
    block = RecoveredPackedSwin(raw_record, record_kind=f"swin{args.channels}").eval().cuda().requires_grad_(False)
    elements = args.windows * 64 * args.channels
    ramp = torch.linspace(-0.25, 0.25, 1 << 16, device="cuda", dtype=torch.float16)
    raw = ramp.repeat((elements + len(ramp) - 1) // len(ramp))[:elements].reshape(
        args.windows, 64 * args.channels
    ).contiguous()
    stream = torch.cuda.current_stream()

    def run(attention_family):
        modules = (f"c{args.channels}_ffn", attention_family)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), execution_policy("native_fp16"), fusion_policy(args.quant_dll), matrix_fusion(
            args.matrix_dll, "wmma_fp16", modules=modules, waves=args.waves
        ) as matrix:
            for _ in range(2):
                output = block(raw)
            torch.cuda.synchronize()
            times = []
            for _ in range(args.iterations):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                output = block(raw)
                end.record(stream)
                end.synchronize()
                times.append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(f"nonfinite C{args.channels} output")
            result = output.clone()
            calls = args.iterations + 2
            row = {
                "modules": list(modules),
                "gpu_event_ms": times,
                "median_gpu_event_ms": statistics.median(times),
                "matrix_launches_per_call": matrix.launches / calls,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "sha256": digest(result),
            }
        return result, row

    baseline, baseline_row = run(f"c{args.channels}_attention")
    candidate, candidate_row = run(f"c{args.channels}_attention_bounded")
    equality = compare(baseline, candidate)
    return {
        "checks_pass": equality["bitwise_exact"],
        "scope": f"same C{args.channels} FFN; six global attention stages versus bounded per-head LDS plus projection",
        "channels": args.channels,
        "record_key": list(record_key),
        "windows": args.windows,
        "waves": args.waves,
        "baseline": baseline_row,
        "candidate": candidate_row,
        "comparison": equality,
        "speedup": baseline_row["median_gpu_event_ms"] / candidate_row["median_gpu_event_ms"],
        "device": props.name,
        "architecture": props.gcnArchName,
        "weights_modified": False,
        "default_promoted": False,
    }


def child(args):
    def phase(name):
        with (args.output / "phases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"phase": name, "pid": os.getpid(), "monotonic": time.monotonic()}) + "\n")

    atexit.register(phase, "python_atexit")
    phase("child_enter")
    import ctypes as ct

    dll = ct.CDLL(str(args.matrix_dll.resolve()))
    required = (f"nr_c{args.channels}_ffn_wmma", f"nr_c{args.channels}_attention_head_fused")
    for symbol in required:
        if not hasattr(dll, symbol):
            raise ValueError(f"missing required symbol {symbol}")
    phase("abi_verified")
    result = exercise(args)
    phase("gpu_work_complete")
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result["allocated_after_release_bytes"] = torch.cuda.memory_allocated()
    result["reserved_after_release_bytes"] = torch.cuda.memory_reserved()
    result["checks_pass"] &= result["allocated_after_release_bytes"] == result["reserved_after_release_bytes"] == 0
    phase("resources_released")
    save(args.output / "child.json", result)
    return result["checks_pass"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dll", type=Path, required=True)
    parser.add_argument("--matrix-dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, choices=(64, 128), required=True)
    parser.add_argument("--windows", type=int, choices=(1, 64, 256), required=True)
    parser.add_argument("--waves", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--iterations", type=int, choices=(1, 12), default=12)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    require_gpu_tests_enabled(f"C{args.channels} bounded-attention GPU gate")
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--quant-dll",
        str(args.quant_dll.resolve()),
        "--matrix-dll",
        str(args.matrix_dll.resolve()),
        "--output",
        str(args.output.resolve()),
        "--channels",
        str(args.channels),
        "--windows",
        str(args.windows),
        "--waves",
        str(args.waves),
        "--iterations",
        str(args.iterations),
    ]
    return supervise(command, args.output, timeout=120)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 2)
