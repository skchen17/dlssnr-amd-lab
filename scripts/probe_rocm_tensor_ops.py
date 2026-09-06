"""Small bounded gfx1201 operator probe. Not a network or performance claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def probe(output: Path):
    import torch
    import torch.nn.functional as F
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU PyTorch required; CPU/CUDA fallback forbidden')
    prop = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in prop.gcnArchName:
        raise RuntimeError(f'wrong GPU architecture: {prop.gcnArchName}')
    torch.manual_seed(20260905)
    torch.cuda.reset_peak_memory_stats()
    records = []

    def check(name, gpu, cpu, tolerance):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        actual = gpu()
        end.record()
        deadline = time.monotonic() + 10
        while not end.query():
            if time.monotonic() > deadline:
                raise TimeoutError('GPU event timeout; do not retry automatically')
            time.sleep(.005)
        if actual.device.type != 'cuda':
            raise RuntimeError('GPU operation returned a host tensor')
        ref, got = cpu().float(), actual.detach().cpu().float()
        error = float((ref - got).abs().max())
        ok = bool(torch.isfinite(got).all()) and error <= tolerance
        records.append({'operation': name, 'pass': ok, 'max_absolute_error': error,
                        'tolerance': tolerance, 'gpu_ms_first_call': start.elapsed_time(end)})
        if not ok:
            raise RuntimeError(f'{name} failed: {records[-1]}')
    a = torch.randn(64, 32).half()
    b = torch.randn(32, 96).half()
    da, db = a.cuda(), b.cuda()
    check('fp16_matmul', lambda: da @ db, lambda: a.float() @ b.float(), .03)
    check('layer_norm', lambda: F.layer_norm(da, (32,)),
          lambda: F.layer_norm(a.float(), (32,)), .004)
    q = torch.randn(2, 1, 64, 32).half()
    k, v = torch.randn_like(q), torch.randn_like(q)
    dq, dk, dv = q.cuda(), k.cuda(), v.cuda()
    check('attention_sdpa', lambda: F.scaled_dot_product_attention(dq, dk, dv),
          lambda: F.scaled_dot_product_attention(q.float(), k.float(), v.float()), .006)
    values = torch.tensor([-448., -1.125, -.001953125, 0., .001953125, 1.125, 448.])
    dvalues = values.cuda()
    check('fp8_e4m3fn_roundtrip', lambda: dvalues.to(torch.float8_e4m3fn).half(),
          lambda: values.to(torch.float8_e4m3fn).float(), 0.)
    x, w = da.clone().requires_grad_(), db.clone().requires_grad_()
    cx, cw = a.float().requires_grad_(), b.float().requires_grad_()
    def backward():
        F.layer_norm(x @ w, (96,)).float().square().mean().backward()
        if w.grad is None or not torch.isfinite(w.grad).all():
            raise RuntimeError('weight gradient missing/nonfinite')
        return x.grad
    def reference_backward():
        F.layer_norm(cx @ cw, (96,)).square().mean().backward()
        return cx.grad
    check('matmul_norm_backward', backward, reference_backward, .002)
    report = {'schema': 1, 'status': 'OPERATOR_CAPABILITY_PASS',
              'torch': torch.__version__, 'hip': torch.version.hip,
              'python': sys.version, 'python_executable': sys.executable,
              'device': prop.name, 'architecture': prop.gcnArchName,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'operations': records, 'native_graph_complete': False,
              'cpu_used_only_for_reference': True}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        probe(args.output)
    else:
        # Process deadline bounds host waiting, NOT cancellation of a hung GPU.
        env = dict(os.environ, PYTHONNOUSERSITE='1')
        try:
            result = subprocess.run([sys.executable, __file__, '--child', '--output', str(args.output)],
                                    env=env, timeout=60)
            raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            print('GPU probe timeout: stop all GPU work; no automatic retry', file=sys.stderr)
            raise SystemExit(2)
