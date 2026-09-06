"""Gradient existence/finite-value probe, without optimizer updates or training."""
import argparse
import json
from pathlib import Path
import torch
from native_swin_torch import RecoveredSwin32, decode_e4, gather_windows
from native_head_torch import RecoveredHead32
from validate_rocm_swin import wait
from freeze_native_baseline import digest


def run(output):
    if not torch.version.hip or not torch.cuda.is_available() or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('gfx1201 ROCm required')
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    cases = Path('results/20260905_052500_swin1h_native_family')
    source = cases / 'slot3'
    model = RecoveredSwin32((source / 'weights.raw').read_bytes()).cuda()
    raw = decode_e4((source / 'input.e4').read_bytes()).cuda()
    windows, _ = gather_windows(raw, 320, 192, -4, -4)
    model.attention_scale.requires_grad_(True)
    before = model.attention_scale.detach().clone()
    x = windows[50:51].clone().requires_grad_(True)
    result = model(x)
    result.float().square().mean().backward()
    event = torch.cuda.Event(); event.record(); wait(event)
    for name, value in [('swin_input', x.grad), ('swin_attention_scale', model.attention_scale.grad)]:
        records.append({'name': name, 'finite': value is not None and bool(torch.isfinite(value).all()),
                        'nonzero': 0 if value is None else int(torch.count_nonzero(value)),
                        'on_gpu': value is not None and value.device.type == 'cuda'})
    unchanged = torch.equal(before, model.attention_scale)
    arena_path = Path('local_models/decoded_310_8/model_arena.raw')
    original_hash = digest(arena_path)
    if original_hash != 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5':
        raise ValueError('original model hash mismatch')
    head = RecoveredHead32(arena_path.read_bytes()[147429888:147429888 + 21808]).cuda()
    head.tail.requires_grad_(True)
    projected = torch.randn(1, 64, 32, dtype=torch.float16, device='cuda', requires_grad=True)
    loss = (projected.float() @ head.tail.float()).square().mean()
    loss.backward(); event.record(); wait(event)
    for name, value in [('head_projection_input', projected.grad), ('head_tail', head.tail.grad)]:
        records.append({'name': name, 'finite': value is not None and bool(torch.isfinite(value).all()),
                        'nonzero': 0 if value is None else int(torch.count_nonzero(value)),
                        'on_gpu': value is not None and value.device.type == 'cuda'})
    passed = unchanged and all(r['finite'] and r['nonzero'] and r['on_gpu'] for r in records)
    report = {'status': 'GRADIENT_PROBE_PASS' if passed else 'GRADIENT_PROBE_FAIL',
              'checks': records, 'weights_unchanged': unchanged, 'training_started': False,
              'optimizer_steps': 0, 'training_gpu_hours': 0, 'original_model_sha256': original_hash,
              'full_network_backward_verified': False}
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
