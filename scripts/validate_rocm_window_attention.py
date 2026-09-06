"""Bounded native attention probe; synthetic boundaries are NOT teacher data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from native_window_attention import LAYOUTS, RecoveredWindowAttention
from native_swin_torch import RecoveredSwin32, decode_e4, gather_windows
from validate_rocm_swin import wait


ORIGINAL_HASH = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'
ORIGINAL_BLOCKS = {
    'swin32': (1, 2, 3, 67, 68, 69), 'head32': (70,),
    'swin64': (5, 6, 7, 63, 64, 65),
    'swin128': (9, 10, 11, 12, 13, 57, 58, 59, 60, 61),
    'swin256': (15, 16, 17, 18, 19, 20, 21, 49, 50, 51, 52, 53, 54, 55),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest().upper()


def run(output, model_dir):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm required; no CPU fallback')
    device = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in device.gcnArchName:
        raise RuntimeError('expected gfx1201')
    output.mkdir(parents=True, exist_ok=False)
    arena = (model_dir / 'model_arena.raw').read_bytes()
    if sha(arena) != ORIGINAL_HASH:
        raise ValueError('original model hash mismatch')
    manifest = json.loads((model_dir / 'manifest.json').read_text())
    checks, original_records = [], []
    # Decode EVERY matching original record, not just one exemplar per family.
    grouped = {kind: [] for kind in LAYOUTS}
    for rec in manifest['tensors']:
        for kind, layout in LAYOUTS.items():
            if rec['name'] not in {f'block{b}.layer0.layer' for b in ORIGINAL_BLOCKS[kind]}:
                continue
            if rec['data_bytes'] != layout.record_bytes:
                raise ValueError('named original block has unexpected size')
            raw = arena[rec['arena_offset']:rec['arena_offset'] + rec['data_bytes']]
            candidate = RecoveredWindowAttention(raw, record_kind=kind)
            grouped[kind].append((rec, raw))
            original_records.append({'name': rec['name'], 'kind': kind,
                                     'sha256': sha(raw), 'finite_decoded_parameters': True,
                                     'layout_status': candidate.layout_status})
    torch.manual_seed(1201)
    torch.cuda.reset_peak_memory_stats()
    for kind, layout in LAYOUTS.items():
        if not grouped[kind]:
            raise ValueError(f'no original record for {kind}')
        record, raw = grouped[kind][0]
        model = RecoveredWindowAttention(raw, record_kind=kind).cuda()
        event = torch.cuda.Event(enable_timing=True)
        start = torch.cuda.Event(enable_timing=True)
        # CPU constructs deterministic TEST inputs only. All neural ops run on GPU.
        x = (torch.randn(12, 64, layout.channels) * .125).half().cuda()
        with torch.no_grad():
            model(x[:1]); event.record(); wait(event)
            start.record()
            actual = model(x)
            event.record(); wait(event)
            elapsed = start.elapsed_time(event)
        finite = bool(torch.isfinite(actual).all())
        if not finite:
            raise RuntimeError(f'nonfinite forward in {kind}; stop without retry')
        model.project.requires_grad_(True)
        before = model.project.detach().clone()
        grad_input = x[:1].detach().clone().requires_grad_(True)
        model(grad_input).float().square().mean().backward()
        event.record(); wait(event)
        gradient = all(v is not None and bool(torch.isfinite(v).all()) and bool(torch.count_nonzero(v))
                       for v in [grad_input.grad, model.project.grad])
        unchanged = bool(torch.equal(before, model.project))
        checks.append({'kind': kind, 'original_record': record['name'], 'weight_sha256': sha(raw),
                       'input': 'SYNTHETIC_POST_FFN_BOUNDARY_NOT_TEACHER', 'windows': [1, 12],
                       'finite_forward': finite, 'finite_nonzero_backward': gradient,
                       'weights_unchanged': unchanged, 'warm_12_window_gpu_ms': elapsed,
                       'layout_status': model.layout_status, 'rtx_comparison': 'NOT_RUN'})
        del model, x, actual, grad_input
    # Actual saved C32 inputs, native before/after boundary refactor comparison.
    controls = []
    cases = Path('results/20260905_052500_swin1h_native_family')
    case_manifest = json.loads((cases / 'manifest.json').read_text())
    for spec in case_manifest['slots']:
        slot = spec['slot']
        folder = cases / f'slot{slot}'
        raw = (folder / 'input.e4').read_bytes()
        weights = (folder / 'weights.raw').read_bytes()
        if sha(raw) != spec['input_sha256'].upper() or sha(weights) != spec['weights_sha256'].upper():
            raise ValueError('native control provenance mismatch')
        old = RecoveredSwin32(weights).cuda()
        new = RecoveredWindowAttention(weights, record_kind='swin32').cuda()
        windows, _ = gather_windows(decode_e4(raw).cuda(), *spec['geometry'])
        selected = windows[windows.shape[0] // 2:windows.shape[0] // 2 + 12]
        with torch.no_grad():
            expected = old(selected)
            actual = new(old.forward_ffn(selected))
            event.record(); wait(event)
        equal = bool(torch.equal(expected, actual))
        controls.append({'slot': slot, 'windows': len(selected), 'native_control_exact': equal,
                         'max_abs': float((expected.float() - actual.float()).abs().max()),
                         'input_sha256': sha(raw), 'weights_sha256': sha(weights)})
        del old, new, windows, selected, expected, actual
    passed = all(c['finite_forward'] and c['finite_nonzero_backward'] and c['weights_unchanged'] for c in checks)
    passed = passed and len(controls) == 4 and all(c['native_control_exact'] for c in controls)
    report = {'status': 'BOUNDED_ATTENTION_PROBE_PASS' if passed else 'PROBE_FAIL',
              'backend': 'pytorch_rocm', 'device': device.name,
              'torch': torch.__version__, 'hip': torch.version.hip,
              'original_model_sha256': sha(arena), 'decoded_original_records': original_records,
              'checks': checks, 'native32_controls': controls,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'native_graph_complete': False, 'wide_swin_ffn_reconstructed': False,
              'rtx_quality_verified': False, 'training_started': False,
              'optimizer_steps': 0, 'game_files_changed': False}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'decoded_original_records'}, indent=2))
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model-dir', type=Path, default=Path('local_models/decoded_310_8'))
    args = p.parse_args()
    raise SystemExit(0 if run(args.output, args.model_dir) else 1)
