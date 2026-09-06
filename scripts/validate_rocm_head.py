"""Isolated original-weight head comparison; not full-network/HDR acceptance."""
import argparse
import json
from pathlib import Path
import torch
from native_swin_torch import decode_e4
from native_head_torch import RecoveredHead32, compose_legacy_sdr_debug
from validate_rocm_swin import wait
from freeze_native_baseline import digest


def run(output):
    if not torch.version.hip or not torch.cuda.is_available() or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('requires gfx1201 ROCm; no CPU fallback')
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path('local_models/decoded_310_8/model_arena.raw')
    model_sha = digest(model_path)
    if model_sha != 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5':
        raise ValueError('original model hash mismatch')
    arena_path = Path('deliverables/postblock_mma_trace_reference_20260904_154238/payload/activation_arena.raw')
    arena = arena_path.read_bytes()
    model = RecoveredHead32(model_path.read_bytes()[147429888:147429888 + 21808]).cuda()
    main = decode_e4(arena[13873152:13873152 + 1966080]).cuda()
    skip = decode_e4(arena[110592:110592 + 7864320]).cuda()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ctas = torch.arange(81 * 49, device='cuda')
    with torch.no_grad():
        model(main, skip, ctas[:1], 640, 384)
        end.record(); wait(end)
        start.record()
        values, fused = [], []
        for chunk in ctas.split(64):
            values.append(model(main, skip, chunk, 640, 384))
            fused.append(model.fuse(main, skip, chunk, 640, 384))
            end.record(); wait(end)
        residual = torch.cat(values)
        image = compose_legacy_sdr_debug(residual, torch.zeros(360, 640, 4, device='cuda', dtype=torch.float16), 640, 384)
        end.record(); wait(end)
        elapsed = start.elapsed_time(end)
    controls = Path('results/20260905_013000_output_head_pipeline_rx9070xt')
    metrics = {}
    for name, value, ref_name in [('fused', torch.cat(fused), 'activation_fp16.raw'),
                                 ('residual', residual, 'rgba_residual_fp16.raw'),
                                 ('surface', image, 'zero_base_output_rgba16f.raw')]:
        actual = value.cpu().flatten()
        reference_path = controls / ref_name
        expected = torch.frombuffer(bytearray(reference_path.read_bytes()), dtype=torch.float16)
        delta = (actual.float() - expected.float()).abs()
        finite = bool(torch.isfinite(actual).all())
        metrics[name] = {'finite': finite, 'mean_absolute_error': delta.mean().item(),
                         'max_absolute_error': delta.max().item(), 'reference_sha256': digest(reference_path),
                         'exact_elements': int((actual == expected).sum()), 'elements': actual.numel()}
        (output / f'{name}.raw').write_bytes(actual.numpy().tobytes())
    passed = all(x['finite'] for x in metrics.values()) and metrics['surface']['mean_absolute_error'] <= .001 and metrics['surface']['max_absolute_error'] <= .01
    report = {'status': 'ISOLATED_HEAD_PORT_PASS' if passed else 'HEAD_PORT_FAIL',
              'backend': 'pytorch_rocm', 'model_sha256': model_sha,
              'activation_sha256': digest(arena_path), 'metrics': metrics,
              'gpu_event_span_ms_with_host_gaps': elapsed,
              'reference_kind': 'previous_native_head_control_not_new_RTX_quality',
              'native_graph_complete': False, 'hdr_supported': False, 'game_runtime_ready': False}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
