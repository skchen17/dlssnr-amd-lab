"""Measure a real-color replay without comparing it against unrelated RTX zero input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_verified(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest().upper() != expected.upper():
        raise ValueError(f'hash mismatch: {path}')
    return data


def rgba(data: bytes) -> np.ndarray:
    if len(data) != 640 * 360 * 8:
        raise ValueError('expected 640x360 RGBA16F')
    return np.frombuffer(data, dtype='<f2').reshape(360, 640, 4).astype('f4')


def compare(base: np.ndarray, output: np.ndarray) -> dict:
    if not np.isfinite(output).all():
        return {'finite': False, 'nonfinite_components': int(np.count_nonzero(~np.isfinite(output)))}
    # The recovered head clips to [0,1]. Report both raw change and change beyond
    # that trivial clipping, otherwise HDR clipping can masquerade as inference.
    clipped = np.clip(base[..., :3], 0, 1)
    error = output[..., :3] - clipped
    return {'finite': True,
            'changed_rgb_components_vs_raw': int(np.count_nonzero(output[..., :3] != base[..., :3])),
            'changed_rgb_components_vs_clamped_base': int(np.count_nonzero(error)),
            'mae_vs_clamped_base': float(np.abs(error).mean()),
            'max_error_vs_clamped_base': float(np.abs(error).max()),
            'rgb_min': float(output[..., :3].min()), 'rgb_max': float(output[..., :3].max()),
            'alpha_exact_vs_input': bool(np.array_equal(output[..., 3], base[..., 3]))}


def analyze(execution_dir: Path, output: Path) -> dict:
    execution = json.loads((execution_dir / 'execution.json').read_text(encoding='utf-8-sig'))
    color = execution.get('color_input', {})
    if not color.get('external') or not execution.get('pass') or execution.get('rtx_intermediate_state_injection'):
        raise ValueError('requires successful external-color execution without injections')
    composition_base = execution.get('post_texture') or color
    base = rgba(read_verified(Path(composition_base['path']), composition_base['sha256']))
    frames, run_reports = [], []
    for run in execution['runs']:
        if [s['slot'] for s in run['slots'] if s.get('executed')] != list(range(156)):
            raise ValueError('missing graph slots')
        final = run['final']
        frame = rgba(read_verified(Path(final['path']), final['sha256']))
        frames.append(frame)
        timings = run['slots']
        run_reports.append({'run': run['run'], 'output_sha256': final['sha256'],
            'metrics': compare(base, frame),
            'sum_launch_and_sync_ms': sum(s['milliseconds'] for s in timings),
            'slowest_slots': sorted(timings, key=lambda s: s['milliseconds'], reverse=True)[:10]})
    repeated = len(frames) >= 2 and all(np.array_equal(frames[0], f) for f in frames[1:])
    finite = all(r['metrics']['finite'] for r in run_reports)
    report = {'status': 'COLOR_REPLAY_EXECUTION_PASS' if finite and repeated else 'FAIL',
        'input': color, 'composition_base': composition_base,
        'model_sha256': execution['model_sha256'], 'runs': run_reports,
        'repeat_exact': repeated, 'same_input_rtx_quality_verified': False,
        'temporal_inputs_bound': False, 'game_runtime_ready': False,
        'color_space_verified': False,
        'activation_initialization': execution.get('activation_initialization', 'unknown'),
        'full_frame_arbitrary_resolution_verified': False,
        'timing_scope': 'host launch + per-slot sync; excludes module load and checkpoint IO; not resident GPU benchmark'}
    output.mkdir(parents=True, exist_ok=False)
    # One shared display transform; output differences are not tone-map differences.
    for name, frame in [('input', base), ('output', frames[0])]:
        if not np.isfinite(frame).all():
            continue
        rgb = np.maximum(frame[..., :3], 0)
        preview = np.power(rgb / (1 + rgb), 1 / 2.2)
        Image.fromarray(np.rint(preview * 255).astype('uint8')).save(output / (name + '.png'))
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.execution, args.output)
    print(json.dumps(report, indent=2))
