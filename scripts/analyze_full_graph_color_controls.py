"""Check color-path causality and internal backend agreement, never RTX quality."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_full_graph_color import read_verified, rgba
except ModuleNotFoundError:
    from analyze_full_graph_color import read_verified, rgba


def difference(a: bytes, b: bytes, pixels: bool = False) -> dict:
    if len(a) != len(b):
        raise ValueError('comparison size mismatch')
    result = {'exact': a == b, 'bytes': len(a),
              'mismatched_bytes': int(np.count_nonzero(np.frombuffer(a, 'u1') != np.frombuffer(b, 'u1')))}
    if pixels:
        x, y = rgba(a), rgba(b)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite comparison')
        error = np.abs(x[..., :3] - y[..., :3])
        result.update(rgb_mae=float(error.mean()), rgb_max=float(error.max()),
                      changed_rgb_components=int(np.count_nonzero(error)),
                      alpha_exact=bool(np.array_equal(x[..., 3], y[..., 3])))
    return result


def load(folder: Path) -> tuple[dict, dict, bytes, bytes]:
    report = json.loads((folder / 'execution.json').read_text(encoding='utf-8-sig'))
    if not report.get('pass') or report.get('rtx_intermediate_state_injection') or len(report['runs']) < 2:
        raise ValueError('requires complete repeated inference without injections')
    finals, heads = [], []
    for run in report['runs']:
        if [s['slot'] for s in run['slots'] if s.get('executed')] != list(range(156)):
            raise ValueError('incomplete graph')
        for name, target in [('final', finals), ('pre_head_activation', heads)]:
            spec = run[name]
            target.append(read_verified(Path(spec['path']), spec['sha256']))
    if any(f != finals[0] for f in finals[1:]) or any(h != heads[0] for h in heads[1:]):
        raise ValueError('nondeterministic graph')
    plan = json.loads(Path(report['plan']).read_text(encoding='utf-8-sig'))
    return report, plan, finals[0], heads[0]


def analyze(captured: Path, zero_arena: Path, zero_color: Path, native: Path, output: Path) -> dict:
    c, cp, cf, ch = load(captured)
    z, zp, zf, zh = load(zero_arena)
    n, np_, nf, nh = load(zero_color)
    if c['activation_initialization'] != 'captured_frame_start' or any(
            r['activation_initialization'] != 'zero' for r in (z, n)):
        raise ValueError('incorrect initialization for controls')
    if cp != zp or zp['slots'] != np_['slots']:
        raise ValueError('controls changed graph implementation')
    for key in ('model_sha256', 'post_texture'):
        if any(r[key] != c[key] for r in (z, n)):
            raise ValueError(f'control differs in {key}')
    if c['color_input'] != z['color_input']:
        raise ValueError('arena control changed neural input')
    zero_spec = n['color_input']
    if read_verified(Path(zero_spec['path']), zero_spec['sha256']) != bytes(640 * 360 * 8):
        raise ValueError('zero color control is not zero')
    arena = {'final': difference(cf, zf, True), 'pre_head_arena': difference(ch, zh)}
    neural = {'final': difference(zf, nf, True), 'pre_head_arena': difference(zh, nh)}
    native_raw = native.read_bytes()
    native_compare = difference(cf, native_raw, True)
    # Aggregate actual measured launch+sync times by exact function, no FPS extrapolation.
    families = {}
    for slot in z['runs'][1]['slots']:
        item = families.setdefault(slot['function'], {'count': 0, 'total_ms': 0.0})
        item['count'] += 1
        item['total_ms'] += slot['milliseconds']
    families = sorted([{'function': k, **v} for k, v in families.items()],
                      key=lambda f: f['total_ms'], reverse=True)
    passed = arena['final']['exact'] and not neural['pre_head_arena']['exact'] and not neural['final']['exact']
    report = {'status': 'COLOR_RESPONSE_CONTROLS_PASS' if passed else 'REVIEW_REQUIRED',
        'executions': {'captured': str(captured.resolve()), 'zero_arena': str(zero_arena.resolve()),
                       'zero_neural_color': str(zero_color.resolve())},
        'captured_vs_zero_initialization': arena,
        'real_vs_zero_neural_color_same_composition_base': neural,
        'native_head_vs_translated_head': native_compare,
        'native_output': {'path': str(native.resolve()), 'sha256': hashlib.sha256(native_raw).hexdigest().upper()},
        'family_launch_sync_timings_run2': families,
        'quality_reference': 'none; backend agreement is NOT same-input RTX agreement',
        'same_input_rtx_quality_verified': False, 'temporal_inputs_verified': False,
        'full_frame_resolution_verified': False, 'game_runtime_ready': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('captured', 'zero-arena', 'zero-color', 'native', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.captured, args.zero_arena, args.zero_color, args.native, args.output), indent=2))
