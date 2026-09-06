"""Validate bounded GPU texture captures against uninstrumented real FFX runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_ffx_dispatch_probe import analyze_provider


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def analyze_pair(baseline: Path, observed: Path) -> dict:
    analyze_provider(baseline)
    analyze_provider(observed)
    a, b = read_json(baseline / 'manifest.json'), read_json(observed / 'manifest.json')
    for key in ('requested_provider', 'requested_provider_id', 'adapter_vendor', 'adapter_device'):
        if key not in a or a[key] != b.get(key):
            raise ValueError('provider/adapter identity mismatch')
    expected = dict(capture_enabled=True, capture_completed=3, capture_pending=0,
                    capture_dropped=1, capture_invalid_rejected=8,
                    capture_early_poll_checks=5, capture_byte_budget_rejected=True)
    if any(b.get(key) != value for key, value in expected.items()):
        raise ValueError('native capture lifecycle/budget gate failed')
    unchanged = []
    for i in range(4):
        raw = (baseline / f'output_{i}.raw').read_bytes()
        if raw != (observed / f'output_{i}.raw').read_bytes():
            raise ValueError('capture changed original output')
        if (baseline / f'input_{i}.raw').read_bytes() != (observed / f'input_{i}.raw').read_bytes():
            raise ValueError('baseline/capture inputs differ')
        unchanged.append(hashlib.sha256(raw).hexdigest())
    captures = observed / 'capture'
    if {p.name for p in captures.iterdir()} != {'0', '1', '2'}:
        raise ValueError('unexpected or missing captured frame')
    if list((observed / 'capture_byte_budget').iterdir()):
        raise ValueError('byte-budget rejection emitted data')
    specs = {'color': (320, 180, 8, 10), 'depth': (320, 180, 4, 41),
             'motion': (320, 180, 4, 34), 'exposure': (1, 1, 4, 41),
             'output': (640, 360, 8, 10)}
    count, raw_bytes, padded = 0, 0, 0
    for i in range(3):
        directory = captures / str(i)
        m = read_json(directory / 'manifest.json')
        if m.get('frame') != i or m.get('fence_completed') is not True or m.get('fence_value') != i + 1:
            raise ValueError('capture completion metadata invalid')
        if m.get('render_size') != [320, 180] or m.get('upscale_size') != [640, 360] or m.get('reset') is not True:
            raise ValueError('frame metadata mismatch')
        if m.get('context_create_flags') != 128 or m.get('requested_provider_id') != b['requested_provider_id']:
            raise ValueError('context metadata mismatch')
        resources = m.get('resources', [])
        if len(resources) != 5 or {r['role'] for r in resources} != set(specs):
            raise ValueError('missing/duplicate captured resource')
        for resource in resources:
            role = resource['role']
            w, h, pixel_bytes, fmt = specs[role]
            row = w * pixel_bytes
            if (resource.get('width'), resource.get('height'), resource.get('dxgi_format'), resource.get('row_bytes'), resource.get('raw_bytes')) != (w, h, fmt, row, row * h):
                raise ValueError('capture footprint metadata mismatch')
            pitch = resource.get('row_pitch', 0)
            if pitch < row or pitch % 256:
                raise ValueError('invalid copy row pitch')
            padded += pitch > row
            if resource.get('ffx_state_restored') != (2 if role == 'output' else 4):
                raise ValueError('resource state not restored')
            raw = (directory / f'{role}.raw').read_bytes()
            if len(raw) != row * h:
                raise ValueError('capture raw size mismatch')
            if role == 'color':
                expected_raw = (observed / f'input_{i}.raw').read_bytes()
            elif role == 'output':
                expected_raw = (observed / f'output_{i}.raw').read_bytes()
            elif role == 'depth':
                expected_raw = np.full((180, 320), .5, dtype='<f4').tobytes()
            elif role == 'motion':
                expected_raw = bytes(320 * 180 * 4)
            else:
                expected_raw = np.array([1], dtype='<f4').tobytes()
            if raw != expected_raw:
                raise ValueError(f'captured {role} bytes differ')
            count += 1
            raw_bytes += len(raw)
    return dict(status='CAPTURE_PASS', original_output_unchanged=True,
                unchanged_output_sha256=unchanged, captured_frames=3,
                captured_resources=count, captured_raw_bytes=raw_bytes,
                padded_row_resources_verified=padded, pending_resources=0,
                game_frame_capture_ready=False, game_launched=False, dlss_nr_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = {}
    for name in ('fsr3', 'fsr4'):
        try:
            report[name] = analyze_pair(args.directory / name, args.directory / (name + '_capture'))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report[name] = dict(status='CAPTURE_FAIL', error=str(exc))
    report['status'] = 'PASS' if all(v['status'] == 'CAPTURE_PASS' for v in report.values()) else 'FAIL'
    (args.directory / 'capture_analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
