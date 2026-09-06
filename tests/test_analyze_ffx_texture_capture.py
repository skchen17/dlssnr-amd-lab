import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('analyze_ffx_dispatch_probe', scripts / 'analyze_ffx_dispatch_probe.py')
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)
sys.modules.setdefault('analyze_ffx_dispatch_probe', dispatch)
spec = importlib.util.spec_from_file_location('ffx_texture_analysis', scripts / 'analyze_ffx_texture_capture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def pair(tmp_path):
    base, capture = tmp_path / 'base', tmp_path / 'observed'
    base.mkdir()
    capture.mkdir()
    (capture / 'capture').mkdir()
    (capture / 'capture_byte_budget').mkdir()
    m = dict(status='DISPATCH_READBACK_PASS', debug_errors=0, completed_fences=4, output_poisoned_nan=True,
             requested_provider='3.1.0', requested_provider_id=123, adapter_vendor=4098, adapter_device=30032)
    write_json(base / 'manifest.json', m)
    m.update(capture_enabled=True, capture_completed=3, capture_pending=0, capture_dropped=1,
             capture_invalid_rejected=8, capture_early_poll_checks=5, capture_byte_budget_rejected=True)
    write_json(capture / 'manifest.json', m)
    specs = {'color': (320, 180, 8, 10), 'depth': (320, 180, 4, 41),
             'motion': (320, 180, 4, 34), 'exposure': (1, 1, 4, 41), 'output': (640, 360, 8, 10)}
    for i in range(4):
        output = np.empty((360, 640, 4), dtype='<f2')
        output[..., 0] = np.linspace(.1, .8, 640)
        output[..., 1] = .5
        output[..., 2] = .75 if i == 2 else .25
        output[..., 3] = 1
        source = output[::2, ::2].tobytes()
        for root in (base, capture):
            (root / f'input_{i}.raw').write_bytes(source)
            (root / f'output_{i}.raw').write_bytes(output.tobytes())
        if i == 3:
            continue
        frame = capture / 'capture' / str(i)
        frame.mkdir()
        resources = []
        blobs = dict(color=source, output=output.tobytes(), depth=np.full((180, 320), .5, dtype='<f4').tobytes(),
                     motion=bytes(320 * 180 * 4), exposure=np.array([1], dtype='<f4').tobytes())
        for role, (w, h, pixel_bytes, fmt) in specs.items():
            row = w * pixel_bytes
            resources.append(dict(role=role, width=w, height=h, dxgi_format=fmt, row_bytes=row,
                                  raw_bytes=row*h, row_pitch=(row+255)//256*256,
                                  ffx_state_restored=2 if role == 'output' else 4))
            (frame / f'{role}.raw').write_bytes(blobs[role])
        write_json(frame / 'manifest.json', dict(frame=i, fence_completed=True, fence_value=i+1,
                   render_size=[320, 180], upscale_size=[640, 360], reset=True, resources=resources,
                   context_create_flags=128, requested_provider_id=123))
    return base, capture


def test_valid_capture_pair(pair):
    report = module.analyze_pair(*pair)
    assert report['status'] == 'CAPTURE_PASS'
    assert report['captured_resources'] == 15
    assert report['captured_raw_bytes'] == 8294412
    assert report['padded_row_resources_verified'] == 3
    assert report['game_frame_capture_ready'] is False


@pytest.mark.parametrize('role', ['color', 'depth', 'motion', 'exposure', 'output'])
def test_corrupted_capture_rejected(pair, role):
    path = pair[1] / 'capture' / '0' / (role + '.raw')
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match='bytes differ'):
        module.analyze_pair(*pair)


@pytest.mark.parametrize('kind', ['fence', 'state', 'pitch', 'missing_role', 'missing_frame', 'extra_frame', 'byte_budget', 'pending', 'provider'])
def test_false_success_rejected(pair, kind):
    root = pair[1]
    path = root / 'capture' / '0' / 'manifest.json'
    m = json.loads(path.read_text())
    if kind == 'fence':
        m['fence_completed'] = False
    elif kind == 'state':
        m['resources'][0]['ffx_state_restored'] = 1
    elif kind == 'pitch':
        m['resources'][0]['row_pitch'] = 1
    elif kind == 'missing_role':
        m['resources'].pop()
    elif kind == 'missing_frame':
        (root / 'capture' / '2').rename(root / 'not_a_capture')
    elif kind == 'extra_frame':
        (root / 'capture' / '3').mkdir()
    elif kind == 'byte_budget':
        (root / 'capture_byte_budget' / 'unexpected.raw').write_bytes(b'bad')
    else:
        manifest = root / 'manifest.json'
        native = json.loads(manifest.read_text())
        native['capture_pending' if kind == 'pending' else 'requested_provider_id'] = 1
        write_json(manifest, native)
    write_json(path, m)
    with pytest.raises(ValueError):
        module.analyze_pair(*pair)
