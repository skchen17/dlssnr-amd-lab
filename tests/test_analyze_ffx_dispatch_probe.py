import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('ffx_dispatch_analysis', Path(__file__).parents[1] / 'scripts/analyze_ffx_dispatch_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def artifacts(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(status='DISPATCH_READBACK_PASS', debug_errors=0, completed_fences=4, output_poisoned_nan=True)))
    a = np.zeros((360, 640, 4), dtype='<f2')
    a[..., 0] = np.linspace(.1, .8, 640)
    a[..., 1] = .5
    a[..., 2] = .25
    a[..., 3] = 1
    b = a.copy()
    b[..., 2] = .75
    for i in range(4):
        frame = b if i == 2 else a
        (tmp_path / f'output_{i}.raw').write_bytes(frame.tobytes())
        (tmp_path / f'input_{i}.raw').write_bytes(frame[::2, ::2].tobytes())
    return tmp_path


def test_valid_control(artifacts):
    result = module.analyze_provider(artifacts)
    assert result['status'] == 'ARTIFACT_PASS'
    assert result['dlss_nr_verified'] is False
    assert result['game_runtime_ready'] is False


@pytest.mark.parametrize('kind,pattern', [('truncated','size'),('nan','nonfinite'),('blank','constant'),('duplicate','sensitivity'),('changed_repeat','repeatability')])
def test_corruption_rejected(artifacts, kind, pattern):
    raw = (artifacts / 'output_0.raw').read_bytes()
    frame = np.frombuffer(raw, '<f2').copy()
    target = 'output_0.raw'
    if kind == 'truncated':
        raw = raw[:-2]
    elif kind == 'nan':
        frame[0] = np.nan
        raw = frame.tobytes()
    elif kind == 'blank':
        raw = np.zeros_like(frame).tobytes()
    elif kind == 'duplicate':
        target = 'output_2.raw'
    else:
        frame[0] = .75
        raw = frame.tobytes()
    (artifacts / target).write_bytes(raw)
    with pytest.raises(ValueError, match=pattern):
        module.analyze_provider(artifacts)


@pytest.mark.parametrize('field,value', [('status','PROBE_FAILED'), ('debug_errors',1), ('completed_fences',3), ('output_poisoned_nan',False)])
def test_native_failure_rejected(artifacts, field, value):
    path = artifacts / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.analyze_provider(artifacts)

@pytest.mark.parametrize('value',[[0,360],[640,8193],[True,360],[],[640],[640,360,1],None])
def test_bad_output_dimensions(artifacts,value):
    path=artifacts/'manifest.json';m=json.loads(path.read_text());m['output_size']=value;path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='dimensions'):
        module.analyze_provider(artifacts)
