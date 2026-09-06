import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_teacher_sequences import audit
from run_teacher_sequence import prepare, run


def fixture(tmp_path):
    def asset(name, values):
        raw = values.tobytes()
        (tmp_path / name).write_bytes(raw)
        return {'path': name, 'sha256': hashlib.sha256(raw).hexdigest()}
    color = asset('color.raw', np.arange(16, dtype='<f2'))
    frame = {'frame_id': 0, 'width': 2, 'height': 2, 'reset': True,
             'color_mode': 'SDR', 'color_contract_id': 'fixture_not_game',
             'capture_provenance': 'synthetic_unit_test_only', 'pre_exposure': 1., 'exposure_scale': 1.,
             'motion_scale': [1., 1.], 'jitter': [0., 0.], 'valid_rect': [0, 0, 2, 2],
             'color': color, 'motion': asset('mv.raw', np.zeros(8, dtype='<f2')),
             'depth': asset('depth.raw', np.ones(4, dtype='<f4')),
             'teacher_output': asset('out.raw', np.arange(16, dtype='<f2')),
             'teacher': {'kind': 'nvidia_component', 'input_sha256': color['sha256'],
                         'model_sha256': 'fixture', 'run_provenance': 'fixture_only'}}
    data = {'schema': 1, 'sequences': [{'id': 's0', 'scene_id': 'scene0', 'split': 'train',
                                      'ngx_create_flags': 0, 'frames': [frame]}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    return path, data, frame


def test_partial_structure_is_not_teacher_acceptance(tmp_path):
    path, data, frame = fixture(tmp_path)
    result = audit(path, require_complete=False)
    assert not result['training_allowed']
    assert not result['teacher_cross_device_acceptance']
    with pytest.raises(ValueError):
        audit(path)


@pytest.mark.parametrize('change', ['hash', 'exposure', 'nan', 'path', 'teacher', 'reset', 'mv', 'rect', 'mode'])
def test_invalid_frames_rejected(tmp_path, change):
    path, data, f = fixture(tmp_path)
    if change == 'hash': f['color']['sha256'] = '0' * 64
    if change == 'exposure': f['pre_exposure'] = -1
    if change == 'nan': f['exposure_scale'] = float('nan')
    if change == 'path': f['color']['path'] = str((tmp_path / 'color.raw').resolve())
    if change == 'teacher': f['teacher']['input_sha256'] = '0' * 64
    if change == 'reset': f['reset'] = 0
    if change == 'mv': f.pop('motion_scale')
    if change == 'rect': f['valid_rect'] = [0, 0, 1, 1]
    if change == 'mode': f['color_mode'] = 'unknown'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit(path, require_complete=False)


def test_scene_leakage_rejected(tmp_path):
    path, data, f = fixture(tmp_path)
    other = dict(data['sequences'][0], id='s1', split='test')
    data['sequences'].append(other)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='leakage'):
        audit(path, require_complete=False)


def test_prepare_keeps_inputs_exact_and_never_fills_missing_motion(tmp_path):
    path, data, f = fixture(tmp_path)
    output = tmp_path / 'prepared'
    prepared = prepare(path, output)
    assert len(prepared) == 1
    assert (output / 'sequence_000/0/color.raw').read_bytes() == (tmp_path / 'color.raw').read_bytes()
    assert (output / 'sequence_000/sequence.txt').read_text().startswith('NRTEACHER1 2 2 1 0\n')
    f.pop('motion')
    path.write_text(json.dumps(data))
    with pytest.raises(KeyError):
        prepare(path, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('change', ['flags', 'hdr', 'reset', 'jitter'])
def test_prepare_requires_actual_metadata(tmp_path, change):
    path, data, f = fixture(tmp_path)
    if change == 'flags': data['sequences'][0].pop('ngx_create_flags')
    if change == 'hdr': f['color_mode'] = 'HDR'
    if change == 'reset': f['reset'] = False
    if change == 'jitter': f.pop('jitter')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepare(path, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_prepare_only_archive_contains_no_runtime(tmp_path):
    import zipfile
    path, data, f = fixture(tmp_path)
    output = tmp_path / 'run'
    report = run(path, tmp_path, tmp_path / 'absent.exe', output, prepare_only=True)
    assert report['status'] == 'PREPARED_NOT_RUN'
    with zipfile.ZipFile(output.with_suffix('.zip')) as z:
        assert all(not n.endswith(('.exe', '.dll', '.addon64')) for n in z.namelist())
    with pytest.raises(FileExistsError):
        run(path, tmp_path, tmp_path / 'absent.exe', output, prepare_only=True)
