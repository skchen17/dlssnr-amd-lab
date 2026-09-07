import hashlib
import json

import numpy as np
import pytest

from scripts.audit_temporal_teacher_sequences import audit


def _write(root, name, array, identity):
    path = root / name
    raw = array.tobytes()
    path.write_bytes(raw)
    return {'path': name, 'sha256': hashlib.sha256(raw).hexdigest(),
            'dtype': array.dtype.str, 'shape': list(array.shape),
            'origin': 'captured_gpu_resource', 'resource_identity': identity}


def fixture(root, *, second_reset=False, omit_motion=False):
    h, w = 2, 3
    frames = []
    for i in range(2):
        reset = i == 0 or second_reset
        f = {'frame_id': 10 + i, 'width': w, 'height': h, 'reset': reset,
             'valid_rect': [0, 0, w, h], 'color_mode': 'SDR',
             'color_contract_id': 'linear-rec709-v1', 'jitter': [0.1, -0.2],
             'motion_scale': [w, h], 'pre_exposure': 1.0, 'exposure_scale': 1.0,
             'capture_provenance': 'fixture', 'run_provenance': 'fixture-run'}
        arrays = {
            'current_color': np.full((h, w, 4), i + 0.25, dtype='<f2'),
            'motion': np.zeros((h, w, 2), dtype='<f2'),
            'depth': np.full((h, w), 0.5, dtype='<f4'),
            'exposure': np.ones((1,), dtype='<f4'),
            'next_history': np.full((h, w, 4), i + 0.5, dtype='<f2'),
            'teacher_output': np.full((h, w, 4), i + 0.75, dtype='<f2'),
            'pre_boundary': np.full((1, 2, 16), i + 1, dtype='<f2'),
            'post_boundary': np.full((1, 2, 4), i + 2, dtype='<f2'),
            'reset_previous_color' if reset else 'previous_history':
                np.full((h, w, 4), i, dtype='<f2'),
        }
        for role, value in arrays.items():
            f[role] = _write(root, f'{i}_{role}.raw', value, f'{role}-{i}')
        if omit_motion:
            del f['motion']
        frames.append(f)
    manifest = {'schema': 2, 'capture_contract': 'original_nr_temporal_v1',
                'teacher_kind': 'nvidia_component', 'teacher_model_sha256': 'A' * 64,
                'sequences': [{'id': 's0', 'scene_id': 'scene0', 'split': 'train',
                               'frames': frames}]}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path


def test_partial_sequence_is_structurally_valid_but_not_semantics_accepted(tmp_path):
    result = audit(fixture(tmp_path), require_complete=False)
    assert result['frames'] == 2
    assert result['history_frames'] == 1
    assert not result['temporal_semantics_accepted']
    assert not result['runtime_temporal_mode_allowed']


def test_missing_real_motion_is_not_filled(tmp_path):
    with pytest.raises(ValueError, match='motion'):
        audit(fixture(tmp_path, omit_motion=True), require_complete=False)


def test_mid_sequence_reset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='only the first frame'):
        audit(fixture(tmp_path, second_reset=True), require_complete=False)
