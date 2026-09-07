"""Fail-closed audit for original-component temporal sequence captures.

This validates identity, continuity and storage contracts only.  It never turns
a structurally valid capture into an accepted teacher or inferred temporal
mapping; that requires a separate bounded-difference contract report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIXED_ASSETS = {
    'current_color': ('<f2', lambda h, w: (h, w, 4)),
    'motion': ('<f2', lambda h, w: (h, w, 2)),
    'depth': ('<f4', lambda h, w: (h, w)),
    'exposure': ('<f4', lambda _h, _w: (1,)),
    'next_history': ('<f2', lambda h, w: (h, w, 4)),
    'teacher_output': ('<f2', lambda h, w: (h, w, 4)),
}
GENERIC_ASSETS = ('pre_boundary', 'post_boundary')


def _finite_vector(value, length):
    return (isinstance(value, list) and len(value) == length and
            all(type(x) in (int, float) and math.isfinite(x) for x in value))


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('asset path must be a non-empty relative path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('asset escaped dataset root')
    return path


def _asset(root: Path, spec: dict, *, dtype: str, shape: tuple[int, ...], role: str):
    if spec.get('dtype') != dtype or spec.get('shape') != list(shape):
        raise ValueError(f'{role} dtype/shape mismatch')
    if spec.get('origin') != 'captured_gpu_resource' or not spec.get('resource_identity'):
        raise ValueError(f'{role} is not tied to a captured GPU resource')
    path = _inside(root, spec.get('path'))
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f'{role} byte size mismatch')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    if digest != str(spec.get('sha256', '')).upper():
        raise ValueError(f'{role} hash mismatch')
    if np.dtype(dtype).kind == 'f' and not np.isfinite(np.frombuffer(raw, dtype=dtype)).all():
        raise ValueError(f'{role} contains non-finite values')
    return digest


def _generic_asset(root: Path, spec: dict, role: str):
    dtype = spec.get('dtype')
    shape = spec.get('shape')
    if dtype not in ('<f2', '<f4', '|u1') or not isinstance(shape, list) or not shape or \
            any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError(f'{role} needs an explicit supported dtype/shape')
    return _asset(root, spec, dtype=dtype, shape=tuple(shape), role=role)


def audit(manifest_path: Path, *, require_complete=True) -> dict:
    root = manifest_path.resolve().parent
    data = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 2 or data.get('capture_contract') != 'original_nr_temporal_v1':
        raise ValueError('unknown temporal dataset contract')
    if data.get('teacher_kind') != 'nvidia_component' or not data.get('teacher_model_sha256'):
        raise ValueError('original component provenance is required')
    splits = {'train': 0, 'validation': 0, 'test': 0}
    scenes, sequence_ids, modes, input_splits = {}, set(), set(), {}
    frame_count = reset_count = history_count = 0
    for sequence in data.get('sequences', []):
        sid, scene, split = sequence.get('id'), sequence.get('scene_id'), sequence.get('split')
        if not sid or not scene or sid in sequence_ids or split not in splits:
            raise ValueError('invalid or duplicate sequence identity')
        if scenes.setdefault(scene, split) != split:
            raise ValueError('scene leakage between splits')
        sequence_ids.add(sid); splits[split] += 1
        frames = sequence.get('frames')
        if not isinstance(frames, list) or not frames or len(frames) > 32 or \
                (require_complete and len(frames) != 32):
            raise ValueError('each temporal sequence must contain exactly 32 frames')
        if frames[0].get('reset') is not True:
            raise ValueError('sequence must begin with reset')
        previous_id = None
        for position, frame in enumerate(frames):
            h, w, frame_id = frame.get('height'), frame.get('width'), frame.get('frame_id')
            if any(type(x) is not int or x <= 0 for x in (h, w)) or \
                    type(frame_id) is not int or frame_id < 0:
                raise ValueError('invalid frame geometry/id')
            if previous_id is not None and frame_id != previous_id + 1:
                raise ValueError('non-consecutive frame IDs')
            previous_id = frame_id
            reset = frame.get('reset')
            if type(reset) is not bool or (position and reset):
                raise ValueError('only the first frame may reset within a sequence')
            mode = frame.get('color_mode')
            if mode not in ('SDR', 'HDR') or not frame.get('color_contract_id'):
                raise ValueError('explicit SDR/HDR color contract required')
            modes.add(mode)
            if frame.get('valid_rect') != [0, 0, w, h] or not _finite_vector(frame.get('jitter'), 2) or \
                    not _finite_vector(frame.get('motion_scale'), 2):
                raise ValueError('valid rect/jitter/motion scale missing')
            for field in ('pre_exposure', 'exposure_scale'):
                value = frame.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError('positive finite exposure metadata required')
            hashes = {}
            for role, (dtype, shape_fn) in FIXED_ASSETS.items():
                hashes[role] = _asset(root, frame.get(role, {}), dtype=dtype,
                                      shape=shape_fn(h, w), role=role)
            for role in GENERIC_ASSETS:
                hashes[role] = _generic_asset(root, frame.get(role, {}), role)
            previous_role = 'reset_previous_color' if reset else 'previous_history'
            hashes[previous_role] = _asset(root, frame.get(previous_role, {}), dtype='<f2',
                                           shape=(h, w, 4), role=previous_role)
            if reset and 'previous_history' in frame:
                raise ValueError('reset frame must identify reset previous-color input, not history')
            if not reset and 'reset_previous_color' in frame:
                raise ValueError('non-reset frame must identify history input')
            current_hash = hashes['current_color']
            if current_hash in input_splits and input_splits[current_hash] != split:
                raise ValueError('identical current input leaks between splits')
            input_splits[current_hash] = split
            if not frame.get('capture_provenance') or not frame.get('run_provenance'):
                raise ValueError('frame capture/run provenance missing')
            reset_count += int(reset); history_count += int(not reset); frame_count += 1
    if not sequence_ids:
        raise ValueError('empty temporal dataset')
    if require_complete and (splits != {'train': 16, 'validation': 4, 'test': 4} or
                             modes != {'SDR', 'HDR'}):
        raise ValueError('24-sequence 16/4/4 SDR+HDR coverage incomplete')
    return {
        'schema': 2,
        'status': 'TEMPORAL_CAPTURE_STRUCTURE_VALID_NOT_SEMANTICS_ACCEPTED',
        'sequences': len(sequence_ids), 'frames': frame_count, 'splits': splits,
        'color_modes': sorted(modes), 'reset_frames': reset_count,
        'history_frames': history_count, 'temporal_semantics_accepted': False,
        'runtime_temporal_mode_allowed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, require_complete=not args.allow_partial), indent=2))


if __name__ == '__main__':
    main()
