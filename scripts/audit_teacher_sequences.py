"""Validate private paired-frame manifests; structural validity is NOT teacher acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_frame_asset(root, spec, shape, dtype):
    path = (root / spec['path']).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or Path(spec['path']).is_absolute():
        raise ValueError('asset must stay inside dataset root')
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f'asset size mismatch: {spec["path"]}')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest().upper() != spec['sha256'].upper():
        raise ValueError(f'asset hash mismatch: {spec["path"]}')
    values = np.frombuffer(raw, dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite source/teacher pixels')
    return spec['sha256'].upper()


def audit(manifest_path, require_complete=True, require_teacher=True):
    root = manifest_path.resolve().parent
    data = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 1:
        raise ValueError('unknown dataset schema')
    scenes, identifiers, frame_hashes = {}, set(), {}
    counts = dict(train=0, validation=0, test=0)
    frames = 0
    modes = set()
    for sequence in data['sequences']:
        sid, scene, split = sequence['id'], sequence['scene_id'], sequence['split']
        if not sid or not scene or sid in identifiers or split not in counts:
            raise ValueError('invalid/duplicate sequence identity or split')
        if scene in scenes and scenes[scene] != split:
            raise ValueError('scene leakage between splits')
        scenes[scene] = split
        identifiers.add(sid)
        counts[split] += 1
        seq = sequence['frames']
        if not seq or (require_complete and len(seq) < 32):
            raise ValueError('sequence requires at least 32 consecutive frames')
        previous = None
        for frame in seq:
            w, h, index = frame['width'], frame['height'], frame['frame_id']
            if type(w) is not int or type(h) is not int or w <= 0 or h <= 0 or type(index) is not int or index < 0:
                raise ValueError('invalid dimensions/frame ID')
            if previous is not None and index != previous + 1:
                raise ValueError('frames are not consecutive')
            previous = index
            if type(frame.get('reset')) is not bool:
                raise ValueError('explicit reset flag required')
            if frame['color_mode'] not in ('SDR', 'HDR'):
                raise ValueError('unknown color mode')
            modes.add(frame['color_mode'])
            if not frame.get('color_contract_id') or not frame.get('capture_provenance'):
                raise ValueError('color/capture provenance missing')
            for name in ('pre_exposure', 'exposure_scale'):
                if not finite_number(frame.get(name)) or frame[name] <= 0:
                    raise ValueError('explicit positive exposure required')
            scales = frame.get('motion_scale')
            if not isinstance(scales, list) or len(scales) != 2 or not all(finite_number(x) for x in scales):
                raise ValueError('explicit motion scale required')
            if frame.get('valid_rect') != [0, 0, w, h]:
                raise ValueError('v1 requires full-frame valid rectangle')
            color_hash = checked_frame_asset(root, frame['color'], (h, w, 4), '<f2')
            checked_frame_asset(root, frame['motion'], (h, w, 2), '<f2')
            checked_frame_asset(root, frame['depth'], (h, w), '<f4')
            if require_teacher:
                checked_frame_asset(root, frame['teacher_output'], (h, w, 4), '<f2')
                teacher = frame['teacher']
                if teacher.get('input_sha256', '').upper() != color_hash:
                    raise ValueError('teacher input identity mismatch')
                if teacher.get('kind') != 'nvidia_component' or not teacher.get('model_sha256') or not teacher.get('run_provenance'):
                    raise ValueError('original teacher model/run provenance missing')
            if color_hash in frame_hashes and frame_hashes[color_hash] != split:
                raise ValueError('identical input leaks between splits')
            frame_hashes[color_hash] = split
            frames += 1
    if not identifiers:
        raise ValueError('empty dataset')
    if require_complete and (any(counts[k] < n for k, n in [('train', 16), ('validation', 4), ('test', 4)]) or modes != {'SDR', 'HDR'}):
        raise ValueError('24-sequence 16/4/4 SDR+HDR coverage incomplete')
    return {'status': 'STRUCTURE_VALID_NOT_TEACHER_ACCEPTED', 'sequences': len(identifiers),
            'frames': frames, 'splits': counts, 'color_modes': sorted(modes),
            'teacher_cross_device_acceptance': False, 'training_allowed': False,
            'note': 'Separate teacher authenticity/RTX comparison and color-contract gates are required.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    print(json.dumps(audit(args.manifest, not args.allow_partial), indent=2))
