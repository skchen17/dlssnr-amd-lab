"""Read-only collector manifest audit; never invent missing FSR resources.

Exit 0 means a candidate sequence has complete metadata/files, NOT proven image
quality or valid NR color semantics. Exit 2 means no eligible sequence exists.
"""
import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def assess_frame(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    errors = []
    if data.get('game_frame') is not True:
        errors.append('not_attested_real_game_frame')
    if data.get('fence_completed') is not True:
        errors.append('capture_fence_not_completed')
    for key in ('render_size', 'effective_upscale_size'):
        size = data.get(key)
        if not isinstance(size, list) or len(size) != 2 or not all(type(x) is int and x > 0 for x in size):
            errors.append('invalid_' + key)
    for key in ('jitter', 'motion_scale'):
        pair = data.get(key)
        if not isinstance(pair, list) or len(pair) != 2 or not all(finite(x) for x in pair):
            errors.append('invalid_' + key)
    for key in ('pre_exposure', 'frame_time_ms', 'camera_near', 'camera_far', 'camera_fov_y', 'view_to_meters'):
        if not finite(data.get(key)):
            errors.append('invalid_' + key)
    for key in ('pre_exposure', 'frame_time_ms', 'view_to_meters'):
        if finite(data.get(key)) and data[key] <= 0:
            errors.append('nonpositive_' + key)
    if type(data.get('reset')) is not bool or type(data.get('frame')) is not int:
        errors.append('invalid_frame_or_reset')
    flags = data.get('context_create_flags')
    if type(flags) is not int or flags < 0:
        errors.append('missing_context_flags')
        flags = 0
    resources = data.get('resources', [])
    if not isinstance(resources, list) or not all(isinstance(r, dict) for r in resources):
        resources = []
        errors.append('invalid_resources')
    roles = {r.get('role'): r for r in resources}
    if len(roles) != len(resources):
        errors.append('duplicate_resource_role')
    required = ['color', 'depth', 'motion']
    if not flags & (1 << 5):
        required.append('exposure')
    for role in required:
        if role not in roles:
            errors.append('missing_' + role)
    for role, resource in roles.items():
        # Existing collector uses these fixed filenames, not caller paths.
        if role not in ('color', 'depth', 'motion', 'exposure', 'reactive', 'transparency', 'output'):
            errors.append('unknown_role')
            continue
        raw = path.parent / (role + '.raw')
        size = resource.get('raw_bytes')
        if type(size) is not int or size <= 0 or not raw.is_file() or raw.stat().st_size != size:
            errors.append('invalid_file_' + role)
        width, height = resource.get('width'), resource.get('height')
        expected = [1, 1] if role == 'exposure' else data.get('effective_upscale_size') if role == 'output' or (role == 'motion' and flags & 2) else data.get('render_size')
        if [width, height] != expected:
            errors.append('dimension_mismatch_' + role)
        row_bytes = resource.get('row_bytes')
        if type(row_bytes) is not int or type(height) is not int or row_bytes <= 0 or row_bytes * height != size:
            errors.append('invalid_tight_rows_' + role)
        if type(resource.get('dxgi_format')) is not int:
            errors.append('missing_format_' + role)
    return {'path': str(path), 'frame': data.get('frame'), 'game_frame': data.get('game_frame'),
            'reset': data.get('reset'), 'render_size': data.get('render_size'),
            'upscale_size': data.get('effective_upscale_size'), 'errors': errors,
            'metadata_files_complete': not errors}


def assess(root, min_frames=32):
    frames, malformed = [], []
    for path in sorted(Path(root).rglob('manifest.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if isinstance(data, dict) and 'game_frame' in data and 'resources' in data:
                frames.append(assess_frame(path))
        except (ValueError, OSError, TypeError) as exc:
            malformed.append({'path': str(path), 'error': str(exc)})
    groups = {}
    for frame in frames:
        groups.setdefault(str(Path(frame['path']).parent.parent), []).append(frame)
    sequences = []
    for directory, group in groups.items():
        valid = all(type(f['frame']) is int for f in group)
        ordered = sorted(group, key=lambda f: f['frame']) if valid else group
        complete = valid and len(ordered) >= min_frames and all(f['metadata_files_complete'] for f in ordered)
        consecutive = valid and all(b['frame'] == a['frame'] + 1 for a, b in zip(ordered, ordered[1:]))
        same_size = len({str((f['render_size'], f['upscale_size'])) for f in ordered}) == 1
        history = ordered[0]['reset'] is True and any(f['reset'] is False for f in ordered[1:])
        sequences.append({'directory': directory, 'count': len(group), 'consecutive': consecutive,
                          'history_retained': history, 'metadata_files_complete': complete,
                          'candidate_for_semantic_review': complete and consecutive and same_size and history})
    return {'collector_frames': len(frames), 'real_game_frames': sum(f['game_frame'] is True for f in frames),
            'frames': frames, 'sequences': sequences, 'malformed': malformed,
            'quality_ready': False,
            'remaining_gates': ['provider_3.1_identity', 'actual_resource_semantics_and_hashes',
                                'NR_preserves_jitter_color_exposure_and_motion_alignment',
                                'persistent_FSR_replay_and_sequence_quality']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--min-frames', type=int, default=32)
    args = parser.parse_args()
    if not args.root.is_dir() or args.min_frames < 2:
        parser.error('root must exist; min-frames must be >= 2')
    result = assess(args.root, args.min_frames)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if any(s['candidate_for_semantic_review'] for s in result['sequences']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
