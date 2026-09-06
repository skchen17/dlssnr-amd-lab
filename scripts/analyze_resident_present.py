"""Audit same-buffer original inference -> accepted Present, not DLSS5 parity."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .analyze_full_graph_color import read_verified
except ImportError:
    from analyze_full_graph_color import read_verified


def packed_to_half(raw, fmt):
    v = np.frombuffer(raw, dtype='<u4')
    if fmt == 24:
        channels = [(v >> shift) & mask for shift, mask in [(0, 1023), (10, 1023), (20, 1023), (30, 3)]]
        scale = np.array([1023, 1023, 1023, 3], dtype='f4')
    elif fmt in (28, 87):
        channels = [(v >> shift) & 255 for shift in (0, 8, 16, 24)]
        if fmt == 87:
            channels[0], channels[2] = channels[2], channels[0]
        scale = np.full(4, 255, dtype='f4')
    else:
        raise ValueError('unsupported packed format')
    return (np.stack(channels, axis=1).astype('f4') / scale).astype('<f2')


def half_to_packed(half, base, fmt):
    rgb = np.clip(half.astype('f4')[:, :3], 0, 1)
    if not np.isfinite(half).all():
        raise ValueError('nonfinite half')
    original = np.frombuffer(base, dtype='<u4')
    if fmt == 24:
        q = np.floor(rgb * 1023 + .5).astype('u4')
        encoded = q[:, 0] | (q[:, 1] << 10) | (q[:, 2] << 20) | (original & 0xc0000000)
    elif fmt in (28, 87):
        if fmt == 87:
            rgb = rgb[:, ::-1]
        q = np.floor(rgb * 255 + .5).astype('u4')
        encoded = q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16) | (original & 0xff000000)
    else:
        raise ValueError('unsupported packed format')
    return encoded.astype('<u4').tobytes()


def analyze(session, output):
    provenance = json.loads((session / 'provenance.json').read_text(encoding='utf-8-sig'))
    status = json.loads((session / 'resident_status.json').read_text(encoding='utf-8-sig'))
    worker_dir = Path(provenance['worker'])
    worker = json.loads((worker_dir / 'worker.json').read_text(encoding='utf-8-sig'))
    if not provenance['display_referred_debug_route'] or provenance['exit_code'] != 0:
        raise ValueError('route provenance')
    if any(status[k] for k in ('enabled', 'busy', 'failed')) or worker['status'] == 'FAIL':
        raise ValueError('session not stopped successfully')
    rt = worker['runtime']
    if rt['model_sha256'] != 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5' or rt['slot_count'] != 156 or rt['rtx_intermediate_injection']:
        raise ValueError('original graph provenance')
    events = [json.loads(x) for x in (session / 'events.jsonl').read_text().splitlines()]
    if any(e.get('event') == 'failure' for e in events):
        raise ValueError('session failure event')
    completed = [e for e in events if e.get('event') == 'resident_present_complete']
    if len(completed) < 2 or len(completed) != len(worker['frames']) or len(completed) != status['completed_frames']:
        raise ValueError('frame count mismatch')
    output.mkdir(parents=True, exist_ok=False)
    changed, images = [], []
    for i, (event, inferred) in enumerate(zip(completed, worker['frames']), 1):
        w, h = event['width'], event['height']
        if event['request_id'] != i or inferred['request_id'] != i or inferred['client_pid'] != provenance['pid'] or (w, h) != (inferred['width'], inferred['height']):
            raise ValueError('frame identity')
        if not event['encoded_readback_exact'] or not event['same_present_frame'] or event['present_hresult'] != 0:
            raise ValueError('Present/writeback contract')
        expected = [[x, y, min(640, w-x), min(360, h-y)] for y in range(0, h, 360) for x in range(0, w, 640)]
        if [t['xywh'] for t in inferred['tiles']] != expected or any(t['slot_count'] != 156 for t in inferred['tiles']):
            raise ValueError('full graph coverage')
        if i > 2:
            continue
        folder = session / 'present_frames'
        a = read_verified(folder / f'input{i}.raw', inferred['input_sha256'])
        b = read_verified(folder / f'output{i}.raw', inferred['output_sha256'])
        if a != read_verified(worker_dir / f'input{i}.raw', inferred['input_sha256']) or b != read_verified(worker_dir / f'output{i}.raw', inferred['output_sha256']):
            raise ValueError('worker byte correspondence')
        before, after = [(folder / f'encoded_{name}{i}.raw').read_bytes() for name in ('input', 'output')]
        if len(before) != w*h*4 or len(after) != len(before):
            raise ValueError('packed shape')
        ah, bh = [np.frombuffer(x, dtype='<f2').reshape(-1, 4) for x in (a, b)]
        if packed_to_half(before, event['format']).tobytes() != a or half_to_packed(bh, before, event['format']) != after:
            raise ValueError('independent packed conversion')
        if not np.isfinite(bh).all() or not np.array_equal(ah[:, 3], bh[:, 3]):
            raise ValueError('finite/alpha')
        count = int(np.count_nonzero(np.frombuffer(before, '<u4') != np.frombuffer(after, '<u4')))
        if not count:
            raise ValueError('identity output')
        changed.append(count)
        if i == 1:
            images = [(name, packed_to_half(data, event['format']).astype('f4').reshape(h, w, 4)) for name, data in [('input', before), ('output', after)]]
    unique = len({f['input_sha256'] for f in worker['frames']})
    if unique < 2:
        raise ValueError('unchanging inputs')
    report = {'status': 'SAME_BUFFER_INFERENCE_PRESENT_PASS', 'pid': provenance['pid'],
        'completed_frames': len(completed), 'unique_input_frames': unique,
        'changed_packed_pixels_first_two': changed, 'original_model_sha256': rt['model_sha256'],
        'width': completed[0]['width'], 'height': completed[0]['height'],
        'host_surface_ms': [f['host_surface_ms'] for f in worker['frames']],
        'same_current_buffer': True, 'present_api_accepted': True,
        'physical_screen_visual_verification': 'separate screenshot audit required',
        'display_referred_debug_route': True, 'includes_hud': True, 'independent_tiles': True,
        'nr_color_contract_verified': False, 'dlss5_quality_verified': False,
        'smooth_realtime_verified': False, 'arbitrary_resolution_quality_verified': False}
    for name, array in images:
        Image.fromarray(np.rint(array[..., :3]*255).astype('uint8')).save(output / f'{name}.png')
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.session, args.output), indent=2))
