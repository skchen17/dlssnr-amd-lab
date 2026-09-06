"""Join game readbacks to per-request original-weight inference evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .analyze_full_graph_color import compare, read_verified
except ImportError:
    from analyze_full_graph_color import compare, read_verified


def analyze(session: Path, worker_dir: Path, output: Path):
    worker = json.loads((worker_dir / 'worker.json').read_text(encoding='utf-8-sig'))
    provenance = json.loads((session / 'provenance.json').read_text(encoding='utf-8-sig'))
    status = json.loads((session / 'resident_status.json').read_text(encoding='utf-8-sig'))
    events = []
    for line in (session / 'session.jsonl').read_text(encoding='utf-8-sig').splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # test harness stdout may also appear; never count it as evidence
    frames = [e for e in events if e.get('event') == 'resident_frame_complete']
    if status['failed'] or status['busy'] or status['enabled'] or worker['status'] == 'FAIL':
        raise ValueError('requires completed, stopped, nonfailed session')
    if provenance['exit_code'] != 0 or not provenance['observation_deferred']:
        raise ValueError('game attachment provenance')
    if len(frames) < 2 or len(frames) != len(worker['frames']) or status['completed_frames'] != len(frames):
        raise ValueError('game/worker frame count mismatch')
    if worker['runtime']['model_sha256'] != 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5':
        raise ValueError('original weight provenance')
    if worker['runtime']['rtx_intermediate_injection'] or worker['runtime']['slot_count'] != 156:
        raise ValueError('incomplete or injected graph')
    for index, (frame, inferred) in enumerate(zip(frames, worker['frames']), 1):
        if frame['request_id'] != index or inferred['request_id'] != index or inferred['client_pid'] != provenance['pid']:
            raise ValueError('frame/client identity mismatch')
        if not frame['worker_output_readback_exact'] or not frame['same_submission_frame']:
            raise ValueError('game writeback mismatch')
        if (frame['width'], frame['height']) != (inferred['width'], inferred['height']):
            raise ValueError('frame dimensions mismatch')
        tiles = inferred['tiles']
        w, h = frame['width'], frame['height']
        expected = [[x, y, min(640, w-x), min(360, h-y)] for y in range(0, h, 360) for x in range(0, w, 640)]
        if [t['xywh'] for t in tiles] != expected or any(t['slot_count'] != 156 for t in tiles):
            raise ValueError('full-surface tile coverage/graph incomplete')
    metrics, previews = [], []
    for frame in worker['frames'][:2]:
        seq = frame['request_id']
        before = read_verified(session / 'resident_frames' / f'input{seq}.raw', frame['input_sha256'])
        after = read_verified(session / 'resident_frames' / f'output{seq}.raw', frame['output_sha256'])
        if before != read_verified(worker_dir / f'input{seq}.raw', frame['input_sha256']) or after != read_verified(worker_dir / f'output{seq}.raw', frame['output_sha256']):
            raise ValueError('IPC/game audit byte mismatch')
        shape = (frame['height'], frame['width'], 4)
        a, b = [np.frombuffer(raw, dtype='<f2').reshape(shape).astype('f4') for raw in (before, after)]
        measured = compare(a, b)
        if not measured['finite'] or not measured['alpha_exact_vs_input'] or not measured['changed_rgb_components_vs_clamped_base']:
            raise ValueError('finite/nonidentity/alpha gate')
        metrics.append({'request_id': seq, **measured})
        if seq == 1:
            previews = [('input', a), ('output', b)]
    unique = len({f['input_sha256'] for f in worker['frames']})
    if unique < 2:
        raise ValueError('no changing real inputs observed')
    # This audit ends at texture readback. A downstream consumer may already
    # have run inside the producer list; exact writeback alone is not presentation.
    report = {'status': 'GAME_TEXTURE_INFERENCE_LOOP_PASS', 'pid': provenance['pid'],
        'completed_frames': len(frames), 'unique_input_frames': unique,
        'game_session': str(session.resolve()), 'worker': str(worker_dir.resolve()),
        'model_sha256': worker['runtime']['model_sha256'], 'metrics': metrics,
        'host_surface_ms': [f['host_surface_ms'] for f in worker['frames']],
        'original_weight_live_inference': True, 'whole_surface_covered': True,
        'final_presentation_verified': False,
        'same_submission_frame': True, 'independent_640x360_tiles': True,
        'resized': False, 'full_frame_attention_equivalence': False,
        'temporal_quality_verified': False, 'dlss5_quality_verified': False,
        'smooth_realtime_verified': False, 'arbitrary_resolution_quality_verified': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in previews:
        rgb = np.maximum(frame[..., :3], 0)
        pixels = np.rint(np.power(rgb / (1 + rgb), 1 / 2.2) * 255).astype('uint8')
        Image.fromarray(pixels).save(output / f'{name}.png')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.session, args.worker, args.output), indent=2))
