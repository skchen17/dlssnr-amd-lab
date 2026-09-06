import json
import struct

import pytest

from scripts.analyze_resident_game import analyze
from scripts.run_full_graph_integrated import sha256


def fixture(tmp_path):
    session, worker = tmp_path / 'session', tmp_path / 'worker'
    (session / 'resident_frames').mkdir(parents=True)
    worker.mkdir()
    (session / 'provenance.json').write_text(json.dumps({'pid': 123, 'exit_code': 0, 'observation_deferred': True}))
    status = {'enabled': False, 'busy': False, 'failed': False, 'completed_frames': 2}
    (session / 'resident_status.json').write_text(json.dumps(status))
    events, frames = [], []
    for i in (1, 2):
        a, b = struct.pack('<4e', i/4, 0, 0, 1), struct.pack('<4e', .75, .25, .125, 1)
        for directory in (worker, session / 'resident_frames'):
            (directory / f'input{i}.raw').write_bytes(a)
            (directory / f'output{i}.raw').write_bytes(b)
        events.append({'event': 'resident_frame_complete', 'request_id': i, 'width': 1, 'height': 1,
                       'worker_output_readback_exact': True, 'same_submission_frame': True})
        frames.append({'request_id': i, 'width': 1, 'height': 1, 'client_pid': 123,
                       'input_sha256': sha256(a), 'output_sha256': sha256(b), 'host_surface_ms': 10,
                       'tiles': [{'xywh': [0, 0, 1, 1], 'slot_count': 156}]})
    (session / 'session.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    report = {'status': 'FRAME_LIMIT', 'frames': frames, 'runtime': {'slot_count': 156,
        'rtx_intermediate_injection': False,
        'model_sha256': 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'}}
    (worker / 'worker.json').write_text(json.dumps(report))
    return session, worker, report


def test_joined_game_evidence_has_no_quality_or_attention_claim(tmp_path):
    session, worker, _ = fixture(tmp_path)
    result = analyze(session, worker, tmp_path / 'analysis')
    assert result['status'] == 'GAME_TEXTURE_INFERENCE_LOOP_PASS'
    assert not result['final_presentation_verified']
    assert result['whole_surface_covered'] and not result['dlss5_quality_verified']
    assert not result['full_frame_attention_equivalence']


@pytest.mark.parametrize('fault', ['wrong_pid', 'missing_tile', 'missing_slot', 'injection', 'identical_inputs'])
def test_broken_provenance_rejected(tmp_path, fault):
    session, worker, report = fixture(tmp_path)
    if fault == 'wrong_pid':
        report['frames'][0]['client_pid'] = 999
    elif fault == 'missing_tile':
        report['frames'][0]['tiles'] = []
    elif fault == 'missing_slot':
        report['frames'][0]['tiles'][0]['slot_count'] = 155
    elif fault == 'injection':
        report['runtime']['rtx_intermediate_injection'] = True
    else:
        report['frames'][1]['input_sha256'] = report['frames'][0]['input_sha256']
    (worker / 'worker.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        analyze(session, worker, tmp_path / 'analysis')
