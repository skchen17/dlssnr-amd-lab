import json
import numpy as np
import pytest
from scripts.analyze_resident_present import packed_to_half, half_to_packed, analyze
from scripts.run_full_graph_integrated import sha256


@pytest.mark.parametrize('fmt', [24, 28, 87])
def test_packed_half_roundtrip_and_alpha(fmt):
    raw = np.arange(65536, dtype='u4') * np.uint32(0x1020401)
    data = raw.astype('<u4').tobytes()
    half = packed_to_half(data, fmt)
    assert half_to_packed(half, data, fmt) == data
    half[:, 3] = 0
    assert half_to_packed(half, data, fmt) == data  # game alpha is preserved
    half[0, 1] = float('nan')
    with pytest.raises(ValueError):
        half_to_packed(half, data, fmt)


def fixture(tmp_path):
    session, worker = tmp_path / 'session', tmp_path / 'worker'
    (session / 'present_frames').mkdir(parents=True)
    worker.mkdir()
    (session / 'provenance.json').write_text(json.dumps({'pid': 123, 'worker': str(worker), 'exit_code': 0, 'display_referred_debug_route': True}))
    (session / 'resident_status.json').write_text(json.dumps({'enabled': False, 'busy': False, 'failed': False, 'completed_frames': 2}))
    events, frames = [], []
    for i in (1, 2):
        raw = np.array([0xff442211+i], dtype='<u4').tobytes()
        a = packed_to_half(raw, 28)
        b = a.copy(); b[:, 0] = .75
        encoded = half_to_packed(b, raw, 28)
        for directory in (session / 'present_frames', worker):
            (directory / f'input{i}.raw').write_bytes(a.tobytes())
            (directory / f'output{i}.raw').write_bytes(b.tobytes())
        (session / 'present_frames' / f'encoded_input{i}.raw').write_bytes(raw)
        (session / 'present_frames' / f'encoded_output{i}.raw').write_bytes(encoded)
        events.append({'event': 'resident_present_complete', 'request_id': i, 'width': 1, 'height': 1, 'format': 28, 'encoded_readback_exact': True, 'same_present_frame': True, 'present_hresult': 0})
        frames.append({'request_id': i, 'width': 1, 'height': 1, 'client_pid': 123, 'input_sha256': sha256(a.tobytes()), 'output_sha256': sha256(b.tobytes()), 'host_surface_ms': 10, 'tiles': [{'xywh': [0, 0, 1, 1], 'slot_count': 156}]})
    (session / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    (worker / 'worker.json').write_text(json.dumps({'status': 'FRAME_LIMIT', 'frames': frames, 'runtime': {'model_sha256': 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5', 'slot_count': 156, 'rtx_intermediate_injection': False}}))
    return session


def test_present_contract_not_quality_or_physical_screen_claim(tmp_path):
    report = analyze(fixture(tmp_path), tmp_path / 'out')
    assert report['status'] == 'SAME_BUFFER_INFERENCE_PRESENT_PASS'
    assert report['includes_hud'] and not report['nr_color_contract_verified']
    assert not report['dlss5_quality_verified']
    assert report['physical_screen_visual_verification'] == 'separate screenshot audit required'


@pytest.mark.parametrize('fault', ['occluded', 'wrong_sequence', 'encoded_corruption', 'failure'])
def test_present_audit_fails_closed(tmp_path, fault):
    session = fixture(tmp_path)
    events = [json.loads(line) for line in (session / 'events.jsonl').read_text().splitlines()]
    if fault == 'occluded':
        events[0]['present_hresult'] = 0x087a0001
    elif fault == 'wrong_sequence':
        events[0]['request_id'] = 10
    elif fault == 'failure':
        events.append({'event': 'failure'})
    else:
        (session / 'present_frames' / 'encoded_output1.raw').write_bytes(bytes(4))
    (session / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError):
        analyze(session, tmp_path / 'out')
