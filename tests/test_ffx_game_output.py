import copy
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_game_output import analyze


def fixture(root):
    root.mkdir()
    output = root / 'output_boundary'; output.mkdir()
    width, height = 8, 4
    raw = b''.join(struct.pack('<4e', x / 8, y / 4, (x + y) / 12, 1) for y in range(height) for x in range(width))
    metadata = dict(width=width, height=height, format=10, raw_bytes=len(raw), fence_completed=True,
                    game_frame_capture_verified=True, nr_verified=False)
    (output / 'boundary_output.raw').write_bytes(raw)
    (output / 'metadata.json').write_text(json.dumps(metadata))
    common = dict(window=1)
    events = [dict(common, event='observation_idle'), dict(common, event='boundary_output_arm'),
              dict(common, event='output_boundary_return', batch_supported=True, lifecycle_match=True,
                   original_callback_returned=True, original_forward_calls=1, capture_authorized=False),
              dict(common, event='boundary_output_recorded', isolated_host_only=False, game_frame_capture_verified=False),
              dict(common, event='boundary_output_submitted', isolated_host_only=False, game_frame_capture_verified=False,
                   submitted_identity_matches=1, identity_aliases=1, closed_identity_aliases=1, raw_pointer_match=True),
              dict(common, event='boundary_output_complete', **metadata)]
    (root / 'session.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    (root / 'provenance.json').write_text(json.dumps(dict(exit_code=0, before={'a': 'h'}, after={'a': 'h'},
                                                            game_files_deployed=[], observation_deferred=True)))
    return events, metadata, raw


def test_live_output_pass(tmp_path):
    fixture(tmp_path / 'run')
    report = analyze(tmp_path / 'run')
    assert report['status'] == 'GAME_OUTPUT_CAPTURE_PASS'
    assert report['game_frame_capture_verified'] and not report['dlss_nr_verified']
    assert not report['scene_content_verified']  # fixture validates transport, not scene content


def test_rejected_startup_boundary_is_retained(tmp_path):
    root = tmp_path / 'run'; events, _, _ = fixture(root)
    rejected = dict(events[2], batch_supported=True, lifecycle_match=False)
    events.insert(2, rejected)
    (root / 'session.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    assert analyze(root)['status'] == 'GAME_OUTPUT_CAPTURE_PASS'


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'order/count'), ('wrong_order', 'order/count'), ('isolated', 'provenance'),
    ('not_fenced', 'provenance'), ('nr', 'provenance'), ('failure', 'session failure')])
def test_event_failures(tmp_path, mutation, match):
    root = tmp_path / 'run'; events, metadata, raw = fixture(root)
    if mutation == 'duplicate': events.append(copy.deepcopy(events[-1]))
    elif mutation == 'wrong_order': events[-1], events[-2] = events[-2], events[-1]
    elif mutation == 'isolated': events[-3]['isolated_host_only'] = True
    elif mutation == 'not_fenced': events[-1]['fence_completed'] = False
    elif mutation == 'nr': events[-1]['nr_verified'] = True
    else: events.append(dict(event='failure', reason='test'))
    (root / 'session.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    with pytest.raises(ValueError, match=match): analyze(root)


@pytest.mark.parametrize('mutation,match', [('truncate', 'byte count'), ('nan', 'non-finite'),
                                             ('constant', 'constant'), ('format', 'metadata'),
                                             ('tamper', 'provenance')])
def test_artifact_failures(tmp_path, mutation, match):
    root = tmp_path / 'run'; events, metadata, raw = fixture(root)
    path = root / 'output_boundary' / 'boundary_output.raw'
    if mutation == 'truncate': path.write_bytes(raw[:-2])
    elif mutation == 'nan': path.write_bytes(struct.pack('<H', 0x7e00) + raw[2:])
    elif mutation == 'constant': path.write_bytes(struct.pack('<e', 0.0) * (len(raw) // 2))
    elif mutation == 'format':
        metadata['format'] = 2
        (root / 'output_boundary' / 'metadata.json').write_text(json.dumps(metadata))
    else:
        (root / 'provenance.json').write_text(json.dumps(dict(exit_code=0, before={'a': 'h'}, after={'a': 'x'},
                                                                game_files_deployed=[], observation_deferred=True)))
    with pytest.raises(ValueError, match=match): analyze(root)
