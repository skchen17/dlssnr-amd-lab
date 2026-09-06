import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_output_filter import analyze


def fixture(root, live=True):
    root.mkdir()
    output = root / 'output_filter'
    output.mkdir()
    width, height = 4, 3
    before = b''.join(struct.pack('<4e', x / 8, y / 8, (x + y) / 16, 1)
                      for y in range(height) for x in range(width))
    after = b''.join(struct.pack('<4e', x / 8 + .01, y / 8 + .01, (x + y) / 16 + .01, 1)
                     for y in range(height) for x in range(width))
    metadata = dict(width=width, height=height, format=10, raw_bytes=len(after), fence_completed=True,
                    write_back_performed=True, replacement_pixels_supplied=True, patch_rect=None,
                    filter_kind='dynamic_residual_5tap', gpu_residual_filter=True, trained_weights=False,
                    game_frame_capture_verified=live, nr_verified=False)
    (output / 'boundary_before.raw').write_bytes(before)
    (output / 'boundary_output.raw').write_bytes(after)
    (output / 'metadata.json').write_text(json.dumps(metadata))
    common = dict(window=1)
    events = [
        dict(common, event='boundary_output_arm', write_back_performed=True, replacement_pixels_supplied=True),
        dict(common, event='output_boundary_return', batch_supported=True, lifecycle_match=True),
        dict(common, event='boundary_output_recorded', isolated_host_only=not live,
             write_back_performed=True, replacement_pixels_supplied=True, gpu_residual_filter_recorded=True,
             internal_hook_bypass=True),
        dict(common, event='boundary_output_submitted', isolated_host_only=not live,
             write_back_performed=True, replacement_pixels_supplied=True, submitted_identity_matches=1,
             submission_split_performed=True, submission_batch_count=3, submission_batch_index=0,
             submission_suffix_lists=2),
        dict(common, event='boundary_output_complete', **metadata),
    ]
    (root / 'session.jsonl').write_text('\n'.join(json.dumps(event) for event in events) + '\n')
    (root / 'provenance.json').write_text(json.dumps(dict(
        exit_code=0, before={'game': 'hash'}, after={'game': 'hash'}, game_files_deployed=[],
        observation_deferred=True)))
    return events, metadata, before, after


def test_live_and_isolated_pass(tmp_path):
    fixture(tmp_path / 'live')
    fixture(tmp_path / 'isolated', False)
    assert analyze(tmp_path / 'live')['status'] == 'GAME_GPU_RESIDUAL_FILTER_PASS'
    assert analyze(tmp_path / 'isolated', False)['status'] == 'ISOLATED_GPU_RESIDUAL_FILTER_PASS'


@pytest.mark.parametrize('mutation,match', [
    ('same', 'residual change'), ('alpha', 'residual change'), ('nan', 'non-finite'),
    ('split', 'provenance'), ('bypass', 'provenance'), ('metadata', 'metadata'),
])
def test_rejects_tampering(tmp_path, mutation, match):
    root = tmp_path / 'run'
    events, metadata, before, after = fixture(root)
    if mutation == 'same':
        (root / 'output_filter' / 'boundary_output.raw').write_bytes(before)
    elif mutation == 'alpha':
        data = bytearray(after)
        data[6:8] = struct.pack('<e', .5)
        (root / 'output_filter' / 'boundary_output.raw').write_bytes(data)
    elif mutation == 'nan':
        data = bytearray(after)
        data[:2] = struct.pack('<H', 0x7e00)
        (root / 'output_filter' / 'boundary_output.raw').write_bytes(data)
    elif mutation == 'split':
        events[-2]['submission_split_performed'] = False
        (root / 'session.jsonl').write_text('\n'.join(json.dumps(event) for event in events) + '\n')
    elif mutation == 'bypass':
        events[-3]['internal_hook_bypass'] = False
        (root / 'session.jsonl').write_text('\n'.join(json.dumps(event) for event in events) + '\n')
    else:
        metadata['trained_weights'] = True
        (root / 'output_filter' / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        analyze(root)
