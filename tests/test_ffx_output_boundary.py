import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_output_boundary import compare_pair


@pytest.fixture
def pair(tmp_path):
    base, copied = tmp_path / 'base', tmp_path / 'copied'
    for directory, enabled in ((base, False), (copied, True)):
        directory.mkdir()
        manifest = dict(status='DISPATCH_READBACK_PASS', debug_errors=0, debug_warnings=0,
                        completed_fences=4, output_poisoned_nan=True,
                        requested_provider='4.1.1 *', requested_provider_id=17700776142811697153,
                        adapter_vendor=4098, adapter_device=1, render_size=[4, 2], output_size=[8, 4],
                        output_boundary_enabled=enabled, output_boundary_completed=4 if enabled else 0,
                        output_boundary_exact=True, output_roundtrip_enabled=False,
                        output_roundtrip_completed=0, output_roundtrip_exact=True)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        for i in range(4):
            frame = np.zeros((4, 8, 4), dtype='<f2')
            frame[..., 0] = np.linspace(.1, .8, 8)
            frame[..., 1] = .5;frame[..., 2] = .75 if i == 2 else .25;frame[..., 3] = 1
            (directory / f'output_{i}.raw').write_bytes(frame.tobytes())
            (directory / f'input_{i}.raw').write_bytes(frame[::2, ::2].tobytes())
            if enabled:
                (directory / f'boundary_{i}.raw').write_bytes(frame.tobytes())
    return base, copied


def test_exact_outputs_do_not_certify_game_or_nr(pair):
    report = compare_pair(*pair)
    assert report['status'] == 'OUTPUT_BOUNDARY_PROVIDER_PASS'
    assert report['on_off_output_equal']
    assert not report['game_frame_capture_verified'] and not report['dlss_nr_verified']


def test_exact_roundtrip_proves_writeback_not_replacement(pair):
    path = pair[1] / 'manifest.json'; data = json.loads(path.read_text())
    data.update(output_boundary_enabled=False, output_boundary_completed=0,
                output_roundtrip_enabled=True, output_roundtrip_completed=4,
                output_roundtrip_exact=True)
    path.write_text(json.dumps(data))
    report = compare_pair(*pair, roundtrip=True)
    assert report['status'] == 'OUTPUT_ROUNDTRIP_PROVIDER_PASS'
    assert report['write_back_performed'] and not report['replacement_pixels_supplied']


@pytest.mark.parametrize('field,value', [('output_boundary_enabled', False),
    ('output_boundary_completed', 3), ('output_boundary_exact', False),
    ('requested_provider_id', 1), ('debug_warnings', 1), ('adapter_vendor', 4318)])
def test_invalid_control_manifest_rejected(pair, field, value):
    path = pair[1] / 'manifest.json';data = json.loads(path.read_text())
    data[field] = value;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        compare_pair(*pair)


def test_captured_corruption_not_hidden_by_native_exact_flag(pair):
    path = pair[1] / 'boundary_0.raw';data = bytearray(path.read_bytes());data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match='output mismatch'):
        compare_pair(*pair)


def test_different_inputs_rejected_even_if_each_repeat_test_passes(pair):
    for i in (0, 1, 3):
        path = pair[1] / f'input_{i}.raw';data = bytearray(path.read_bytes());data[0] ^= 1;path.write_bytes(data)
    with pytest.raises(ValueError, match='input mismatch'):
        compare_pair(*pair)


def test_missing_boundary_file_rejected(pair):
    (pair[1] / 'boundary_2.raw').unlink()
    with pytest.raises(FileNotFoundError):
        compare_pair(*pair)
