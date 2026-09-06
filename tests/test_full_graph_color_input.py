import json
import struct

import pytest

from scripts.run_full_graph_integrated import load_color_input, sha256
from scripts.analyze_full_graph_integrated import analyze
from scripts.validate_reconstructed_head_full_frame import validate
from scripts.analyze_full_graph_color import compare
from scripts.analyze_full_graph_color_controls import difference
from scripts.prepare_full_graph_game_color import prepare as prepare_crop
from scripts.prepare_full_graph_color_control import prepare as prepare_control
import numpy as np


def test_default_is_original_zero_input():
    raw, info = load_color_input(None)
    assert raw == bytes(640 * 360 * 8)
    assert info['external'] is False


def test_external_half_pixels_are_not_tonemapped(tmp_path):
    raw = struct.pack('<4e', -0.25, 0.5, 5.75, 1.0) * (640 * 360)
    path = tmp_path / 'hdr.raw'
    path.write_bytes(raw)
    loaded, info = load_color_input(path)
    assert loaded == raw
    assert info['sha256'] == sha256(raw)
    assert info['external'] is True


@pytest.mark.parametrize('raw', [b'bad', struct.pack('<e', float('nan')) + bytes(640 * 360 * 8 - 2),
                                 struct.pack('<e', float('inf')) + bytes(640 * 360 * 8 - 2)],
                         ids=['short', 'nan', 'inf'])
def test_reject_invalid_input_before_gpu(tmp_path, raw):
    path = tmp_path / 'invalid.raw'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_color_input(path)


def test_zero_reference_analyzers_reject_external_input(tmp_path):
    (tmp_path / 'execution.json').write_text(json.dumps({'color_input': {'external': True}}))
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    with pytest.raises(ValueError, match='zero-input'):
        analyze(plan, tmp_path)
    with pytest.raises(ValueError, match='zero-input'):
        validate(tmp_path, tmp_path / 'missing_reference', tmp_path / 'model', tmp_path / 'exe', tmp_path / 'out')


def test_hdr_clipping_is_not_counted_as_neural_residual():
    base = np.array([[[5.0, -0.5, 0.5, 1.0]]], dtype='f4')
    out = np.clip(base, 0, 1)
    metrics = compare(base, out)
    assert metrics['changed_rgb_components_vs_raw'] == 2
    assert metrics['changed_rgb_components_vs_clamped_base'] == 0
    assert metrics['mae_vs_clamped_base'] == 0


def test_crop_preserves_bytes_and_rejects_corrupt_capture(tmp_path):
    session = tmp_path / 'session'
    folder = session / 'output_network'
    folder.mkdir(parents=True)
    source = np.zeros((362, 642, 4), dtype='<f2')
    source[1, 1] = [5, -1, 0.125, 1]
    raw = source.tobytes()
    (folder / 'boundary_before.raw').write_bytes(raw)
    (folder / 'metadata.json').write_text(json.dumps(dict(width=642, height=362, format=10,
        raw_bytes=len(raw), fence_completed=True, game_frame_capture_verified=True)))
    audit = session / 'output_network_analysis.json'
    audit.write_text(json.dumps({'before_sha256': sha256(raw)}))
    result = prepare_crop(session, tmp_path / 'crop')
    assert result['crop_xywh'] == [1, 1, 640, 360]
    assert (tmp_path / 'crop/input_rgba16f.raw').read_bytes() == source[1:361, 1:641].tobytes()
    assert result['dlss5_quality_verified'] is False
    with pytest.raises(ValueError, match='outside'):
        prepare_crop(session, tmp_path / 'bad_crop', 99, 99)
    audit.write_text(json.dumps({'before_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='provenance'):
        prepare_crop(session, tmp_path / 'bad_capture')
    assert not (tmp_path / 'bad_capture').exists()


def test_control_changes_only_neural_color_and_preserves_base(tmp_path):
    parent = tmp_path / 'base.json'
    original = {'color_input': {'external': True, 'path': 'real.raw'},
                'post_texture': {'path': 'real.raw'}, 'slots': [{'ptx': 'kernel.ptx'}]}
    parent.write_text(json.dumps(original))
    result = prepare_control(parent, tmp_path / 'control')
    control = json.loads((tmp_path / 'control/plan.json').read_text())
    assert result['status'] == 'CONTROL_PREPARED'
    assert control['post_texture']['path'] == str((tmp_path / 'real.raw').resolve())
    assert control['slots'][0]['ptx'] == str((tmp_path / 'kernel.ptx').resolve())
    assert (tmp_path / 'control/zero_rgba16f.raw').read_bytes() == bytes(640 * 360 * 8)
    assert json.loads(parent.read_text()) == original
    with pytest.raises(FileExistsError):
        prepare_control(parent, tmp_path / 'control')


def test_byte_comparison_checks_sizes_and_differences():
    assert difference(b'abc', b'abc')['exact']
    assert difference(b'abc', b'abd')['mismatched_bytes'] == 1
    with pytest.raises(ValueError, match='size'):
        difference(b'a', b'ab')
