import json
from scripts.assess_nr_fsr_inputs import assess, assess_frame


def fixture(tmp_path, frame=0, **changes):
    directory = tmp_path / str(frame)
    directory.mkdir()
    data = dict(frame=frame, game_frame=True, fence_completed=True,
                render_size=[2, 2], effective_upscale_size=[4, 4], jitter=[0.1, -0.2],
                motion_scale=[2, 2], reset=frame == 0, pre_exposure=1,
                frame_time_ms=16.67, camera_near=0.1, camera_far=100,
                camera_fov_y=1, view_to_meters=1, context_create_flags=32,
                resources=[])
    for role in ('color', 'depth', 'motion'):
        (directory / (role + '.raw')).write_bytes(b'\x00' * 16)
        data['resources'].append(dict(role=role, width=2, height=2, row_bytes=8, raw_bytes=16, dxgi_format=41))
    data.update(changes)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(data), encoding='utf8')
    return path


def test_complete_is_not_semantic_or_quality_proof(tmp_path):
    fixture(tmp_path)
    fixture(tmp_path, 1)
    report = assess(tmp_path, 2)
    assert report['sequences'][0]['candidate_for_semantic_review']
    assert not report['quality_ready']  # Deliberately synthetic test bytes cannot prove quality.


def test_synthetic_capture_rejected(tmp_path):
    assert 'not_attested_real_game_frame' in assess_frame(fixture(tmp_path, game_frame=False))['errors']


def test_missing_motion_is_not_filled(tmp_path):
    path = fixture(tmp_path)
    (path.parent / 'motion.raw').unlink()
    assert 'invalid_file_motion' in assess_frame(path)['errors']


def test_gap_and_reset_every_frame_rejected(tmp_path):
    fixture(tmp_path)
    fixture(tmp_path, 2, reset=True)
    report = assess(tmp_path, 2)
    assert not report['sequences'][0]['candidate_for_semantic_review']


def test_exposure_required_without_auto(tmp_path):
    assert 'missing_exposure' in assess_frame(fixture(tmp_path, context_create_flags=0))['errors']


def test_bad_metadata(tmp_path):
    errors = assess_frame(fixture(tmp_path, pre_exposure=float('nan'), motion_scale=[], resources=[]))['errors']
    assert 'invalid_pre_exposure' in errors
    assert 'invalid_motion_scale' in errors
    assert 'missing_color' in errors
