import ctypes as ct
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import resident_full_graph as resident


def test_pixels_keep_hdr_and_reject_wrong_size_nonfinite():
    raw = np.full((360, 640, 4), 5.5, dtype='<f2').tobytes()
    resident.validate_pixels(raw)
    for data in (b'bad', np.full((360, 640, 4), np.nan, dtype='<f2').tobytes()):
        with pytest.raises(ValueError):
            resident.validate_pixels(data)


def test_assets_hash_is_required(tmp_path):
    (tmp_path / 'data').write_bytes(b'abc')
    spec = {'path': 'data', 'bytes': 3, 'sha256': resident.replay.sha256(b'abc')}
    assert resident.checked_asset(tmp_path, spec) == b'abc'
    spec['bytes'] = 4
    with pytest.raises(ValueError, match='size/hash'):
        resident.checked_asset(tmp_path, spec)


@pytest.mark.parametrize('injection', [False, True])
def test_bad_graph_rejected_before_device(tmp_path, injection):
    plan = {'slots': [{'slot': i} for i in range(156 if injection else 155)],
            'diagnostic_injections': [{'after_slot': 1}] if injection else []}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='injections' if injection else '156-slot'):
        resident.load_assets(path)


def fake_graph(monkeypatch, event_status=0):
    calls = []
    g = resident.ResidentGraph.__new__(resident.ResidentGraph)
    g.owner = threading.get_ident()
    g.closed = g.failed = g.busy = False
    g.frame_id = 0
    g.plan = {'activation_arena_bytes': 16}
    g.arena, g.model, g.output = [ct.c_uint64(i) for i in (1, 2, 3)]
    g.arrays = [4, 5]
    g.events = [6, 7]
    g.launches = [(8, (1, 1, 1), (32, 1, 1), 0, b'', None)] * 156
    def check(op, code):
        assert code == 0
    def log(kind):
        def fn(*args):
            calls.append((kind, args))
            return 0
        return fn
    g.api = SimpleNamespace(check=check, cuMemsetD32=log('clear'),
                            cuLaunchKernel=log('launch'), cuCtxSynchronize=log('sync'))
    g._upload = lambda *args: calls.append(('upload', args))
    g.event_record = log('event')
    g.event_query = lambda e: event_status
    def elapsed(out, *events):
        out._obj.value = 12
        return 0
    g.event_elapsed = elapsed
    monkeypatch.setattr(resident.replay, 'd_to_h', lambda *args: bytes(resident.FRAME_BYTES))
    return g, calls


def test_frame_path_caches_launches_no_slot_sync_and_updates_both_textures(monkeypatch):
    g, calls = fake_graph(monkeypatch)
    data = bytes(resident.FRAME_BYTES)
    result, info = g.infer(data)
    assert result == data and info['gpu_graph_ms'] == 12
    assert sum(x[0] == 'launch' for x in calls) == 156
    assert not any(x[0] == 'sync' for x in calls)
    assert [x[1][0] for x in calls if x[0] == 'upload'] == [4, 5]
    assert [x[1][0].value for x in calls if x[0] == 'clear'] == [1, 3]
    assert info['frame_id'] == 1 and not g.busy


def test_invalid_input_does_not_poison_or_touch_gpu(monkeypatch):
    g, calls = fake_graph(monkeypatch)
    with pytest.raises(ValueError):
        g.infer(b'bad')
    assert not calls and not g.failed and not g.busy


def test_failed_deadline_poisoned_no_second_launch(monkeypatch):
    g, calls = fake_graph(monkeypatch, 600)
    with pytest.raises(TimeoutError):
        g.infer(bytes(resident.FRAME_BYTES), timeout_seconds=0.000001)
    assert g.failed and not g.busy
    count = len(calls)
    with pytest.raises(RuntimeError):
        g.infer(bytes(resident.FRAME_BYTES))
    assert len(calls) == count
    g.close()  # deliberately retains uncertain in-flight objects
    assert g.closed


def test_wrong_thread_and_closed_rejected(monkeypatch):
    g, calls = fake_graph(monkeypatch)
    g.owner = -1
    with pytest.raises(RuntimeError):
        g.infer(bytes(resident.FRAME_BYTES))
    with pytest.raises(RuntimeError):
        g.close()
    assert not calls
