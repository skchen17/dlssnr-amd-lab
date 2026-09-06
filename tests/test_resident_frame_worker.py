import numpy as np
import pytest

from scripts.resident_frame_worker import tile_frame, validate_request, MAGIC, MAX_BYTES


class IdentityGraph:
    def __init__(self):
        self.calls = 0

    def infer(self, data, **kwargs):
        assert len(data) == 640 * 360 * 8
        self.calls += 1
        return data, {'frame_id': self.calls}


@pytest.mark.parametrize('width,height', [(1, 1), (640, 360), (641, 361), (2342, 1317), (3840, 2160), (8192, 1)])
def test_arbitrary_transport_shape_covers_every_pixel_without_resize(width, height):
    source = (np.arange(width * height * 4, dtype=np.uint32) % 1024).astype('<f2').tobytes()
    graph = IdentityGraph()
    output, info = tile_frame(graph, source, width, height)
    assert output == source
    assert graph.calls == ((width + 639) // 640) * ((height + 359) // 360)
    assert not info['resized'] and not info['equivalent_full_frame_attention']


def test_invalid_shapes_before_inference():
    g = IdentityGraph()
    for w, h, data in [(0, 1, b''), (1, 1, b'x'), (8193, 1, bytes(8193*8))]:
        with pytest.raises(ValueError):
            tile_frame(g, data, w, h)
    assert g.calls == 0


def test_deadline_does_not_publish_partial_frame():
    with pytest.raises(TimeoutError):
        tile_frame(IdentityGraph(), bytes(640*360*8), 640, 360, budget_seconds=-1)


def test_ipc_rejects_old_identity_wrong_owner_and_wrong_geometry():
    good = (MAGIC, 2, 1, 640, 360, 640*360*8, 2, 1234, 1)
    assert validate_request(good, 1) == (2, 640, 360, 640*360*8, 1234)
    for index, value in [(0, b'bad'), (1, 1), (2, 0), (3, 0), (5, MAX_BYTES+1), (6, 1), (7, 0), (8, 2)]:
        bad = list(good)
        bad[index] = value
        with pytest.raises(ValueError):
            validate_request(tuple(bad), 1)
