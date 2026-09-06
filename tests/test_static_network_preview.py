import struct

import numpy as np
import pytest

from scripts.prepare_static_network_preview import panel


def test_labeled_panel_preserves_every_half_pixel():
    raw = struct.pack('<4e', 5.5, -0.25, 0.125, 0.5) * (640 * 360)
    packed = panel(raw, 'NETWORK OUTPUT')
    assert len(packed) == 644 * 384 * 8
    p = np.frombuffer(packed, '<u2').reshape(384, 644, 4)
    assert p[22:382, 2:642].tobytes() == raw
    assert np.isfinite(np.frombuffer(packed, '<f2')).all()
    assert panel(raw, 'ORIGINAL INPUT') != packed


def test_nonfinite_preview_rejected():
    raw = struct.pack('<e', float('nan')) + bytes(640 * 360 * 8 - 2)
    with pytest.raises(ValueError, match='nonfinite'):
        panel(raw, 'NETWORK OUTPUT')
