import sys
from pathlib import Path
import pytest
pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_whole_frame import SingleColorWholeFrame,feature_geometry


def test_padding_does_not_resize_or_stitch_independent_images():
    assert feature_geometry(640,360,1048576)==(640,384)
    assert feature_geometry(641,361,1048576)==(768,384)
    assert feature_geometry(2342,1382,4000000)==(2432,1408)


def test_unreviewed_4k_budget_fails_instead_of_downscaling():
    with pytest.raises(ValueError,match='no silent downscale'):
        feature_geometry(3840,2160,1048576)
    assert feature_geometry(3840,2160,9000000)==(3840,2176)
    for dims in [(0,360,1048576),(640,-1,1048576),(640,360,0)]:
        with pytest.raises(ValueError):
            feature_geometry(*dims)


def test_single_color_candidate_does_not_claim_temporal_hdr_or_game():
    assert SingleColorWholeFrame.single_color_tensor_chain_connected
    assert not SingleColorWholeFrame.native_graph_complete
    assert not SingleColorWholeFrame.hdr_supported
    assert not SingleColorWholeFrame.game_runtime_ready
