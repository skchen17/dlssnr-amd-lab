import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_graph_replay import CapturedTensorRunner, validate_input


def test_shape_change_requires_reprepare_never_silent_resize():
    value = torch.ones(1, 17, 32).half()
    validate_input(value.clone(), value)
    with pytest.raises(ValueError):
        validate_input(torch.ones(1, 18, 32).half(), value)
    with pytest.raises(ValueError):
        validate_input(value.float(), value)


def test_cpu_deployment_rejected_before_cuda_initialization():
    with pytest.raises(ValueError):
        CapturedTensorRunner(lambda x: x, torch.zeros(1, 32).half(), precision='native_fp16', wait=lambda: None)
