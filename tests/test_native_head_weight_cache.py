import torch
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_head_torch import RecoveredHead32


def test_tail_cache_preserves_rounding_and_invalidates_mutation_and_training():
    torch.manual_seed(24)
    head=RecoveredHead32(bytes(21808)).eval()
    head.block=torch.nn.Identity()
    head.block.inference_cache_enabled=False
    x=torch.randn(2,64,32,dtype=torch.float16)
    with torch.no_grad():
        head.tail.copy_(torch.randn_like(head.tail))
        expected=head.project(x)
        head.block.inference_cache_enabled=True
        assert torch.equal(expected.view(torch.int16),head.project(x).view(torch.int16))
        old=head._tail_cache[2]
        head.tail.add_(.125)
        changed=head.project(x)
        assert head._tail_cache[2] is not old
        head.block.inference_cache_enabled=False
        assert torch.equal(changed.view(torch.int16),head.project(x).view(torch.int16))
    head.train()
    assert head._tail_cache is None
