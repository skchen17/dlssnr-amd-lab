import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_nvidia_execution_granularity import entry_stats
import pytest


def test_source_sites_are_scoped_to_entry_and_not_claimed_dynamic():
    text='''.visible .entry first() {
.reg .b32 %r<42>;
.shared .align 16 .b8 scratch[1024];
ld.global.b32 %r1, [%rd1];
@%p1 st.global.b32 [%rd2], %r1;
{ .reg .b128 v; mov.b128 v, %r1; st.global.L1::no_allocate.b128 [%rd3], v; }
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 %r1, %r2;
}
.visible .entry second() {
ld.global.b32 %r1, [%rd1];
}
'''
    result=entry_stats(text,'first')
    assert result['static_shared_bytes']==1024
    assert result['virtual_register_declarations']==[('.b32','42')]
    assert result['instruction_sites']['ld.global.b32']==1
    assert result['instruction_sites']['st.global.b32']==1
    assert result['instruction_sites']['st.global.L1::no_allocate.b128']==1
    assert result['line_start']==1
    with pytest.raises(ValueError):entry_stats(text,'absent')
