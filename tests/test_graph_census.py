import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_graph_census import category,dtype_signature,module_for,summarize


def test_block_to_module_and_transition_override():
    assert module_for(0,'x')=='Pre'
    assert module_for(4,'x')=='C32'
    assert module_for(31,'x')=='ViT'
    assert module_for(70,'x')=='Head'
    assert module_for(4,'_ZL24pool_permute_skip_kernel')=='transition'


def test_kernel_classification_is_conservative():
    assert category('_Z19wide_group_ffn_wmmaILi128E')=='GEMM'
    assert category('_ZN2at_softmax_warp_forward')=='attention'
    assert dtype_signature('_ZL15quantize_kernelPKtPty')=='FP16->E4M3'
    assert dtype_signature('opaque')=='not_encoded_in_symbol'


def test_markers_are_removed_and_stage_attribution_is_exact(tmp_path):
    names=['nrplan_stage_marker']
    names+=['_ZN2at_elementwise_add']
    names+=['nrplan_stage_marker']*70
    names+=['_ZL19head_compose_kernel']
    names+=['nrplan_stage_marker']
    dot='digraph dot {\n'+''.join(f'"graph_0_node_{i}"[style="bold"shape="octagon"label="{i}\n{name}\n"];\n' for i,name in enumerate(names))+'}\n'
    raw={'nodes':[{'grid':[1,1,1],'block':[64,1,1],'registers_per_thread':4,
                         'static_shared_bytes':0,'dynamic_shared_bytes':0,'local_bytes_per_thread':0}
                        for _ in names]}
    rp=tmp_path/'raw.json';dp=tmp_path/'graph.dot';rp.write_text(json.dumps(raw));dp.write_text(dot)
    result=summarize(rp,dp,1920,1080)
    assert result['census_marker_nodes']==72 and result['deploy_kernel_nodes']==2
    assert [x['block'] for x in result['nodes']]==[0,70]
    assert result['module_kernel_counts']=={'Pre':1,'Head':1}
