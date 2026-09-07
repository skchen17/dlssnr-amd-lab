"""Summarize an annotated NRPlan HIP Graph without using RGP counters.

The census graph contains 72 one-thread `nrplan_stage_marker` kernels around
the 71 recovered blocks.  Markers are excluded from deployment counts.  HIP's
Windows kernel-name query currently returns unresolved handles for captured
nodes, so names are read from the official Graph DOT dump and paired with the
fixed launch/ISA attributes returned by the C++ runtime.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter,defaultdict
from pathlib import Path


KERNEL_RE=re.compile(r'\[style="bold"shape="octagon"label="\d+\n([^\n]+)\n"\];')
MARKER='nrplan_stage_marker'
AUTHORED=('quantize_kernel','cubic_quantize_kernel','wide_group_ffn_wmma','c512_group_ffn_wmma',
          'head_ffn_wmma','head_attention_window_fused','head_tail_exact','head_input_kernel',
          'head_compose_kernel','head_qkv_project_wmma','head_softmax_e4',
          'c32_ffn_wmma','c32_qkv_norm','c32_qk_wmma','c32_pv_wmma','c32_project_wmma',
          'wide_group_ffn_fp8a','wide_group_mix_fp8a',
          'pre_project_pack_kernel','pool_permute_skip_kernel',
          'quantize_pack_kernel','unpack_permute_kernel','expand_skip_pack_kernel')


def module_for(block,name):
    if any(x in name for x in ('pool_permute_skip_kernel','quantize_pack_kernel',
                                'unpack_permute_kernel','expand_skip_pack_kernel')):
        return 'transition'
    if block==0:return 'Pre'
    if 1<=block<=4 or 66<=block<=69:return 'C32'
    if 5<=block<=8 or 62<=block<=65:return 'C64'
    if 9<=block<=14 or 56<=block<=61:return 'C128'
    if 15<=block<=22 or 48<=block<=55:return 'C256'
    if 23<=block<=30 or 40<=block<=47:return 'C512'
    if 31<=block<=38:return 'ViT'
    if block==39:return 'transition'
    if block==70:return 'Head'
    raise ValueError(f'unassigned block {block}')


def category(name):
    low=name.lower()
    if 'attention' in low or 'softmax' in low:return 'attention'
    if 'wmma' in low or 'gemm' in low or 'rocblas' in low or ' matmul' in low:return 'GEMM'
    if 'quantize' in low or 'e4m3' in low:return 'quantize'
    if 'float16_copy' in low or 'float16tofloat32' in low or ('cast_kernel' in low and 'nocast' not in low):return 'cast'
    if any(x in low for x in ('pack_kernel','unpack','index_','gather','scatter','catarray','transpose')):return 'layout'
    if any(x in low for x in ('reduce_kernel','rsqrt','normalization','norm_kernel')):return 'norm/reduction'
    if any(x in low for x in ('clamp','relu','gelu','activation','cubic_')):return 'activation'
    if any(x in low for x in ('copy_kernel','memcpy')):return 'copy'
    if any(x in low for x in ('elementwise','functor_add','mulfunctor','remainder','div_floor',
                               'bitwise','compare_scalar','where_kernel','fillfunctor','arange')):return 'pointwise'
    return 'other'


def family(name):
    for token,label in (
        ('nrplan_stage_marker','census_marker'),('wide_group_ffn_wmma','grouped_ffn_wmma'),
        ('wide_group_ffn_fp8a','grouped_ffn_fp8a_producer'),
        ('wide_group_mix_fp8a','grouped_ffn_fp8a_consumer'),
        ('c512_group_ffn_wmma','grouped_ffn_wmma'),('head_ffn_wmma','head_ffn_wmma'),
        ('c32_ffn_wmma','c32_ffn_wmma'),('c32_qkv_norm','c32_qkv_norm'),
        ('c32_qk_wmma','c32_qk_wmma'),('c32_pv_wmma','c32_pv_wmma'),
        ('c32_project_wmma','c32_project_wmma'),('head_qkv_project_wmma','qkv_project_wmma'),
        ('head_softmax_e4','softmax_e4'),
        ('head_attention_window_fused','head_attention'),('head_tail_exact','head_tail'),
        ('pre_project_pack_kernel','pre_project_pack'),('pool_permute_skip_kernel','encoder_transition'),
        ('quantize_pack_kernel','encoder_transition'),('unpack_permute_kernel','decoder_transition'),
        ('expand_skip_pack_kernel','decoder_transition'),('cubic_quantize_kernel','cubic_quantize'),
        ('quantize_kernel','quantize_e4'),('softmax','aten_softmax'),('rsqrt','aten_rsqrt'),
        ('reduce_kernel','aten_reduce'),('direct_copy_kernel','aten_copy'),
        ('float16tofloat32_copy','aten_cast_fp16_fp32'),('float16_copy_kernel','aten_cast_fp32_fp16'),
        ('index_elementwise_kernel','aten_index'),('vectorized_gather_kernel','aten_gather'),
        ('catarraybatchedcopy','aten_cat'),('arange_cuda','aten_arange'),
        ('div_floor','aten_floor_divide'),('remainder_kernel','aten_remainder'),
        ('bitwise','aten_bitwise'),('clamp','aten_clamp'),('fillfunctor','aten_fill'),
        ('add','aten_add'),('mul','aten_mul'),('elementwise','aten_elementwise')):
        if token in name.lower():return label
    return 'other'


def implementation(name):
    if any(x in name for x in AUTHORED):return 'project_HIP'
    if name.startswith('_ZN2at') or 'rocprim' in name.lower() or 'rocblas' in name.lower():return 'PyTorch_or_library'
    return 'other_library_or_unresolved'


def dtype_signature(name):
    low=name.lower()
    if 'quantize' in low or 'e4m3' in low:return 'FP16->E4M3'
    if 'float16tofloat32' in low:return 'FP16->FP32'
    if 'float16_copy' in low:return 'FP32->FP16'
    if 'df16' in low or 'half' in low or '__half' in low:return 'FP16_or_mixed'
    if any(x in low for x in ('bitwise','arange','compare','remainder','div_floor')):return 'integer_or_metadata'
    return 'not_encoded_in_symbol'


def feature_bytes(block,width,height):
    pw=(width+127)//128*128;ph=(height+127)//128*128
    if block==0 or 1<=block<=4 or 66<=block<=70:w,h,c=pw//2,ph//2,32
    elif 5<=block<=8 or 62<=block<=65:w,h,c=pw//4,ph//4,64
    elif 9<=block<=14 or 56<=block<=61:w,h,c=pw//8,ph//8,128
    elif 15<=block<=22 or 48<=block<=55:w,h,c=pw//16,ph//16,256
    elif 23<=block<=30 or 39<=block<=47:w,h,c=pw//32,ph//32,512
    else:
        w=((pw//64)+3)//4*4;h=((ph//64)+3)//4*4;c=1024
    return w*h*c*2


def parse_names(dot):
    return KERNEL_RE.findall(Path(dot).read_text(encoding='utf-8'))


def summarize(raw_path,dot_path,width,height,stage_times=None):
    raw=json.loads(Path(raw_path).read_text(encoding='utf-8'))['nodes'];names=parse_names(dot_path)
    if len(raw)!=len(names):raise ValueError(f'DOT/raw kernel count mismatch: {len(names)} != {len(raw)}')
    boundary=-1;nodes=[]
    for base,name in zip(raw,names):
        if MARKER in name:
            boundary+=1
            continue
        if not 0<=boundary<71:raise ValueError('kernel is outside the 72 stage markers')
        item=dict(base);item['name']=name;item['block']=boundary
        item['module']=module_for(boundary,name);item['category']=category(name);item['family']=family(name)
        item['implementation']=implementation(name);item['dtype_signature']=dtype_signature(name)
        item['stage_resident_fp16_bytes']=feature_bytes(boundary,width,height)
        nodes.append(item)
    if boundary!=71:raise ValueError(f'expected 72 stage markers, found {boundary+1}')
    family_rows=[]
    groups=defaultdict(list)
    for node in nodes:groups[(node['module'],node['family'],node['category'],node['implementation'])].append(node)
    for key,values in groups.items():
        # This is an activation-traffic proxy, not a hardware DRAM counter. It
        # ranks primitive multiplicity while keeping the evidence boundary clear.
        proxy=sum(2*v['stage_resident_fp16_bytes'] for v in values)
        family_rows.append({'module':key[0],'family':key[1],'category':key[2],'implementation':key[3],
                            'count_per_frame':len(values),'activation_io_proxy_bytes':proxy,
                            'max_stage_resident_fp16_bytes':max(v['stage_resident_fp16_bytes'] for v in values),
                            'max_registers_per_thread':max(v['registers_per_thread'] for v in values),
                            'max_lds_bytes':max(v['static_shared_bytes']+v['dynamic_shared_bytes'] for v in values),
                            'max_local_bytes_per_thread':max(v['local_bytes_per_thread'] for v in values),
                            'cumulative_kernel_busy_ms':None})
    family_rows.sort(key=lambda x:(x['count_per_frame'],x['activation_io_proxy_bytes']),reverse=True)
    module_counts=Counter(n['module'] for n in nodes)
    module_times={}
    if stage_times:
        timing=json.loads(Path(stage_times).read_text(encoding='utf-8'))
        for row in timing['stages']:
            mod=module_for(row['block'],'')
            module_times[mod]=module_times.get(mod,0)+row['gpu_stream_interval_ms']
    for row in family_rows:
        envelope=module_times.get(row['module'])
        row['module_stream_interval_ms']=envelope
        row['count_weighted_module_interval_proxy_ms']=(
            envelope*row['count_per_frame']/module_counts[row['module']]
            if envelope is not None else None)
    kernel_counts=Counter(n['name'] for n in nodes)
    return {'schema':1,'resolution':[width,height],'graph_kernel_nodes_including_markers':len(raw),
            'census_marker_nodes':len(raw)-len(nodes),'deploy_kernel_nodes':len(nodes),
            'all_nodes_stage_attributed':len(nodes)==sum(module_counts.values()),
            'name_source':'hipGraphDebugDotPrint; hipKernelGetName was unresolved on ROCm Windows',
            'time_scope':'exact per-family kernel busy time unavailable while RGP safety halt is active; module_stream_interval_ms is a non-RGP HIP event envelope and count_weighted_module_interval_proxy_ms is only a prioritization proxy',
            'traffic_scope':'activation_io_proxy_bytes is 2x stage-resident FP16 bytes per primitive, not a DRAM/L2 hardware counter',
            'module_kernel_counts':dict(module_counts),'module_stream_interval_ms':module_times or None,
            'implementation_counts':dict(Counter(n['implementation'] for n in nodes)),
            'category_counts':dict(Counter(n['category'] for n in nodes)),
            'family_ranking':family_rows,
            'kernel_name_ranking':[{'name':name,'count_per_frame':count} for name,count in kernel_counts.most_common()],
            'nodes':nodes}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--dot',type=Path,required=True);p.add_argument('--width',type=int,required=True)
    p.add_argument('--height',type=int,required=True);p.add_argument('--stage-times',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.raw,a.dot,a.width,a.height,a.stage_times)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('deploy_kernel_nodes','census_marker_nodes','module_kernel_counts','category_counts')},indent=2))


if __name__=='__main__':main()
