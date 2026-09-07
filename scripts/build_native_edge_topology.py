"""Build CPU-only descriptors for the reset/single-color Pre and Output Head edges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def records(manifest):
    values={row['name']:row for row in manifest['records']}
    if len(values)!=len(manifest['records']):raise ValueError('duplicate weight record')
    return values


def offset(values,name,dtype,shape):
    row=values[name]
    if row.get('dtype')!=dtype or row.get('logical_shape')!=shape:
        raise ValueError(f'unexpected edge weight metadata: {name}')
    return row['offset']


def build(weight_manifest,model,width=1920,height=1080):
    if weight_manifest.get('target_arch')!='gfx1201' or \
            weight_manifest.get('status')!='DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED':
        raise ValueError('gfx1201 derived weight manifest required')
    if [width,height]!=[1920,1080]:raise ValueError('this topology gate is 1080p only')
    settings=model.get('settings')
    if settings!={'color_scale':.125,'conditioning':[0.,1.,1.,-1.,-1.]}:
        raise ValueError('unreviewed single-color conditioning')
    w=records(weight_manifest);pw=(width+127)//128*128;ph=(height+127)//128*128
    pre={
        'struct_size':160,'padded_width':pw,'padded_height':ph,
        'windows':(pw//8)*(ph//8),'color_scale':settings['color_scale'],
        'frame_seed_xor':0,'conditioning':settings['conditioning'],'reserved':0,
        'input_project_weight_offset':offset(w,'pre.input_project','fp16_le',[16,32]),
        'ffn_expand_weight_offset':offset(w,'pre.swin.block.ffn.expand','e4m3fn',[32,128]),
        'ffn_contract_weight_offset':offset(w,'pre.swin.block.ffn.contract','e4m3fn',[1,128,32]),
        'ffn_scale_weight_offset':offset(w,'pre.swin.block.ffn.residual_scale','fp16_le',[32]),
        'a_index_weight_offset':offset(w,'pre.swin.a_index','int32_le',[64,32]),
        'residual_index_weight_offset':offset(w,'pre.swin.residual_index','int32_le',[64,32]),
        'ffn_permutation_weight_offset':offset(w,'pre.swin.block.ffn.permutation','int32_le',[32]),
        'ffn_inverse_permutation_weight_offset':offset(w,'pre.swin.block.ffn.inverse_permutation','int32_le',[32]),
        'qkv_weight_offset':offset(w,'pre.swin.block.attention.qkv','e4m3fn',[32,96]),
        'qscale_weight_offset':offset(w,'pre.swin.block.attention.q_scale','fp16_le',[1]),
        'permutation_weight_offset':offset(w,'pre.swin.block.attention.permutation','int32_le',[32]),
        'position_bias_weight_offset':offset(w,'pre.swin.block.attention.position_bias','fp16_le',[1,64,64]),
        'project_weight_offset':offset(w,'pre.swin.block.attention.project','e4m3fn',[32,32]),
        'residual_scale_weight_offset':offset(w,'pre.swin.block.attention.attention_scale','fp16_le',[32]),
    }
    head={
        'struct_size':136,'padded_width':pw,'padded_height':ph,
        'windows':((pw+11)//8)*((ph+11)//8),
        'main_scale_weight_offset':offset(w,'head.main_scale','fp16_le',[32]),
        'skip_scale_weight_offset':offset(w,'head.skip_scale','fp16_le',[32]),
        'tail_weight_offset':offset(w,'head.tail','fp16_le',[32,4]),
        'ffn_expand_weight_offset':offset(w,'head.block.expand','e4m3fn',[4,32,32]),
        'ffn_contract_weight_offset':offset(w,'head.block.contract','e4m3fn',[4,32,32]),
        'ffn_scale_weight_offset':offset(w,'head.block.ffn_scale','fp16_le',[32]),
        'a_index_weight_offset':offset(w,'head.block.a_index','int32_le',[64,32]),
        'residual_index_weight_offset':offset(w,'head.block.residual_index','int32_le',[64,32]),
        'ffn_permutation_weight_offset':offset(w,'head.block.permutation','int32_le',[32]),
        'qkv_weight_offset':offset(w,'head.block.qkv','e4m3fn',[32,96]),
        'qscale_weight_offset':offset(w,'head.block.q_scale','fp16_le',[]),
        'permutation_weight_offset':offset(w,'head.block.permutation','int32_le',[32]),
        'position_bias_weight_offset':offset(w,'head.block.position_bias','fp16_le',[64,64]),
        'project_weight_offset':offset(w,'head.block.project','e4m3fn',[32,32]),
        'residual_scale_weight_offset':offset(w,'head.block.attention_scale','fp16_le',[32]),
    }
    return {'schema':1,'abi_version':3,'target_arch':'gfx1201','resolution':[width,height],
            'precision_profile':'approx_fp8',
            'status':'FULL_NATIVE_SINGLE_COLOR_EDGES_NOT_RUNTIME_ACCEPTED',
            'temporal_contract_verified':False,'pre':pre,'head':head,
            'graph_kernel_nodes_if_recorded':19,
            'note':'dynamic ABI-v3 color/residual pointers; reset/no-history route only'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=build(json.loads(a.weights.read_text(encoding='utf-8')),
                 json.loads(a.model.read_text(encoding='utf-8')))
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'pre_windows':result['pre']['windows'],
                      'head_windows':result['head']['windows']},indent=2))


if __name__=='__main__':main()
