"""Build C++ NRPlan descriptor records for native wide-stage Swin blocks.

This is a CPU-only topology compiler. It proves that every C64/C128/C256
FFN/attention block has a weight/arena binding, but it does not claim that the
missing stage transitions or the complete 71-block graph have been migrated.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SUPPORTED = (32, 64, 128, 256)


def _records(manifest: dict) -> dict[str, dict]:
    result = {}
    for record in manifest['records']:
        if record['name'] in result:
            raise ValueError('duplicate weight name')
        result[record['name']] = record
    return result


def build(weight_manifest: dict, arena: dict, window_origins: dict) -> dict:
    if weight_manifest.get('target_arch') != 'gfx1201' or \
            weight_manifest.get('status') != 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED':
        raise ValueError('unexpected weight cache provenance/status')
    if arena.get('abi_version') != 3 or arena.get('precision_profile') != 'approx_fp8' or \
            arena.get('resolution') != [1920, 1080]:
        raise ValueError('1080p approximate ABI-v3 arena required')
    weights = _records(weight_manifest)
    regions = {row['name']: row for row in arena['regions']}
    required_regions = {'operator.fp16_workspace', 'operator.qkv_projection_fp16',
                        'operator.fp8_grouped',
                        'operator.fp8_post',
                        'operator.fp8_q', 'operator.fp8_k',
                        'operator.fp8_v', 'operator.fp8_value'}
    required_regions |= {f'c{channel}.{side}' for channel in SUPPORTED
                         for side in ('ping', 'pong')}
    if not required_regions <= regions.keys():
        raise ValueError('arena lacks resident stage slots')
    geometry = {}
    for channel in SUPPORTED:
        padded_width, padded_height = 1920, 1152
        geometry[channel] = (padded_width // 2 // (channel // 32),
                             padded_height // 2 // (channel // 32))
    qkv_names = [name for name, row in weights.items()
                 if name.endswith('attention.qkv') and
                 name.startswith(('encoder.', 'decoder.blocks.')) and
                 row.get('logical_shape', [None])[0] in SUPPORTED]
    blocks = []
    family_ordinals = {channel: 0 for channel in SUPPORTED}
    for name in qkv_names:
        qkv = weights[name]
        channel = qkv['logical_shape'][0]
        number_match = re.search(r'(?:encoder|decoder\.blocks)\.?([0-9]+)', name)
        if not number_match or number_match.group(1) not in window_origins:
            raise ValueError(f'missing recovered window origin: {name}')
        record_number = int(number_match.group(1))
        origin_x, origin_y = window_origins[number_match.group(1)]
        if origin_x not in (0, -4) or origin_y not in (0, -4):
            raise ValueError(f'invalid recovered window origin: {name}')
        feature_width, feature_height = geometry[channel]
        block_windows = ((feature_width-origin_x+7)//8)*((feature_height-origin_y+7)//8)
        if qkv.get('dtype') != 'e4m3fn' or \
                qkv.get('storage_layout') != 'column_major_transposed_storage' or \
                qkv.get('logical_shape') != [channel, 3 * channel]:
            raise ValueError(f'invalid QKV cache record: {name}')
        prefix = name[:-3]
        siblings = {
            'project': weights[prefix + 'project'],
            'qscale': weights[prefix + 'q_scale'],
            'permutation': weights[prefix + 'permutation'],
            'bias': weights[prefix + 'position_bias'],
            'scale': weights[prefix + 'attention_scale'],
        }
        if siblings['project']['dtype'] != 'e4m3fn' or \
                siblings['project']['logical_shape'] != [channel, channel] or \
                siblings['qscale']['logical_shape'] != [channel // 32] or \
                siblings['permutation']['logical_shape'] != [channel] or \
                siblings['bias']['logical_shape'] != [channel // 32, 64, 64] or \
                siblings['scale']['logical_shape'] != [channel]:
            raise ValueError(f'invalid attention siblings: {name}')
        owner = name[:-len('.block.attention.qkv')]
        ffn_prefix = owner + '.block.ffn.'
        ffn = {
            'expand': weights[ffn_prefix + 'expand'],
            'contract': weights[ffn_prefix + 'contract'],
            'mix': weights.get(ffn_prefix + 'mix'),
            'scale': weights[ffn_prefix + 'residual_scale'],
            'permutation': weights[ffn_prefix + 'permutation'],
            'inverse': weights[ffn_prefix + 'inverse_permutation'],
            'ai': weights[owner + '.a_index'],
            'ri': weights[owner + '.residual_index'],
        }
        if ffn['expand']['dtype'] != 'e4m3fn' or \
                ffn['expand']['logical_shape'] != [channel, 4 * channel] or \
                ffn['contract']['dtype'] != 'e4m3fn' or \
                ffn['contract']['logical_shape'] != [channel // 32, 128, 32] or \
                (channel > 32 and (ffn['mix'] is None or
                 ffn['mix']['dtype'] != 'e4m3fn' or
                 ffn['mix']['logical_shape'] != [channel, channel])) or \
                (channel == 32 and ffn['mix'] is not None) or \
                ffn['scale']['logical_shape'] != [channel] or \
                ffn['permutation']['logical_shape'] != [32] or \
                ffn['inverse']['logical_shape'] != [32] or \
                ffn['ai']['logical_shape'] != [64, channel] or \
                ffn['ri']['logical_shape'] != [64, channel]:
            raise ValueError(f'invalid FFN siblings: {name}')
        stage_bytes = arena['stage_feature_bytes'][f'c{channel}']
        window_bytes = block_windows * 64 * channel
        fp16 = regions['operator.fp16_workspace']
        qkv_projection = regions['operator.qkv_projection_fp16']
        if any(regions[f'operator.fp8_{part}']['bytes'] < window_bytes
               for part in ('grouped', 'post', 'q', 'k', 'v', 'value')) or \
                fp16['bytes'] < 6 * window_bytes or \
                qkv_projection['bytes'] < 6 * window_bytes:
            # FP16 stage is twice resident bytes, and two FP16 stage views are
            # needed: post residual + projection output.
            raise ValueError('shared operator workspace is too small')
        blocks.append({
            'name': name[:-len('.attention.qkv')],
            'record_number': record_number,
            'struct_size': 240,
            'channels': channel,
            'windows': block_windows,
            'feature_width': feature_width,
            'feature_height': feature_height,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'raw_resident_offset': regions[f'c{channel}.ping']['offset'],
            'grouped_offset': regions['operator.fp8_grouped']['offset'],
            'seed_fp16_offset': fp16['offset'],
            'post_fp16_offset': fp16['offset'] + 2 * window_bytes,
            'post_resident_offset': regions['operator.fp8_post']['offset'],
            'qkv_projection_fp16_offset': qkv_projection['offset'],
            'q_offset': regions['operator.fp8_q']['offset'],
            'k_offset': regions['operator.fp8_k']['offset'],
            'v_offset': regions['operator.fp8_v']['offset'],
            'value_offset': regions['operator.fp8_value']['offset'],
            'output_fp16_offset': fp16['offset'] + 4 * window_bytes,
            'next_resident_offset': regions[f'c{channel}.ping']['offset'],
            'ffn_expand_weight_offset': ffn['expand']['offset'],
            'ffn_contract_weight_offset': ffn['contract']['offset'],
            'ffn_mix_weight_offset': ffn['mix']['offset'] if ffn['mix'] else 0,
            'ffn_scale_weight_offset': ffn['scale']['offset'],
            'a_index_weight_offset': ffn['ai']['offset'],
            'residual_index_weight_offset': ffn['ri']['offset'],
            'ffn_permutation_weight_offset': ffn['permutation']['offset'],
            'ffn_inverse_permutation_weight_offset': ffn['inverse']['offset'],
            'qkv_weight_offset': qkv['offset'],
            'qscale_weight_offset': siblings['qscale']['offset'],
            'permutation_weight_offset': siblings['permutation']['offset'],
            'position_bias_weight_offset': siblings['bias']['offset'],
            'project_weight_offset': siblings['project']['offset'],
            'residual_scale_weight_offset': siblings['scale']['offset'],
        })
        family_ordinals[channel] += 1
    expected = {32: 8, 64: 8, 128: 12, 256: 16}
    if family_ordinals != expected:
        raise ValueError(f'wide block coverage mismatch: {family_ordinals}')
    blocks.sort(key=lambda row: row['record_number'])
    record_numbers = [row['record_number'] for row in blocks]
    if len(set(record_numbers)) != len(record_numbers):
        raise ValueError('duplicate native wide-stage record number')
    return {
        'schema': 1,
        'abi_version': 3,
        'target_arch': 'gfx1201',
        'resolution': [1920, 1080],
        'precision_profile': 'approx_fp8',
        'status': 'PARTIAL_NATIVE_SWIN32_256_BLOCKS_NOT_RUNTIME_ACCEPTED',
        'complete_native_topology': False,
        'block_count': len(blocks),
        'family_counts': {f'c{key}': value for key, value in family_ordinals.items()},
        'graph_kernel_nodes_if_recorded': sum(6 if row['channels'] == 32 else 7
                                              for row in blocks),
        'blocks': blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--model', type=Path,
                        default=Path('local_models/native_single_color_v1/model.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.weights.read_text(encoding='utf-8')),
                   json.loads(args.arena.read_text(encoding='utf-8')),
                   json.loads(args.model.read_text(encoding='utf-8'))['window_origins'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ('status', 'block_count',
                                                   'graph_kernel_nodes_if_recorded')}, indent=2))


if __name__ == '__main__':
    main()
