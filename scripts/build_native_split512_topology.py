"""Build CPU-only NRPlan descriptors for the sixteen C512 split blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RECORDS = tuple(range(23, 31)) + tuple(range(40, 48))


def records_by_name(manifest: dict) -> dict[str, dict]:
    result = {}
    for row in manifest['records']:
        if row['name'] in result:
            raise ValueError('duplicate cached tensor name')
        result[row['name']] = row
    return result


def build(weight_manifest: dict, arena: dict, window_origins: dict) -> dict:
    if weight_manifest.get('target_arch') != 'gfx1201' or \
            weight_manifest.get('status') != 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED':
        raise ValueError('unexpected weight cache')
    if arena.get('abi_version') != 3 or arena.get('precision_profile') != 'approx_fp8' or \
            arena.get('resolution') != [1920, 1080]:
        raise ValueError('1080p approximate arena required')
    weights = records_by_name(weight_manifest)
    regions = {row['name']: row for row in arena['regions']}
    required = {'c512.ping', 'operator.fp8_grouped', 'operator.fp8_post',
                'operator.fp8_q', 'operator.fp8_k', 'operator.fp8_v',
                'operator.fp8_value', 'operator.fp16_workspace'}
    if not required <= regions.keys():
        raise ValueError('arena lacks C512 resident workspace')
    width, height = 60, 36
    result = []
    for number in RECORDS:
        family = 'encoder512' if number < 31 else 'decoder512'
        prefix = f'{family}.{number}'
        ox, oy = window_origins[str(number)]
        windows = ((width - ox + 7) // 8) * ((height - oy + 7) // 8)
        elements = windows * 64 * 512
        fp16 = regions['operator.fp16_workspace']
        if any(regions[name]['bytes'] < elements for name in
               ('operator.fp8_grouped', 'operator.fp8_post', 'operator.fp8_q',
                'operator.fp8_k', 'operator.fp8_v', 'operator.fp8_value')) or \
                fp16['bytes'] < elements * 4:
            raise ValueError('C512 shared operator workspace too small')
        names = {
            'preproject_weight_offset': prefix + '.block.ffwd.preproject',
            'expand_weight_offset': prefix + '.block.ffwd.expand',
            'contract_weight_offset': prefix + '.block.ffwd.contract',
            'ffn_project_weight_offset': prefix + '.block.ffwd_projection.weight',
            'ffn_scale_weight_offset': prefix + '.block.ffwd_projection.residual_scale',
            'a_index_weight_offset': prefix + '.a_index',
            'residual_index_weight_offset': prefix + '.residual_index',
            'perm64_weight_offset': prefix + '.block.ffwd.perm64',
            'perm256_weight_offset': prefix + '.block.ffwd.perm256',
            'ffn_permutation_weight_offset': prefix + '.block.ffwd_projection.permutation',
            'qkv_weight_offset': prefix + '.block.attention.qkv',
            'qscale_weight_offset': prefix + '.block.attention.q_scale',
            'attention_permutation_weight_offset': prefix + '.block.attention.permutation',
            'position_bias_weight_offset': prefix + '.block.attention.position_bias',
            'attention_project_weight_offset': prefix + '.block.attention_projection.weight',
            'attention_scale_weight_offset': prefix + '.block.attention_projection.residual_scale',
        }
        expected = {
            'preproject_weight_offset': ('e4m3fn', [512, 512]),
            'expand_weight_offset': ('e4m3fn', [8, 64, 256]),
            'contract_weight_offset': ('e4m3fn', [8, 256, 64]),
            'ffn_project_weight_offset': ('e4m3fn', [512, 512]),
            'ffn_scale_weight_offset': ('fp16_le', [512]),
            'a_index_weight_offset': ('int32_le', [64, 512]),
            'residual_index_weight_offset': ('int32_le', [64, 512]),
            'perm64_weight_offset': ('int32_le', [64]),
            'perm256_weight_offset': ('int32_le', [256]),
            'ffn_permutation_weight_offset': ('int32_le', [512]),
            'qkv_weight_offset': ('e4m3fn', [512, 1536]),
            'qscale_weight_offset': ('fp16_le', [16]),
            'attention_permutation_weight_offset': ('int32_le', [512]),
            'position_bias_weight_offset': ('fp16_le', [16, 64, 64]),
            'attention_project_weight_offset': ('e4m3fn', [512, 512]),
            'attention_scale_weight_offset': ('fp16_le', [512]),
        }
        offsets = {}
        for field, name in names.items():
            row = weights.get(name)
            if not row or (row['dtype'], row['logical_shape']) != expected[field]:
                raise ValueError(f'invalid C512 cache tensor: {name}')
            if field == 'qkv_weight_offset' and \
                    row['storage_layout'] != 'column_major_transposed_storage':
                raise ValueError('C512 QKV must use native column-major cache')
            offsets[field] = row['offset']
        result.append({
            'name': prefix + '.block', 'record_number': number, 'struct_size': 248,
            'windows': windows, 'feature_width': width, 'feature_height': height,
            'origin_x': ox, 'origin_y': oy,
            'raw_resident_offset': regions['c512.ping']['offset'],
            'projected_offset': regions['operator.fp8_grouped']['offset'],
            'grouped_offset': regions['operator.fp8_post']['offset'],
            'post_fp16_offset': fp16['offset'],
            'post_resident_offset': regions['operator.fp8_grouped']['offset'],
            'q_offset': regions['operator.fp8_q']['offset'],
            'k_offset': regions['operator.fp8_k']['offset'],
            'v_offset': regions['operator.fp8_v']['offset'],
            'value_offset': regions['operator.fp8_value']['offset'],
            'output_fp16_offset': fp16['offset'] + elements * 2,
            'next_resident_offset': regions['c512.ping']['offset'],
            **offsets,
        })
    return {
        'schema': 1, 'abi_version': 3, 'target_arch': 'gfx1201',
        'resolution': [1920, 1080], 'precision_profile': 'approx_fp8',
        'status': 'PARTIAL_NATIVE_C512_SPLIT_NOT_RUNTIME_ACCEPTED',
        'complete_native_topology': False, 'block_count': len(result),
        'record_numbers': list(RECORDS),
        'graph_kernel_nodes_if_recorded': len(result) * 7,
        'blocks': result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--model', type=Path,
                        default=Path('local_models/native_single_color_v1/model.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = build(json.loads(args.weights.read_text(encoding='utf-8')),
                  json.loads(args.arena.read_text(encoding='utf-8')),
                  json.loads(args.model.read_text(encoding='utf-8'))['window_origins'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps({key: value[key] for key in
                      ('status', 'block_count', 'graph_kernel_nodes_if_recorded')}, indent=2))


if __name__ == '__main__':
    main()
