"""Build CPU-only NRPlan descriptors for the eight resident-FP8 ViT blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _records(manifest: dict) -> dict[str, dict]:
    result = {}
    for row in manifest['records']:
        if row['name'] in result:
            raise ValueError('duplicate weight name')
        result[row['name']] = row
    return result


def build(weight_manifest: dict, arena: dict) -> dict:
    if weight_manifest.get('target_arch') != 'gfx1201' or \
            weight_manifest.get('status') != 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED':
        raise ValueError('unexpected weight cache provenance/status')
    if arena.get('abi_version') != 3 or arena.get('precision_profile') != 'approx_fp8' or \
            arena.get('resolution') != [1920, 1080]:
        raise ValueError('1080p approximate ABI-v3 arena required')
    records = _records(weight_manifest)
    regions = {row['name']: row for row in arena['regions']}
    required = {'vit.ping', 'vit.pong', 'operator.vit_hidden_fp8',
                'operator.fp8_post', 'operator.fp8_q', 'operator.fp8_k',
                'operator.fp8_v', 'operator.fp8_value'}
    if not required <= regions.keys():
        raise ValueError('arena lacks resident ViT regions')
    tokens = arena['stage_feature_bytes']['vit'] // 1024
    if tokens != 640:
        raise ValueError('unexpected 1080p padded ViT token count')
    if regions['operator.vit_hidden_fp8']['bytes'] < tokens * 4096:
        raise ValueError('ViT hidden arena too small')
    for name in ('operator.fp8_post', 'operator.fp8_q', 'operator.fp8_k',
                 'operator.fp8_v', 'operator.fp8_value'):
        if regions[name]['bytes'] < tokens * 1024:
            raise ValueError(f'ViT arena region too small: {name}')
    specs = {
        'expand': ('e4m3fn', [1024, 4096]),
        'contract': ('e4m3fn', [4096, 1024]),
        'qkv': ('e4m3fn', [1024, 3072]),
        'q_scale': ('fp16_le', [32]),
        'project': ('e4m3fn', [1024, 1024]),
        'ffn_scale': ('fp16_le', [1024]),
        'attention_scale': ('fp16_le', [1024]),
        'perm1024': ('int32_le', [1024]),
        'perm4096': ('int32_le', [4096]),
    }
    blocks = []
    for ordinal, record_number in enumerate(range(31, 39)):
        prefix = f'vit.blocks.{ordinal}.'
        weights = {}
        for suffix, (dtype, shape) in specs.items():
            row = records.get(prefix + suffix)
            if row is None or row.get('dtype') != dtype or row.get('logical_shape') != shape:
                raise ValueError(f'invalid ViT cache record: {prefix + suffix}')
            if suffix == 'qkv' and row.get('storage_layout') != 'column_major_transposed_storage':
                raise ValueError('ViT QKV must use native column-major cache layout')
            weights[suffix] = row
        source = regions['vit.ping' if ordinal % 2 == 0 else 'vit.pong']
        target = regions['vit.pong' if ordinal % 2 == 0 else 'vit.ping']
        blocks.append({
            'name': f'vit.blocks.{ordinal}',
            'struct_size': 152,
            'tokens': tokens,
            'record_number': record_number,
            'input_offset': source['offset'],
            'hidden_offset': regions['operator.vit_hidden_fp8']['offset'],
            'post_offset': regions['operator.fp8_post']['offset'],
            'q_offset': regions['operator.fp8_q']['offset'],
            'k_offset': regions['operator.fp8_k']['offset'],
            'v_offset': regions['operator.fp8_v']['offset'],
            'value_offset': regions['operator.fp8_value']['offset'],
            'next_offset': target['offset'],
            **{f'{name}_weight_offset': row['offset'] for name, row in weights.items()},
        })
    return {
        'schema': 1,
        'abi_version': 3,
        'target_arch': 'gfx1201',
        'resolution': [1920, 1080],
        'precision_profile': 'approx_fp8',
        'status': 'PARTIAL_NATIVE_VIT8_NOT_RUNTIME_ACCEPTED',
        'complete_native_topology': False,
        'block_count': 8,
        'tokens': tokens,
        'attention': 'streamed_global_16_key_tiles_no_nxn_materialization',
        'graph_kernel_nodes_if_recorded': 40,
        'blocks': blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.weights.read_text(encoding='utf-8')),
                   json.loads(args.arena.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in
                      ('status', 'block_count', 'tokens', 'graph_kernel_nodes_if_recorded')},
                     indent=2))


if __name__ == '__main__':
    main()
