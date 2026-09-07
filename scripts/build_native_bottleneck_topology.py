"""Build the fixed 1080p C512<->ViT bottleneck transition descriptor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(weight_manifest: dict, arena: dict) -> dict:
    if weight_manifest.get('target_arch') != 'gfx1201' or \
            weight_manifest.get('status') != 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED':
        raise ValueError('unexpected weight cache provenance/status')
    if arena.get('abi_version') != 3 or arena.get('precision_profile') != 'approx_fp8' or \
            arena.get('resolution') != [1920, 1080]:
        raise ValueError('1080p approximate ABI-v3 arena required')
    records = {row['name']: row for row in weight_manifest['records']}
    regions = {row['name']: row for row in arena['regions']}
    specs = {
        'enc_project.weight': ('e4m3fn', [512, 1024]),
        'enc_project.permutation': ('int32_le', [512]),
        'dec_project.weight': ('e4m3fn', [1024, 512]),
        'dec_project.residual_scale': ('fp16_le', [512]),
        'dec_project.permutation': ('int32_le', [1024]),
    }
    for name, (dtype, shape) in specs.items():
        row = records.get(name)
        if row is None or row.get('dtype') != dtype or row.get('logical_shape') != shape:
            raise ValueError(f'invalid bottleneck weight: {name}')
    required = {'c512.ping', 'c512.skip', 'vit.ping'}
    if not required <= regions.keys():
        raise ValueError('arena lacks bottleneck regions')
    row = {
        'struct_size': 112,
        'tokens': 640,
        'feature_width': 60,
        'feature_height': 36,
        'low_width': 32,
        'low_height': 20,
        'c512_input_offset': regions['c512.ping']['offset'],
        'c512_skip_offset': regions['c512.skip']['offset'],
        'vit_input_offset': regions['vit.ping']['offset'],
        'vit_output_offset': regions['vit.ping']['offset'],
        'c512_output_offset': regions['c512.ping']['offset'],
        'encoder_weight_offset': records['enc_project.weight']['offset'],
        'encoder_permutation_offset': records['enc_project.permutation']['offset'],
        'decoder_weight_offset': records['dec_project.weight']['offset'],
        'decoder_scale_offset': records['dec_project.residual_scale']['offset'],
        'decoder_permutation_offset': records['dec_project.permutation']['offset'],
    }
    return {
        'schema': 1,
        'abi_version': 3,
        'target_arch': 'gfx1201',
        'resolution': [1920, 1080],
        'precision_profile': 'approx_fp8',
        'status': 'PARTIAL_NATIVE_BOTTLENECK_TRANSITIONS_NOT_RUNTIME_ACCEPTED',
        'encoder_after_record': 30,
        'decoder_record': 39,
        'graph_kernel_nodes_if_recorded': 2,
        'graph_memcpy_nodes_if_recorded': 1,
        'complete_native_topology': False,
        'descriptor': row,
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
    print(json.dumps({key: result[key] for key in ('status', 'encoder_after_record',
                                                   'decoder_record')}, indent=2))


if __name__ == '__main__':
    main()
