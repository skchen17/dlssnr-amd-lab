"""Cross-check an NR model package, derived cache and native stage topology."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from native_model_package_v2 import inspect
except ModuleNotFoundError:  # imported as scripts.audit_native_stage_package
    from scripts.native_model_package_v2 import inspect


def audit(package_raw: bytes, cache: dict, topology: dict) -> dict:
    package = inspect(package_raw)
    fp8 = package['sections'].get('gfx1201_fp8')
    if not fp8 or fp8['sha256'] != cache.get('weights_sha256'):
        raise ValueError('model package does not contain the selected gfx1201 cache')
    if topology.get('status') != 'PARTIAL_NATIVE_SWIN32_256_BLOCKS_NOT_RUNTIME_ACCEPTED' or \
            topology.get('complete_native_topology') or topology.get('block_count') != 44:
        raise ValueError('unexpected topology completeness/status')
    records = {row['name']: row for row in cache['records']}
    record_numbers = []
    for block in topology['blocks']:
        record_numbers.append(block['record_number'])
        prefix = block['name'] + '.attention.'
        fields = {
            'qkv_weight_offset': 'qkv', 'qscale_weight_offset': 'q_scale',
            'permutation_weight_offset': 'permutation',
            'position_bias_weight_offset': 'position_bias',
            'project_weight_offset': 'project',
            'residual_scale_weight_offset': 'attention_scale',
        }
        for field, suffix in fields.items():
            if block[field] != records[prefix + suffix]['offset']:
                raise ValueError(f'topology/cache offset mismatch: {block["name"]}:{field}')
        owner = block['name'].removesuffix('.block')
        ffn_prefix = owner + '.block.ffn.'
        ffn_fields = {
            'ffn_expand_weight_offset': ffn_prefix + 'expand',
            'ffn_contract_weight_offset': ffn_prefix + 'contract',
            'ffn_scale_weight_offset': ffn_prefix + 'residual_scale',
            'ffn_permutation_weight_offset': ffn_prefix + 'permutation',
            'ffn_inverse_permutation_weight_offset': ffn_prefix + 'inverse_permutation',
            'a_index_weight_offset': owner + '.a_index',
            'residual_index_weight_offset': owner + '.residual_index',
        }
        if block['channels'] == 32:
            if block['ffn_mix_weight_offset'] != 0:
                raise ValueError('C32 must not bind a nonexistent FFN mix')
        else:
            ffn_fields['ffn_mix_weight_offset'] = ffn_prefix + 'mix'
        for field, name in ffn_fields.items():
            if block[field] != records[name]['offset']:
                raise ValueError(f'topology/cache offset mismatch: {block["name"]}:{field}')
    if record_numbers != sorted(record_numbers) or len(set(record_numbers)) != 44:
        raise ValueError('native stage records are not uniquely ordered')
    return {
        'schema': 1,
        'status': 'PACKAGE_AND_PARTIAL_TOPOLOGY_CROSS_CHECKED_NOT_RUNTIME_ACCEPTED',
        'package_sha256': hashlib.sha256(package_raw).hexdigest().upper(),
        'gfx1201_weights_sha256': fp8['sha256'],
        'mapped_stage_blocks': len(topology['blocks']),
        'first_record_number': record_numbers[0],
        'last_record_number': record_numbers[-1],
        'complete_native_topology': False,
        'gpu_executed': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.package.read_bytes(),
                   json.loads(args.cache.read_text(encoding='utf-8')),
                   json.loads(args.topology.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
