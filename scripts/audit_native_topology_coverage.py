"""Combine partial native topologies and report exact 71-block coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def family(number: int) -> str:
    if number == 0:
        return 'Pre'
    if 1 <= number <= 4 or 66 <= number <= 69:
        return 'C32'
    if 5 <= number <= 8 or 62 <= number <= 65:
        return 'C64'
    if 9 <= number <= 14 or 56 <= number <= 61:
        return 'C128'
    if 15 <= number <= 22 or 48 <= number <= 55:
        return 'C256'
    if 23 <= number <= 30 or 40 <= number <= 47:
        return 'C512'
    if 31 <= number <= 38:
        return 'ViT'
    if number == 39:
        return 'transition'
    if number == 70:
        return 'Head'
    raise ValueError('invalid recovered block number')


def audit(wide: dict, split: dict, vit: dict, bottleneck: dict,
          transitions: dict) -> dict:
    if wide.get('status') != 'PARTIAL_NATIVE_SWIN32_256_BLOCKS_NOT_RUNTIME_ACCEPTED' or \
            wide.get('block_count') != 44 or wide.get('complete_native_topology'):
        raise ValueError('invalid standard wide topology')
    if split.get('status') != 'PARTIAL_NATIVE_C512_SPLIT_NOT_RUNTIME_ACCEPTED' or \
            split.get('block_count') != 16 or split.get('complete_native_topology'):
        raise ValueError('invalid C512 topology')
    if vit.get('status') != 'PARTIAL_NATIVE_VIT8_NOT_RUNTIME_ACCEPTED' or \
            vit.get('block_count') != 8 or vit.get('complete_native_topology'):
        raise ValueError('invalid ViT topology')
    if bottleneck.get('status') != 'PARTIAL_NATIVE_BOTTLENECK_TRANSITIONS_NOT_RUNTIME_ACCEPTED' or \
            bottleneck.get('decoder_record') != 39 or bottleneck.get('complete_native_topology'):
        raise ValueError('invalid bottleneck topology')
    if transitions.get('status') != 'PARTIAL_NATIVE_SCALE_TRANSITIONS_NOT_RUNTIME_ACCEPTED' or \
            transitions.get('transition_count') != 8 or transitions.get('complete_native_topology'):
        raise ValueError('invalid scale-transition topology')
    if wide.get('resolution') != split.get('resolution') or \
            wide.get('resolution') != vit.get('resolution') or \
            wide.get('precision_profile') != split.get('precision_profile') or \
            wide.get('precision_profile') != vit.get('precision_profile') or \
            wide.get('target_arch') != split.get('target_arch') or \
            wide.get('target_arch') != vit.get('target_arch') or \
            wide.get('target_arch') != bottleneck.get('target_arch') or \
            wide.get('resolution') != bottleneck.get('resolution') or \
            wide.get('target_arch') != transitions.get('target_arch') or \
            wide.get('resolution') != transitions.get('resolution') or \
            wide.get('precision_profile') != transitions.get('precision_profile'):
        raise ValueError('partial topology target mismatch')
    blocks = sorted(wide['blocks'] + split['blocks'] + vit['blocks'],
                    key=lambda row: row['record_number'])
    numbers = [row['record_number'] for row in blocks]
    if len(set(numbers)) != len(numbers):
        raise ValueError('partial native topologies overlap')
    numbers.append(39)
    numbers.sort()
    expected = list(range(1, 70))
    if numbers != expected:
        raise ValueError('unexpected native stage block coverage')
    missing = [number for number in range(71) if number not in set(numbers)]
    missing_by_family = {}
    for number in missing:
        missing_by_family.setdefault(family(number), []).append(number)
    nodes = (wide['graph_kernel_nodes_if_recorded'] +
             split['graph_kernel_nodes_if_recorded'] +
             vit['graph_kernel_nodes_if_recorded'] +
             bottleneck['graph_kernel_nodes_if_recorded'] +
             transitions['graph_kernel_nodes_if_recorded'])
    return {
        'schema': 1,
        'status': 'PARTIAL_69_OF_71_NATIVE_RECORDS_WITH_SCALE_TRANSITIONS_NOT_RUNTIME_ACCEPTED',
        'target_arch': wide['target_arch'],
        'resolution': wide['resolution'],
        'precision_profile': wide['precision_profile'],
        'native_block_count': len(numbers),
        'native_record_numbers': numbers,
        'theoretical_graph_kernel_nodes': nodes,
        'theoretical_graph_memcpy_nodes': (bottleneck['graph_memcpy_nodes_if_recorded'] +
                                           transitions['graph_memcpy_nodes_if_recorded']),
        'kernel_node_budget_limit': 512,
        'remaining_kernel_node_budget': 512 - nodes,
        'missing_block_count': len(missing),
        'missing_record_numbers': missing,
        'missing_by_family': missing_by_family,
        'unimplemented_record_subpaths': {
            'pre_temporal_import_and_swin': [0],
            'head_and_history_export': [70],
        },
        'native_scale_transition_count': transitions['transition_count'],
        'complete_native_topology': False,
        'graph_captured': False,
        'gpu_executed': False,
        'runtime_accepted': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--split512', type=Path, required=True)
    parser.add_argument('--vit', type=Path, required=True)
    parser.add_argument('--bottleneck', type=Path, required=True)
    parser.add_argument('--transitions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.wide.read_text(encoding='utf-8')),
                   json.loads(args.split512.read_text(encoding='utf-8')),
                   json.loads(args.vit.read_text(encoding='utf-8')),
                   json.loads(args.bottleneck.read_text(encoding='utf-8')),
                   json.loads(args.transitions.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in
                      ('status', 'native_block_count', 'theoretical_graph_kernel_nodes',
                       'missing_by_family')}, indent=2))


if __name__ == '__main__':
    main()
