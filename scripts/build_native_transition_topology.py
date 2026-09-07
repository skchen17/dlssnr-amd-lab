"""Build fixed resident-FP8 scale-transition descriptors for NRPlan.

This compiler is CPU-only.  It binds the four encoder downsample boundaries
and four symmetric decoder upsample boundaries to C++-owned arena/weight
offsets; it does not execute or validate GPU math.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ENCODER = ((4, 32), (8, 64), (14, 128), (22, 256))
DECODER = ((48, 256), (56, 128), (62, 64), (66, 32))


def _records(manifest: dict) -> dict[str, dict]:
    records = {row['name']: row for row in manifest['records']}
    if len(records) != len(manifest['records']):
        raise ValueError('duplicate weight-cache record')
    return records


def _weight(records: dict, name: str, shape: list[int], dtype: str) -> dict:
    row = records.get(name)
    if row is None or row.get('logical_shape') != shape or row.get('dtype') != dtype:
        raise ValueError(f'invalid transition weight: {name}')
    return row


def build(weight_manifest: dict, arena: dict, wide_topology: dict) -> dict:
    if weight_manifest.get('target_arch') != 'gfx1201' or \
            weight_manifest.get('status') != 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED':
        raise ValueError('unexpected weight cache')
    if arena.get('abi_version') != 3 or arena.get('precision_profile') != 'approx_fp8' or \
            arena.get('resolution') != [1920, 1080] or arena.get('resident_item_bytes') != 1:
        raise ValueError('1080p resident-FP8 ABI-v3 arena required')
    records = _records(weight_manifest)
    regions = {row['name']: row for row in arena['regions']}
    wide = {row['record_number']: row for row in wide_topology.get('blocks', [])}
    if wide_topology.get('status') != 'PARTIAL_NATIVE_SWIN32_256_BLOCKS_NOT_RUNTIME_ACCEPTED':
        raise ValueError('native wide topology required for FP16 encoder boundaries')
    padded_width, padded_height = 1920, 1152
    rows = []
    for record_number, channels in ENCODER:
        source_width = padded_width // 2 // (channels // 32)
        source_height = padded_height // 2 // (channels // 32)
        project = _weight(records, f'encoder.{record_number}.pool_project',
                          [channels, channels * 2], 'e4m3fn')
        permutation = _weight(records, f'encoder.{record_number}.permutation',
                              [channels], 'int32_le')
        rows.append({
            'struct_size': 96,
            'direction': 1,
            'anchor_record': record_number,
            'channels': channels,
            'source_width': source_width,
            'source_height': source_height,
            'target_width': source_width // 2,
            'target_height': source_height // 2,
            'source_origin_x': wide[record_number]['origin_x'],
            'source_origin_y': wide[record_number]['origin_y'],
            'source_offset': wide[record_number]['output_fp16_offset'],
            'source_resident_offset': regions[f'c{channels}.ping']['offset'],
            'skip_offset': regions[f'c{channels}.skip']['offset'],
            'target_offset': regions[f'c{channels * 2}.ping']['offset'],
            'project_weight_offset': project['offset'],
            'permutation_weight_offset': permutation['offset'],
            'skip_scale_weight_offset': 0,
        })
    for record_number, channels in DECODER:
        target_width = padded_width // 2 // (channels // 32)
        target_height = padded_height // 2 // (channels // 32)
        project = _weight(records, f'decoder.blocks.{record_number}.project',
                          [channels * 2, channels], 'e4m3fn')
        permutation = _weight(records, f'decoder.blocks.{record_number}.permutation',
                              [channels * 2], 'int32_le')
        scale = _weight(records, f'decoder.blocks.{record_number}.skip_scale',
                        [channels], 'fp16_le')
        rows.append({
            'struct_size': 96,
            'direction': 2,
            'anchor_record': record_number,
            'channels': channels,
            'source_width': target_width // 2,
            'source_height': target_height // 2,
            'target_width': target_width,
            'target_height': target_height,
            'source_origin_x': 0,
            'source_origin_y': 0,
            'source_offset': regions[f'c{channels * 2}.ping']['offset'],
            'source_resident_offset': regions[f'c{channels * 2}.ping']['offset'],
            'skip_offset': regions[f'c{channels}.skip']['offset'],
            'target_offset': regions[f'c{channels}.ping']['offset'],
            'project_weight_offset': project['offset'],
            'permutation_weight_offset': permutation['offset'],
            'skip_scale_weight_offset': scale['offset'],
        })
    rows.sort(key=lambda row: (row['anchor_record'], row['direction']))
    return {
        'schema': 1,
        'abi_version': 3,
        'target_arch': 'gfx1201',
        'resolution': [1920, 1080],
        'precision_profile': 'approx_fp8',
        'status': 'PARTIAL_NATIVE_SCALE_TRANSITIONS_NOT_RUNTIME_ACCEPTED',
        'complete_native_topology': False,
        'transition_count': len(rows),
        'encoder_count': len(ENCODER),
        'decoder_count': len(DECODER),
        'graph_kernel_nodes_if_recorded': len(rows),
        'graph_memcpy_nodes_if_recorded': len(ENCODER),
        'descriptors': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.weights.read_text(encoding='utf-8')),
                   json.loads(args.arena.read_text(encoding='utf-8')),
                   json.loads(args.wide.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in
                      ('status', 'transition_count', 'graph_kernel_nodes_if_recorded',
                       'graph_memcpy_nodes_if_recorded')}, indent=2))


if __name__ == '__main__':
    main()
