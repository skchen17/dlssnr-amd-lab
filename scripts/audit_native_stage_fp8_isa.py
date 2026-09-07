"""Audit static gfx1201 resource metadata for resident-FP8 stage kernels."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FAMILIES = {
    'ffn_group': 'stage_ffn_group_fp8',
    'ffn_mix': 'stage_ffn_mix_fp8',
    'qkv_project': 'stage_qkv_project_fp8',
    'qkv_norm_fused': 'stage_qkv_norm_fused_fp8',
    'attention': 'stage_attention_fp8',
    'project': 'stage_project_fp8',
}
NON_MMA_FAMILIES = {
    'qkv_norm_fp16': 'stage_qkv_norm_fp16_inplace',
    'qkv_pack_e4': 'stage_qkv_pack_e4x4_part_from_fp16',
    'scatter': 'stage_scatter_fp8',
}
C512_FAMILIES = {
    'c512_preproject': 'stage_c512_preproject_fp8',
    'c512_group': 'stage_c512_group_fp8',
    'c512_ffn_project': 'stage_c512_ffn_project_fp8',
}
C32_FAMILIES = {'c32_ffn': 'stage_c32_ffn_fp8'}

FAMILY_CHANNELS = {
    'ffn_group': (64, 128, 256),
    'ffn_mix': (64, 128, 256),
    'qkv_project': (32, 64, 128, 256),
    'qkv_norm_fp16': (32, 64, 128, 256),
    'qkv_pack_e4': (32, 64, 128, 256),
    'qkv_norm_fused': (512,),
    'attention': (32, 64, 128, 256, 512),
    'project': (32, 64, 128, 256, 512),
    'scatter': (32, 64, 128, 256, 512),
    **{name: (32,) for name in C32_FAMILIES},
    **{name: (512,) for name in C512_FAMILIES},
}


def audit(text: str) -> dict:
    sections = re.findall(r'; -- Begin function (\S+)(.*?); -- End function', text, re.S)
    rows = []
    for name, body in sections:
        family = next((kind for kind, marker in {**C512_FAMILIES, **C32_FAMILIES, **FAMILIES,
                                                  **NON_MMA_FAMILIES}.items()
                       if marker in name), None)
        if family is None:
            continue
        channel = (512 if 'stage_c512_' in name else 32 if 'stage_c32_' in name else
                   next((value for value in (32, 64, 128, 256, 512)
                         if f'ILi{value}E' in name), None))
        get = lambda pattern: (lambda match: int(match.group(1)) if match else None)(re.search(pattern, body))
        rows.append({
            'family': family,
            'channels': channel,
            'symbol': name,
            'fp8_wmma_sites': len(re.findall(r'\bv_wmma_f32_16x16x16_fp8_fp8\b', body)),
            'vgpr': get(r'\.amdhsa_next_free_vgpr (\d+)'),
            'sgpr': get(r'\.amdhsa_next_free_sgpr (\d+)'),
            'lds_bytes': get(r'\.amdhsa_group_segment_fixed_size (\d+)'),
            'scratch_bytes': get(r'\.amdhsa_private_segment_fixed_size (\d+)'),
            'kernarg_bytes_including_hidden': get(r'\.amdhsa_kernarg_size (\d+)'),
            'global_store_b8_sites': len(re.findall(r'\bglobal_store_b8\b', body)),
            'global_store_b16_sites': len(re.findall(r'\bglobal_store_b16\b', body)),
            'global_store_b32_sites': len(re.findall(r'\bglobal_store_b32\b', body)),
            'hardware_fp8_convert_sites': len(re.findall(r'\bv_cvt_pk_(?:fp8|bf8)_f32\b', body)),
            'wavefront_size': 32 if re.search(r'\.amdhsa_wavefront_size32 1', body)
                              else 64,
        })
    expected = {(family, channel) for family, channels in FAMILY_CHANNELS.items()
                for channel in channels}
    actual = {(row['family'], row['channels']) for row in rows}
    if actual != expected or any(row['fp8_wmma_sites'] < 1 for row in rows
                                 if row['family'] in FAMILIES or
                                 row['family'] in C32_FAMILIES or
                                 row['family'] in C512_FAMILIES) or \
            any(row['wavefront_size'] != 32 for row in rows
                if row['family'] in ('qkv_norm_fp16', 'qkv_pack_e4')) or \
            any(row['global_store_b32_sites'] < 1 or row['global_store_b16_sites'] != 0
                for row in rows if row['family'] == 'qkv_pack_e4') or \
            any(row['global_store_b16_sites'] < 1
                for row in rows if row['family'] == 'qkv_norm_fp16') or \
            any(row['hardware_fp8_convert_sites'] != 0 for row in rows):
        raise ValueError(f'incomplete resident stage ISA: missing={sorted(expected-actual)}')
    return {
        'schema': 1,
        'target': 'gfx1201',
        'scope': 'Static code-object resources only; not runtime occupancy or throughput.',
        'all_matrix_stage_kernels_emit_fp8_wmma': True,
        'all_standard_qkv_norm_and_pack_kernels_are_wave32': True,
        'all_standard_qkv_pack_kernels_publish_software_e4m3x4_words': True,
        'all_fp8_boundaries_avoid_unreliable_gfx1201_hardware_convert': True,
        'kernels': sorted(rows, key=lambda row: (row['channels'], row['family'])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.assembly.read_text(encoding='utf-8'))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
