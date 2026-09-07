"""Audit static gfx1201 resource metadata for resident-FP8 ViT kernels."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FAMILIES = {
    'ffn_expand': 'vit_ffn_expand_fp8',
    'ffn_contract': 'vit_ffn_contract_fp8',
    'qkv_norm': 'vit_qkv_norm_fp8',
    'global_attention': 'vit_global_attention_fp8',
    'project': 'vit_project_fp8',
}


def audit(text: str) -> dict:
    sections = re.findall(r'; -- Begin function (\S+)(.*?); -- End function', text, re.S)
    rows = []
    for family, marker in FAMILIES.items():
        matches = [(name, body) for name, body in sections if marker in name]
        if len(matches) != 1:
            raise ValueError(f'expected one {family} kernel, found {len(matches)}')
        name, body = matches[0]
        def number(pattern):
            match = re.search(pattern, body)
            return int(match.group(1)) if match else None
        row = {
            'family': family,
            'symbol': name,
            'fp8_wmma_sites': len(re.findall(r'\bv_wmma_f32_16x16x16_fp8_fp8\b', body)),
            'vgpr': number(r'\.amdhsa_next_free_vgpr (\d+)'),
            'sgpr': number(r'\.amdhsa_next_free_sgpr (\d+)'),
            'lds_bytes': number(r'\.amdhsa_group_segment_fixed_size (\d+)'),
            'scratch_bytes': number(r'\.amdhsa_private_segment_fixed_size (\d+)'),
        }
        if row['fp8_wmma_sites'] < 1 or any(row[key] is None for key in
                                            ('vgpr', 'sgpr', 'lds_bytes', 'scratch_bytes')):
            raise ValueError(f'incomplete ISA metadata for {family}')
        rows.append(row)
    return {
        'schema': 1,
        'target': 'gfx1201',
        'scope': 'Static code-object resources only; not runtime occupancy or throughput.',
        'all_matrix_kernels_emit_fp8_wmma': True,
        'attention_materializes_nxn': False,
        'kernels_per_block': 5,
        'kernels': rows,
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
