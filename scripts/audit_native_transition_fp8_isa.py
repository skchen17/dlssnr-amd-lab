"""Audit static gfx1201 resources for resident-FP8 scale transitions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def audit(text: str) -> dict:
    sections = re.findall(r'; -- Begin function (\S+)(.*?); -- End function', text, re.S)
    rows = []
    for name, body in sections:
        direction = ('encoder' if 'encoder_transition_fp8' in name else
                     'decoder' if 'decoder_transition_fp8' in name else None)
        if direction is None:
            continue
        channel = next((value for value in (32, 64, 128, 256)
                        if f'ILi{value}E' in name), None)
        if channel is None:
            continue
        def number(pattern):
            match = re.search(pattern, body)
            return int(match.group(1)) if match else None
        rows.append({
            'direction': direction,
            'channels': channel,
            'symbol': name,
            'fp8_wmma_sites': len(re.findall(r'\bv_wmma_f32_16x16x16_fp8_fp8\b', body)),
            'vgpr': number(r'\.amdhsa_next_free_vgpr (\d+)'),
            'sgpr': number(r'\.amdhsa_next_free_sgpr (\d+)'),
            'lds_bytes': number(r'\.amdhsa_group_segment_fixed_size (\d+)'),
            'scratch_bytes': number(r'\.amdhsa_private_segment_fixed_size (\d+)'),
        })
    expected = {(direction, channel) for direction in ('encoder', 'decoder')
                for channel in (32, 64, 128, 256)}
    actual = {(row['direction'], row['channels']) for row in rows}
    if actual != expected or any(row['fp8_wmma_sites'] < 1 for row in rows):
        raise ValueError(f'incomplete transition ISA: missing={sorted(expected-actual)}')
    if any(row[key] is None for row in rows for key in
           ('vgpr', 'sgpr', 'lds_bytes', 'scratch_bytes')):
        raise ValueError('transition ISA lacks static resource metadata')
    return {
        'schema': 1,
        'target': 'gfx1201',
        'scope': 'Static code-object resources only; not measured occupancy or throughput.',
        'all_matrix_kernels_emit_fp8_wmma': True,
        'all_kernels_no_lds': all(row['lds_bytes'] == 0 for row in rows),
        'all_kernels_no_scratch': all(row['scratch_bytes'] == 0 for row in rows),
        'kernels': sorted(rows, key=lambda row: (row['direction'], row['channels'])),
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
