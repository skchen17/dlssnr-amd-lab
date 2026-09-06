"""Extract per-candidate WMMA/resource evidence from clang AMDGPU assembly."""
import argparse
import json
import re
from pathlib import Path


FAMILIES = {
    'head_ffn': 'head_ffn_wmma',
    'head_output': 'head_project_wmma',
    'head_qk': 'head_qk_wmma',
    'head_qkv': 'head_qkv_project_wmma',
    'head_pv': 'head_pv_wmma',
    'c32_ffn': 'c32_ffn_wmma',
    'c32_qk': 'c32_qk_wmma',
    'c32_pv': 'c32_pv_wmma',
    'c32_output': 'c32_project_wmma',
    'c64_ffn': 'wide_ffn_wmmaILi64E',
    'c128_ffn': 'wide_ffn_wmmaILi128E',
    'c64_qkv': 'wide_qkv_wmmaILi64E',
    'c64_qk': 'wide_qk_wmmaILi64E',
    'c64_pv': 'wide_pv_wmmaILi64E',
    'c64_output': 'wide_project_wmmaILi64E',
    'c128_qkv': 'wide_qkv_wmmaILi128E',
    'c128_qk': 'wide_qk_wmmaILi128E',
    'c128_pv': 'wide_pv_wmmaILi128E',
    'c128_output': 'wide_project_wmmaILi128E',
    'c256_ffn': 'wide_ffn_wmmaILi256E',
    'c512_group_ffn': 'c512_group_ffn_wmma',
}


def audit(text):
    sections = re.findall(r'; -- Begin function (\S+)(.*?); -- End function', text, re.S)
    result = {}
    for family, marker in FAMILIES.items():
        variants = []
        for name, body in sections:
            if marker not in name:
                continue
            fp16 = len(re.findall(r'\bv_wmma_f32_16x16x16_f16\b', body))
            fp8 = len(re.findall(r'\bv_wmma_f32_16x16x16_fp8_fp8\b', body))
            vgpr = re.search(r'\.amdhsa_next_free_vgpr (\d+)', body)
            scratch = re.search(r'\.amdhsa_private_segment_fixed_size (\d+)', body)
            lds = re.search(r'\.amdhsa_group_segment_fixed_size (\d+)', body)
            variants.append({'symbol': name, 'fp16_wmma_sites': fp16, 'fp8_wmma_sites': fp8,
                             'vgpr': int(vgpr.group(1)) if vgpr else None,
                             'scratch_bytes': int(scratch.group(1)) if scratch else None,
                             'lds_bytes': int(lds.group(1)) if lds else None})
        if not variants or not all(row['fp16_wmma_sites'] + row['fp8_wmma_sites'] > 0 for row in variants):
            raise ValueError(f'missing WMMA evidence for {family}')
        result[family] = variants
    return {'families': result, 'all_candidate_variants_contain_wmma': True,
            'scope': 'Static clang AMDGPU assembly; proves emitted instructions/resources, not runtime occupancy or throughput.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = audit(arguments.assembly.read_text(encoding='utf-8'))
    with arguments.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
