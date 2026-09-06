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
    'head_attention_bounded': ('head_attention_window_fused','ELb0EEv'),
    'c32_attention_bounded': ('head_attention_window_fused','ELb1EEv'),
    'c32_attention_staged_qkv': 'c32_qkv_norm_staged',
    'c32_attention_staged_core': 'c32_attention_core_staged',
    'c32_ffn': 'c32_ffn_wmma',
    'c32_qk': 'c32_qk_wmma',
    'c32_pv': 'c32_pv_wmma',
    'c32_output': 'c32_project_wmma',
    'c64_ffn': 'wide_ffn_wmmaILi64E',
    'c64_group_ffn': 'wide_group_ffn_wmmaILi64E',
    'c64_group_ffn_fp8w': 'wide_group_ffn_fp8wILi64E',
    'c128_ffn': 'wide_ffn_wmmaILi128E',
    'c128_group_ffn': 'wide_group_ffn_wmmaILi128E',
    'c128_group_ffn_fp8w': 'wide_group_ffn_fp8wILi128E',
    'c64_qkv': 'wide_qkv_wmmaILi64E',
    'c64_qk': 'wide_qk_wmmaILi64E',
    'c64_pv': 'wide_pv_wmmaILi64E',
    'c64_output': 'wide_project_wmmaILi64E',
    'c64_attention_bounded': 'wide_attention_head_fusedILi64E',
    'c64_attention_query': 'wide_attention_query_fusedILi64E',
    'c128_qkv': 'wide_qkv_wmmaILi128E',
    'c128_qk': 'wide_qk_wmmaILi128E',
    'c128_pv': 'wide_pv_wmmaILi128E',
    'c128_output': 'wide_project_wmmaILi128E',
    'c128_attention_bounded': 'wide_attention_head_fusedILi128E',
    'c128_attention_query': 'wide_attention_query_fusedILi128E',
    'c256_ffn': 'wide_ffn_wmmaILi256E',
    'c256_group_ffn': 'wide_group_ffn_wmmaILi256E',
    'c256_group_ffn_fp8w': 'wide_group_ffn_fp8wILi256E',
    'c512_group_ffn': 'c512_group_ffn_wmma',
}


def audit(text):
    sections = re.findall(r'; -- Begin function (\S+)(.*?); -- End function', text, re.S)
    result = {}
    for family, marker in FAMILIES.items():
        variants = []
        for name, body in sections:
            parts=(marker,) if isinstance(marker,str) else marker
            if not all(part in name for part in parts):
                continue
            fp16 = len(re.findall(r'\bv_wmma_f32_16x16x16_f16\b', body))
            fp8 = len(re.findall(r'\bv_wmma_f32_16x16x16_fp8_fp8\b', body))
            vgpr = re.search(r'\.amdhsa_next_free_vgpr (\d+)', body)
            sgpr = re.search(r'\.amdhsa_next_free_sgpr (\d+)', body)
            scratch = re.search(r'\.amdhsa_private_segment_fixed_size (\d+)', body)
            lds = re.search(r'\.amdhsa_group_segment_fixed_size (\d+)', body)
            variants.append({'symbol': name, 'fp16_wmma_sites': fp16, 'fp8_wmma_sites': fp8,
                             'vgpr': int(vgpr.group(1)) if vgpr else None,
                             'sgpr': int(sgpr.group(1)) if sgpr else None,
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
    parser.add_argument('--asic-spm-summary', type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.assembly.read_text(encoding='utf-8'))
    if arguments.asic_spm_summary:
        spm=json.loads(arguments.asic_spm_summary.read_text(encoding='utf-8-sig'))
        asic=spm['asic'];gran=asic['vgpr_alloc_granularity'];capacity=asic['vgprs_per_simd']
        max_waves=asic['waves_per_simd'];simds=asic['simds_per_cu'];lds_capacity=asic['lds_bytes_per_cu']
        for variants in report['families'].values():
            for row in variants:
                values=[int(value) for value in re.findall(r'Li(\d+)E',row['symbol'])]
                waves=next((value for value in reversed(values) if value in (1,2,4)),None)
                if ('head_attention_window_fused' in row['symbol'] or 'wide_attention_head_fused' in row['symbol']
                        or 'c32_qkv_norm_staged' in row['symbol'] or 'c32_attention_core_staged' in row['symbol']):waves=4
                if 'wide_group_ffn_wmma' in row['symbol'] or 'wide_group_ffn_fp8w' in row['symbol'] or 'wide_attention_query_fused' in row['symbol']:waves=1
                allocated=((row['vgpr']+gran-1)//gran)*gran
                vgpr_waves=min(max_waves,capacity//allocated)
                if row['lds_bytes']:
                    workgroups=lds_capacity//row['lds_bytes']
                    lds_waves=min(max_waves,(workgroups*waves)//simds) if waves else None
                else:lds_waves=max_waves
                resident=min(vgpr_waves,lds_waves) if lds_waves is not None else vgpr_waves
                row.update({'waves_per_workgroup':waves,'allocated_vgpr_per_wave':allocated,
                    'vgpr_limited_waves_per_simd':vgpr_waves,'lds_limited_waves_per_simd':lds_waves,
                    'theoretical_resident_waves_per_simd':resident,
                    'theoretical_occupancy_percent':100.0*resident/max_waves})
        report['occupancy_scope']='Static resource upper bound from code-object metadata and captured ASIC limits; not measured runtime occupancy.'
        report['asic']=asic
    with arguments.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
