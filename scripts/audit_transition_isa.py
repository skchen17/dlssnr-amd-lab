"""Extract static resource metadata for native scale-transition HIP kernels."""
import argparse,json,re
from pathlib import Path


MARKERS={
    'encoder_pool_permute_skip':'pool_permute_skip_kernel',
    'encoder_quantize_pack':'quantize_pack_kernel',
    'decoder_unpack_permute':'unpack_permute_kernel',
    'decoder_expand_skip_pack':'expand_skip_pack_kernel',
}


def audit(text):
    sections=re.findall(r'; -- Begin function (\S+)(.*?); -- End function',text,re.S)
    result={}
    for name,marker in MARKERS.items():
        matches=[(symbol,body) for symbol,body in sections if marker in symbol]
        if len(matches)!=1:raise ValueError(f'expected one ISA function for {name}, found {len(matches)}')
        symbol,body=matches[0]
        def field(pattern):
            value=re.search(pattern,body);return int(value.group(1)) if value else None
        result[name]={'symbol':symbol,'vgpr':field(r'\.amdhsa_next_free_vgpr (\d+)'),
            'sgpr':field(r'\.amdhsa_next_free_sgpr (\d+)'),
            'scratch_bytes':field(r'\.amdhsa_private_segment_fixed_size (\d+)'),
            'lds_bytes':field(r'\.amdhsa_group_segment_fixed_size (\d+)')}
    return {'schema':1,'scope':'Static clang AMDGPU code-object metadata; not measured occupancy or throughput.',
            'kernels':result,'all_kernels_no_scratch':all(v['scratch_bytes']==0 for v in result.values()),
            'all_kernels_no_lds':all(v['lds_bytes']==0 for v in result.values())}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assembly',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x',encoding='utf-8') as f:json.dump(audit(a.assembly.read_text(encoding='utf-8')),f,indent=2)
