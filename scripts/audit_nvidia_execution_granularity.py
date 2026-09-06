"""Read-only static PTX/recorded-launch audit; never executes proprietary code.

Counts are source instruction sites, not dynamic executions or DRAM traffic.
Prints metadata only; no weights, PTX bodies or captured pointers are exported.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def entry_stats(text, name):
    entries=list(re.finditer(r'\.visible\s+\.entry\s+(\w+)\s*\(',text))
    for i,m in enumerate(entries):
        if m[1]!=name:continue
        end=entries[i+1].start() if i+1<len(entries) else len(text)
        body=text[m.start():end]
        ops=Counter()
        for line in re.split(r'[;\n]',body):
            match=re.match(r'\s*(?:@!?%\w+\s+)?([a-z][\w.:]*)\s',line)
            if match:ops[match[1]]+=1
        return dict(line_start=text.count('\n',0,m.start())+1,
                    line_end=text.count('\n',0,end)+1,
                    static_shared_bytes=sum(map(int,re.findall(r'\.shared\s+\.align\s+\d+\s+\.b8\s+\w+\[(\d+)\]',body))),
                    virtual_register_declarations=re.findall(r'\.reg\s+(\.\w+)\s+%\w+<(\d+)>',body),
                    instruction_sites={k:v for k,v in sorted(ops.items()) if any(x in k for x in
                    ('mma.','shared','global','local','surf','sust.','tex.','f16','e4m3','atom','membar','bar.'))})
    raise ValueError('entry missing: '+name)


def audit(root):
    capture=root/'results/20260831_002356_rtx5070_feature18_full_frame'
    modules=root/'results/20260831_010100_all_runtime_modules'
    sequence=capture/'frame_001_sequence.csv'
    trace=capture/'module_trace.jsonl'
    manifest=modules/'extraction_manifest.json'
    rows=list(csv.DictReader(sequence.open(encoding='utf-8-sig')))
    # Legacy baseline path events contain unescaped Windows paths; parse only
    # launch-call records (strict JSON), never silently discard a malformed call.
    events=[json.loads(line) for line in trace.read_text(encoding='utf-8-sig').splitlines()
            if '"ev":"nvapi_launch_cu_kernel_chain_call"' in line]
    calls=[e for e in events if e.get('ev')=='nvapi_launch_cu_kernel_chain_call' and e.get('frame')==1]
    assets=[sequence,trace,manifest]
    module_map={int(m['dll_offset'],16):m for m in json.loads(manifest.read_text())['modules']}
    output=[]
    for slot in (1,2,3,5,6,7,9,146,149,150,153,154):
        row=rows[slot]; module=module_map[int(row['module_dll_offset'],16)]
        path=next(modules.glob(f"module_{module['index']:02d}_*.ptx"))
        if path not in assets:assets.append(path)
        stats=entry_stats(path.read_text(),row['function_name'])
        output.append(dict(slot=slot,function=row['function_name'],ptx=str(path.relative_to(root)),
                           grid=[int(row['grid_'+a]) for a in 'xyz'],block=[int(row['block_'+a]) for a in 'xyz'],
                           dynamic_shared=int(row['dynamic_shared']),**stats))
    return dict(schema=1,scope='640x360 variant02 captured frame; static PTX sites only',
                recorded_slots=len(rows),distinct_functions=len(set(r['function_name'] for r in rows)),
                frame1_chain_calls=len(calls),frame1_command_list_count=len(set(c['command_list'] for c in calls)),
                frame1_kernels_per_chain=dict(Counter(str(c['num_kernels']) for c in calls)),
                sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},
                entries=output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    print(json.dumps(audit(p.parse_args().root),indent=2))
