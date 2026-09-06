import json,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from aggregate_whole_frame_abba import aggregate


def test_whole_frame_abba_requires_exact_hash_and_uses_warm_samples(tmp_path):
    paths=[]
    for index,label in enumerate(('reference','native','native','reference')):
        root=tmp_path/str(index);root.mkdir();paths.append(root);value=2.0 if label=='reference' else 1.0
        report={'checks_pass':True,'allocated_after_release_bytes':0,'reserved_after_release_bytes':0,
            'runs':[{'warm':False,'host_forward_submit_wait_ms':9,'output_sha256':'A','peak_allocated_bytes':20,'peak_reserved_bytes':30,'device_used_bytes_sample':40},
                    {'warm':True,'host_forward_submit_wait_ms':value,'output_sha256':'A','peak_allocated_bytes':20,'peak_reserved_bytes':30,'device_used_bytes_sample':40},
                    {'warm':True,'host_forward_submit_wait_ms':value,'output_sha256':'A','peak_allocated_bytes':20,'peak_reserved_bytes':30,'device_used_bytes_sample':40}]}
        (root/'child.json').write_text(json.dumps(report),encoding='utf-8')
    row=aggregate({'1080p':paths})['groups']['1080p']
    assert row['speedup']==2 and row['bitwise_exact'] and len(row['samples']['native'])==4
