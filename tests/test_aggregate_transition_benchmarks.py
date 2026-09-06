import json,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from aggregate_transition_benchmarks import aggregate
import pytest


def test_three_abba_processes_produce_twelve_samples(tmp_path):
    paths=[]
    for index in range(3):
        root=tmp_path/str(index);root.mkdir();paths.append(root)
        phases=[]
        for label in ('reference','native','native','reference')*2:
            phases.append({'implementation':label,'gpu_event_ms':[2.0 if label=='reference' else 1.0],
                           'incremental_peak_allocated_bytes':20 if label=='reference' else 10})
        report={'scope':'1080p encoder transition A-B-B-A, two groups, 1 iterations per phase','checks_pass':True,
            'allocated_after_release_bytes':0,'reserved_after_release_bytes':0,
            'rows':[{'transition':'C32->C64','phases':phases,'correctness':{'skip':{'bitwise_exact':True},'resident':{'bitwise_exact':True}}}]}
        (root/'child.json').write_text(json.dumps(report),encoding='utf-8')
    result=aggregate(paths)['groups']['1080p_encoder']
    assert result['samples_per_implementation_per_transition']==12
    assert result['sum_speedup']==2
    assert result['rows'][0]['native_max_incremental_peak_bytes']==10


def test_negative_event_sample_is_rejected(tmp_path):
    paths=[]
    for index in range(3):
        root=tmp_path/str(index);root.mkdir();paths.append(root)
        phases=[]
        for label in ('reference','native','native','reference')*2:
            value=-1.0 if index==1 and label=='native' and len(phases)==1 else 1.0
            phases.append({'implementation':label,'gpu_event_ms':[value],'incremental_peak_allocated_bytes':1})
        report={'scope':'1440p decoder transition A-B-B-A, two groups, 1 iterations per phase','checks_pass':True,
            'allocated_after_release_bytes':0,'reserved_after_release_bytes':0,
            'rows':[{'transition':'C64->C32','phases':phases,'correctness':{'resident':{'bitwise_exact':True}}}]}
        (root/'child.json').write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='invalid A-B-B-A'):
        aggregate(paths)
