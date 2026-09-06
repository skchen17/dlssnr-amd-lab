import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_model_package import decode,sha,ORIGINAL_SHA,load_package


def fixture():
    keys=[(b,l) for b in range(71) for l in
          (range(5) if 31<=b<=38 or b==30 else range(4) if 23<=b<=29 or 40<=b<=47 else range(1))]
    raw=bytes(range(len(keys)))
    spec={'schema':1,'architecture':'single_color_71_v1','original_model_sha256':ORIGINAL_SHA,
          'fine_tuned':False,'precision':'native_fp16','composition':'LEGACY_SDR_DIAGNOSTIC_ONLY',
          'weights_sha256':sha(raw),'records':[{'name':f'block{b}.layer{l}.layer','offset':i,'bytes':1,
                                               'sha256':sha(raw[i:i+1])} for i,(b,l) in enumerate(keys)],
          'window_origins':{str(b):[0,0] for b in range(1,70)},
          'settings':{'color_scale':.125,'conditioning':[0.,1.,1.,-1.,-1.]}}
    return spec,raw


def test_complete_record_scope_does_not_imply_quality():
    spec,raw=fixture();records,origins,settings=decode(spec,raw)
    assert len(records)==152 and len(origins)==69
    assert (70,1) not in records


@pytest.mark.parametrize('mutation',[
    lambda s:s.update(fine_tuned=True),lambda s:s.update(original_model_sha256='0'*64),
    lambda s:s.update(composition='HDR'),lambda s:s.update(precision='recovered_k32'),
    lambda s:s['records'].pop(),lambda s:s['records'][0].update(offset=1),
    lambda s:s['records'][0].update(sha256='0'*64),lambda s:s['window_origins'].pop('1'),
    lambda s:s['settings'].update(color_scale=1),lambda s:s['records'][1].update(name='block0.layer0.layer')])
def test_corrupt_and_unsupported_packages_fail(mutation):
    spec,raw=fixture();mutation(spec)
    with pytest.raises(ValueError):decode(spec,raw)


def test_package_hash_pinned_by_caller_and_no_external_path(tmp_path):
    spec,raw=fixture();encoded=json.dumps(spec).encode()
    (tmp_path/'model.json').write_bytes(encoded);(tmp_path/'weights.bin').write_bytes(raw)
    assert len(load_package(tmp_path,sha(encoded))[0])==152
    with pytest.raises(ValueError):load_package(tmp_path,'0'*64)
    (tmp_path/'weights.bin').write_bytes(raw+b'bad')
    with pytest.raises(ValueError):load_package(tmp_path,sha(encoded))
