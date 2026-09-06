"""Private, non-pickle tensor-model package. Runtime never opens capture plans.

Hash binding proves provenance/integrity, not recovered mathematical correctness.
This version intentionally accepts only the single-color diagnostic branch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ORIGINAL_SHA='A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'


def sha(raw):
    return hashlib.sha256(raw).hexdigest().upper()


def decode(manifest, raw):
    if (manifest.get('schema')!=1 or manifest.get('architecture')!='single_color_71_v1'
            or manifest.get('original_model_sha256')!=ORIGINAL_SHA
            or manifest.get('fine_tuned') is not False
            or manifest.get('precision')!='native_fp16'
            or manifest.get('composition')!='LEGACY_SDR_DIAGNOSTIC_ONLY'
            or manifest.get('weights_sha256')!=sha(raw)):
        raise ValueError('unsupported model package or hash mismatch')
    records={}; end=0
    for r in manifest['records']:
        match=re.fullmatch(r'block(\d+)\.layer(\d+)\.layer',r['name'])
        if not match or r['offset']!=end or type(r['bytes']) is not int or r['bytes']<=0:
            raise ValueError('invalid original record layout')
        key=tuple(map(int,match.groups())); end+=r['bytes']
        value=raw[r['offset']:end]
        if key in records or len(value)!=r['bytes'] or sha(value)!=r['sha256']:
            raise ValueError('duplicate/truncated/modified record')
        records[key]=value
    expected={(b,l) for b in range(71) for l in
              (range(5) if 31<=b<=38 or b==30 else range(4) if 23<=b<=29 or 40<=b<=47 else range(1))}
    if set(records)!=expected or len(records)!=152 or end!=len(raw):
        raise ValueError('complete 152-record ABI scope required; includes eight non-neural placeholders')
    origins={int(b):tuple(v) for b,v in manifest['window_origins'].items()}
    if set(origins)!=set(range(1,70)) or any(len(v)!=2 or any(x not in (0,-4) for x in v) for v in origins.values()):
        raise ValueError('invalid window geometry')
    settings=manifest['settings']
    if settings!={'color_scale':.125,'conditioning':[0.,1.,1.,-1.,-1.]}:
        raise ValueError('unsupported conditioning; do not infer exposure/temporal data')
    return records,origins,settings


def load_package(root, expected_manifest_sha256):
    root=Path(root); spec=(root/'model.json').read_bytes()
    if sha(spec)!=expected_manifest_sha256:
        raise ValueError('model description hash mismatch')
    return decode(json.loads(spec),(root/'weights.bin').read_bytes())


def export(root):
    # Offline-only import: exports weights/math configuration, never activations.
    from native_capture_fixture import whole_frame_fixture
    records,origins,settings=whole_frame_fixture()
    raw=bytearray(); table=[]
    for (b,l),value in sorted(records.items()):
        table.append({'name':f'block{b}.layer{l}.layer','offset':len(raw),'bytes':len(value),'sha256':sha(value)})
        raw.extend(value)
    manifest={'schema':1,'architecture':'single_color_71_v1','original_model_sha256':ORIGINAL_SHA,
              'fine_tuned':False,'precision':'native_fp16','composition':'LEGACY_SDR_DIAGNOSTIC_ONLY',
              'weights_sha256':sha(raw),'records':table,'window_origins':origins,'settings':settings,
              'active_neural_records':144,'abi_placeholders':8,'inactive_temporal_scalar_included':False,
              'game_ready':False,'hdr_supported':False,'private':True}
    spec=json.dumps(manifest,indent=2,allow_nan=False).encode()
    decode(json.loads(spec),raw)
    root.mkdir(parents=True,exist_ok=False)
    with (root/'weights.bin').open('xb') as f: f.write(raw)
    with (root/'model.json').open('xb') as f: f.write(spec)
    result={'manifest_sha256':sha(spec),'weights_sha256':sha(raw),'private':True,'game_ready':False}
    with (root/'hashes.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    export(p.parse_args().output)
