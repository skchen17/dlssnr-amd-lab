"""Private offline evidence reader. Never imported by a deployed model."""
import json
from pathlib import Path
import struct
from run_swin1h_native_validation import checked
from describe_native_reconstruction import ORIGINAL_SHA


class OriginalRecords:
    def __init__(self, root=Path('local_models/decoded_310_8')):
        self.arena = checked(root/'model_arena.raw', ORIGINAL_SHA)
        self.records = {r['name']: r for r in json.loads((root/'manifest.json').read_bytes())['tensors']}

    def get(self, block, layer=0):
        rec = self.records[f'block{block}.layer{layer}.layer']
        start, size = rec['arena_offset'], rec['data_bytes']
        return self.arena[start:start+size]

    def bind(self, block, layer, offset):
        if self.records[f'block{block}.layer{layer}.layer']['arena_offset'] != offset:
            raise ValueError('captured weight binding differs from original named record')


class CaptureCases:
    def __init__(self, root):
        self.root = Path(root)
        self.specs = {s['slot']: s for s in json.loads((self.root/'manifest.json').read_bytes())['slots']}

    def asset(self, slot, name):
        spec = self.specs[slot]
        item = spec.get('assets', spec)[name]
        return checked(self.root/f'slot{slot}'/item['path'], item['sha256'])

    def integers(self, slot, offset, count):
        return struct.unpack_from(f'<{count}i', self.asset(slot, 'params'), offset)

    def input(self, slot, param_offset, size):
        spec = self.specs[slot]
        arena = self.asset(slot, 'activation_arena')
        offset = next(v['arena_offset'] for v in spec['activation_param_views'] if v['param_offset'] == param_offset)
        raw = arena[offset:offset+size]
        if len(raw) != size:
            raise ValueError('truncated actual input')
        return raw

    def output(self, slot, param_offset):
        item = next(o for o in self.specs[slot]['outputs'] if o['param_offset'] == param_offset)
        a = item['asset']
        return checked(self.root/f'slot{slot}'/a['path'], a['sha256'])[:item['logical_bytes']]


def whole_frame_fixture():
    """Export only original records and mathematical configuration from old ABI.

    Does not read/execute PTX, load a DLL, or read any activation capture.
    The old plan is private provenance for window origins, not a runtime plan.
    """
    original=OriginalRecords()
    root=Path('results/20260831_234000_full_graph_integrated_plan')
    plan=json.loads((root/'plan_fine_n0_jointulp_slots2to6_square_postblock.json').read_bytes())
    specs={s['slot']:s for s in plan['slots']}
    records={}
    for name in original.records:
        if name.endswith('.layer'):
            b,l,_=name.split('.')
            records[(int(b[5:]),int(l[5:]))]=original.get(int(b[5:]),int(l[5:]))
    def params(slot):
        spec=specs[slot]['params']; path=Path(spec['path'])
        return checked(path if path.is_absolute() else root/path,spec['sha256'])
    origins={}
    for b in range(1,70):
        if b<=22:
            slot=b+1; layer=0; offset=24 if b<=4 else 32
        elif b<=30:
            slot=26+(b-23)*4; layer=2; offset=24
        elif b<=38:
            origins[b]=(0,0)  # global attention has no spatial window origin
            continue
        elif b==39:
            origins[b]=(0,0)
            continue
        elif b<=47:
            slot=102+(b-40)*4; layer=2; offset=24
        else:
            slot=b+84; layer=0; offset=24 if b>=66 else 32
        original.bind(b,layer,specs[slot]['weight_param_views'][0]['weight_offset'])
        _,_,ox,oy=struct.unpack_from('<4i',params(slot),offset)
        if ox not in (0,-4) or oy not in (0,-4):
            raise ValueError('unexpected original spatial origin')
        origins[b]=(ox,oy)
    pre=params(1)
    if any(struct.unpack_from('<4Q',pre,8)):
        raise ValueError('only captured null-history/null-motion/null-depth/null-mask branch implemented')
    def f(offset):
        return struct.unpack_from('<f',pre,offset)[0]
    if struct.unpack_from('<i',pre,192)[0]!=1 or f(184)>=0 or f(188)>=0:
        raise ValueError('unsupported preblock condition branch')
    conditioning=(f(180),f(172),f(176),-1.,-1.)
    return records,origins,{'color_scale':2*f(196),'conditioning':conditioning}
