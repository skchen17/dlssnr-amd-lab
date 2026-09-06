"""Twenty-two original decoder blocks48..69 with whole-feature scale adapters.

Inputs are the genuine bottleneck-side outview and four encoder skip tensors.
This is NOT a complete RGB model; the encoder and final color head are separate.
No captured tensor or runtime PTX/DLL is consumed by this module.
"""
import torch
from torch import nn
from native_upsample_swin import RecoveredUpsampleSwin
from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
from native_multiscale import unpack_image, pack_outview
from native_swin_torch import quantize_e4


def stage_role(block):
    for first,last,c in [(48,55,256),(56,61,128),(62,65,64),(66,69,32)]:
        if first<=block<=last:
            return c, 'upsample' if block==first else 'outview' if block==last else 'plain'
    raise ValueError('decoder block must be48..69')


class RecoveredDecoderPyramid(nn.Module):
    native_graph_complete=False
    image_quality_verified=False
    gpu_verified=False

    def __init__(self,records,origins):
        super().__init__()
        expected=set(range(48,70))
        if set(records)!=expected or set(origins)!=expected:
            raise ValueError('all22 distinct original decoder blocks and window origins required')
        self.origins={}
        modules={}
        for block in range(48,70):
            if tuple(origins[block]) not in ((0,0),(0,-4),(-4,0),(-4,-4)):
                raise ValueError('unsupported window origin')
            self.origins[block]=tuple(origins[block])
            c,role=stage_role(block)
            cls=RecoveredUpsampleSwin if role=='upsample' else RecoveredPackedSwin
            modules[str(block)]=cls(records[block],record_kind=f'swin{c}')
        self.blocks=nn.ModuleDict(modules)

    def stages(self,low_outview,skips,width,height,*,window_batch=12):
        if width<=0 or height<=0 or width%64 or height%64 or window_batch<=0:
            raise ValueError('explicit full feature dimensions need64-alignment and positive batch')
        if set(skips)!={32,64,128,256}:
            raise ValueError('four genuine encoder skip inputs required')
        from native_transition_policy import active_transition_policy
        transition=active_transition_policy()
        value=low_outview;value_layout='resident' if transition.resident else 'outview'
        for block in range(48,70):
            c,role=stage_role(block)
            w,h=width//(c//32),height//(c//32)
            model=self.blocks[str(block)]
            ox,oy=self.origins[block]
            if role=='upsample':
                value=(model.forward_resident(value,skips[c],w,h,ox,oy,window_batch=window_batch)
                       if value_layout=='resident' else model(value,skips[c],w,h,ox,oy,window_batch=window_batch))
                value_layout='resident'
            else:
                windows,mapping=gather_packed(value,w,h,c,ox,oy)
                from native_grid_policy import run_windows
                logical=run_windows(model,windows,window_batch,f'c{c}')
                packed=scatter_packed(logical,mapping,w,h)
                # Blocks55/61/65 feed another native upsample transition and
                # can stay packed. Block69 still feeds the recovered Head ABI,
                # whose planar/outview contract has not changed.
                needs_outview=role=='outview' and (not transition.resident or block==69)
                value=pack_outview(unpack_image(packed,w,h,c)) if needs_outview else packed
                value_layout='outview' if needs_outview else 'resident'
                value=quantize_e4(value)
            yield block,value

    def forward(self,low_outview,skips,width,height,*,window_batch=12):
        output=None
        for _,output in self.stages(low_outview,skips,width,height,window_batch=window_batch):
            pass
        return output
