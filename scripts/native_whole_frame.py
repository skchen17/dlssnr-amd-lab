"""Original-weight single-color whole-frame tensor candidate, not game deployment.

No captured activation, PTX, DLL, address, size or CPU fallback is loaded here.
All encoder skips are generated in this forward pass. Input conditioning/window
origins are explicit model configuration; temporal and formal HDR remain outside
this candidate. Diagnostic legacy SDR composition is explicitly named.
"""
import torch
from torch import nn
from native_preblock import RecoveredSingleColorPreblock
from native_multiscale import RecoveredDownsampleSwin,RecoveredOutviewSwin,unpack_image,pack_image
from native_packed_swin import RecoveredPackedSwin,gather_packed,scatter_packed
from native_split_image512 import RecoveredSplitImage512
from native_transition_projections import EncoderFinalProjection,DecoderInputProjection
from native_vit1024 import RecoveredVit1024,RecoveredVitBottleneck1024,RECORD_SIZES as VIT_PARTS
from native_split_swin512 import RECORD_SIZES as SPLIT_PARTS
from native_decoder_pyramid import RecoveredDecoderPyramid
from native_head_torch import RecoveredHead32,compose_legacy_sdr_debug
from native_swin_torch import quantize_e4


def feature_geometry(width,height,max_padded_pixels):
    if not all(isinstance(v,int) and v>0 for v in (width,height,max_padded_pixels)):
        raise ValueError('positive integer image dimensions and resource budget required')
    pw,ph=(width+127)//128*128,(height+127)//128*128
    if pw*ph>max_padded_pixels:
        raise ValueError('offline candidate feature budget exceeded; no silent downscale or tile fallback')
    return pw,ph


class SingleColorWholeFrame(nn.Module):
    native_graph_complete=False  # full original temporal/color contract not yet recovered
    single_color_tensor_chain_connected=True
    game_runtime_ready=False
    hdr_supported=False

    def __init__(self,records,origins,*,color_scale,conditioning,max_padded_pixels=1048576):
        super().__init__()
        if set(origins)!=set(range(1,70)):
            raise ValueError('all explicit original window configurations required')
        self.origins=origins
        if max_padded_pixels<=0:
            raise ValueError('explicit positive offline workspace pixel budget required')
        self.max_padded_pixels=max_padded_pixels
        self.color_scale,self.conditioning=color_scale,conditioning
        self.pre=RecoveredSingleColorPreblock(records[(0,0)])
        modules={}
        for first,last,c in [(1,4,32),(5,8,64),(9,14,128),(15,22,256)]:
            for b in range(first,last+1):
                cls=RecoveredOutviewSwin if b==first else RecoveredDownsampleSwin if b==last else RecoveredPackedSwin
                modules[str(b)]=cls(records[(b,0)],record_kind=f'swin{c}')
        self.encoder=nn.ModuleDict(modules)
        def split(b,**kw):
            return RecoveredSplitImage512({k:records[(b,i)] for i,k in enumerate(SPLIT_PARTS)},**kw)
        self.encoder512=nn.ModuleDict({str(b):split(b,input_layout='outview' if b==23 else 'packed',
                                                  output_layout='pool' if b==30 else 'packed') for b in range(23,31)})
        self.enc_project=EncoderFinalProjection(records[(30,4)])
        self.vit=RecoveredVitBottleneck1024({b:RecoveredVit1024(**{k:records[(b,i)] for i,k in enumerate(VIT_PARTS)})
                                           for b in range(31,39)})
        self.dec_project=DecoderInputProjection(records[(39,0)])
        self.decoder512=nn.ModuleDict({str(b):split(b,output_layout='outview' if b==47 else 'packed') for b in range(40,48)})
        self.decoder=RecoveredDecoderPyramid({b:records[(b,0)] for b in range(48,70)},
                                            {b:origins[b] for b in range(48,70)})
        self.head=RecoveredHead32(records[(70,0)])

    def stages(self,color,frame_seed,*,window_batch=12,boundary_batch=None,query_chunk=32):
        # Scheduling only: each window keeps its original full-frame coordinates.
        # Opt-in until same-input numerical/resource checks pass; deployment stays12.
        boundary_batch=window_batch if boundary_batch is None else boundary_batch
        if type(window_batch) is not int or type(boundary_batch) is not int or min(window_batch,boundary_batch)<=0:
            raise ValueError('positive integer window batches required')
        if type(query_chunk) is not int or query_chunk not in (32,128,256,512,1024):
            raise ValueError('unreviewed query chunk; key reduction remains128')
        if color.ndim!=3 or color.shape[-1]!=4 or color.dtype!=torch.float16:
            raise ValueError('actual RGBA16F image required')
        h,w,_=color.shape
        # Explicit candidate feature-padding policy; no image downscaling or tile stitching.
        pw,ph=feature_geometry(w,h,self.max_padded_pixels)
        pre=self.pre(color,pw,ph,frame_seed,color_scale=self.color_scale,conditioning=self.conditioning,
                     window_batch=boundary_batch)
        value=pre['outview']; skips={}
        yield 0,value
        for first,last,c in [(1,4,32),(5,8,64),(9,14,128),(15,22,256)]:
            sw,sh=pw//2//(c//32),ph//2//(c//32)
            for b in range(first,last+1):
                model=self.encoder[str(b)]; ox,oy=self.origins[b]
                if b in (first,last):
                    value=model(value,sw,sh,ox,oy,window_batch=window_batch)
                else:
                    windows,mapping=gather_packed(value,sw,sh,c,ox,oy)
                    from native_grid_policy import run_windows
                    value=quantize_e4(scatter_packed(run_windows(model,windows,window_batch,f'c{c}'),mapping,sw,sh))
                if b==last:
                    skips[c]=value['skip']; value=value['captured_outview']
                yield b,value
        sw,sh=pw//32,ph//32
        for b in range(23,31):
            value=self.encoder512[str(b)](value,sw,sh,*self.origins[b],window_batch=window_batch)
            if b==30:
                skips[512]=value['skip']; value=self.enc_project(value['pooled'])
            yield b,value
        bh,bw,_=value.shape
        value=value.reshape(1,bh*bw,1024)
        for b in range(31,39):
            value=self.vit.blocks[b-31](value,query_chunk=query_chunk)
            yield b,value
        value=self.dec_project.fuse(value[0].reshape(bh,bw,1024),unpack_image(skips[512],sw,sh,512),sw,sh)
        value=pack_image(value)
        yield 39,value
        for b in range(40,48):
            value=self.decoder512[str(b)](value,sw,sh,*self.origins[b],window_batch=window_batch)
            yield b,value
        for b,value in self.decoder.stages(value,{c:skips[c] for c in (32,64,128,256)},pw//2,ph//2,window_batch=window_batch):
            yield b,value
        cta_count=((pw+11)//8)*((ph+11)//8)
        from native_head_fusion import active_head_fusion
        fusion=active_head_fusion()
        if fusion is not None and fusion.whole_grid:
            # One full-grid input dispatch and one full-grid invocation of each
            # following Head boundary.  No arange, split, Python batch loop or
            # cat is present in this opt-in path.  Kernel fusion is evaluated
            # independently; reducing host scheduling alone is not treated as
            # a performance pass.
            residual=self.head.forward_range(value,pre['skip'],0,cta_count,pw,ph)
        else:
            ctas=torch.arange(cta_count,device=color.device)
            residual=torch.cat([self.head.forward_range(value,pre['skip'],start,min(boundary_batch,len(ctas)-start),pw,ph)
                            for start in range(0,len(ctas),boundary_batch)]) if self.head.cache_layout_enabled or fusion is not None else torch.cat(
                                [self.head(value,pre['skip'],batch,pw,ph) for batch in ctas.split(boundary_batch)])
        yield 70,compose_legacy_sdr_debug(residual,color,pw,ph)

    def forward(self,color,frame_seed,*,window_batch=12,boundary_batch=None,query_chunk=32):
        for _,value in self.stages(color,frame_seed,window_batch=window_batch,boundary_batch=boundary_batch,query_chunk=query_chunk):
            pass
        return value
