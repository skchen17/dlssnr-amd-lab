"""Original-weight single-color/no-history preblock mathematical candidate.

This implements the actually captured null-history/null-motion/null-mask branch,
not a fabricated zero history. Temporal/depth/mask branches must reject until
implemented. Noise is generated from full-frame coordinates and explicit seed.
"""
import torch
from torch import nn
from native_split_swin512 import parameter
from native_packed_swin import RecoveredPackedSwin,gather_packed,scatter_packed
from native_multiscale import pack_image,unpack_image,pack_outview,average_pool2x2
from native_swin_torch import quantize_e4


def positional_noise(width,height,frame_seed,device):
    if width<=0 or height<=0 or not isinstance(frame_seed,int) or not 0<=frame_seed<=0xffffffff:
        raise ValueError('positive geometry and explicit uint32 frame seed required')
    mask=0xffffffff
    x=torch.arange(width,device=device,dtype=torch.int64)[None,:]
    y=torch.arange(height,device=device,dtype=torch.int64)[:,None]
    state=((x*-1918454973)&mask)^((y*-669632447)&mask)^((frame_seed*-1640531527)&mask)^608135816
    v=(((state>>( (state>>28)+4))^state)*277803737)&mask
    hashed=(v>>22)^v
    uniforms=[]
    for a,b in [(747796405,-1403630843),(-93469191,1192405134),(-895109107,568162667),(-2094846927,878960812)]:
        state=(hashed*a+b)&mask
        v=(((state>>((state>>28)+4))^state)*277803737)&mask
        uniforms.append(((v>>30)^(v>>8)).add(1).float()*2**-24)
    u0,u1,u2,u3=uniforms
    radius0=torch.sqrt(-2*torch.log(u0))
    radius1=torch.sqrt(-2*torch.log(u2))
    return torch.stack([radius0*torch.cos(u1*6.2831854820251465),
                        radius0*torch.sin(u1*6.2831854820251465),
                        radius1*torch.cos(u3*6.2831854820251465)],-1).half()


def reflect_indices(size,padded,device):
    if size<=0 or padded<size:
        raise ValueError('padding cannot shrink actual pixels')
    if size==1:
        return torch.zeros(padded,device=device,dtype=torch.long)
    phase=torch.arange(padded,device=device)%(2*size-2)
    return torch.where(phase<size,phase,2*size-2-phase)


def single_color_features(color,padded_width,padded_height,frame_seed,*,color_scale,conditioning):
    if color.ndim!=3 or color.shape[-1]!=4 or color.dtype!=torch.float16:
        raise ValueError('actual HWC RGBA16F color input required')
    if len(conditioning)!=5:
        raise ValueError('explicit captured conditioning fields required')
    h,w,_=color.shape
    from native_pre_fusion import active_pre_fusion
    fusion=active_pre_fusion()
    if fusion is not None:
        noise=positional_noise(padded_width,padded_height,frame_seed,color.device)
        return fusion(color,noise,padded_width,padded_height,color_scale,conditioning)
    ys=reflect_indices(h,padded_height,color.device)
    xs=reflect_indices(w,padded_width,color.device)
    rgb=color[ys[:,None],xs[None,:],:3]
    centered=((rgb-.5).half()*color_scale).half()
    noise=positional_noise(padded_width,padded_height,frame_seed,color.device)
    one=torch.ones_like(centered[...,:1])
    constants=(conditioning if torch.is_tensor(conditioning) else
               torch.tensor(conditioning,device=color.device,dtype=torch.float16))
    if constants.device!=color.device or constants.dtype!=torch.float16 or constants.shape!=(5,):
        raise ValueError('conditioning tensor must be resident FP16[5]')
    constants=constants.expand(padded_height,padded_width,5)
    # Shared filler: noise3,1,currentRGB3,previousRGB3, five condition values,0.
    # Null-history branch explicitly uses currentRGB as previousRGB.
    return torch.cat([noise,one,centered,centered,constants,torch.zeros_like(one)],-1)


def project_input_features(features,weight,*,rows_per_chunk=65536):
    """Pointwise16->32 FP32 projection, bounded GEMM rows then original FP16 store.

    The unbounded3D GEMM at2432x1408 failed an independent row oracle and repeat
    checks on the tested ROCm build. Row batches preserve every pixel's formula;
    no spatial filtering, crop, coordinate change, FP8 conversion or CPU fallback.
    Cat/casts retain autograd for the separate future training path.
    """
    if (features.ndim!=3 or features.shape[-1]!=16 or weight.shape!=(16,32)
            or features.dtype!=torch.float16 or weight.dtype!=torch.float16
            or features.device!=weight.device or type(rows_per_chunk) is not int or rows_per_chunk<=0):
        raise ValueError('HWC16 FP16 features, same-device16x32 weight and positive row batch required')
    from native_matrix_fusion import active_matrix_fusion
    matrix=active_matrix_fusion()
    if matrix is not None and 'pre_input' in matrix.modules:return matrix.pre_project(features,weight)
    flat=features.reshape(-1,16);matrix=weight.float()
    return torch.cat([(part.float()@matrix).half() for part in flat.split(rows_per_chunk)]).reshape(*features.shape[:2],32)


class RecoveredSingleColorPreblock(nn.Module):
    temporal_inputs_supported=False
    image_quality_verified=False

    def __init__(self,raw):
        super().__init__()
        if len(raw)!=21696:
            raise ValueError('original block0 record required')
        k,n=torch.arange(16)[:,None],torch.arange(32)[None,:]
        offset=8208+n//16*512+(n%8*4+k%8//2)*16+n%16//8*8+(k%2+(k>=8)*2)*2
        parameter(self,'input_project',torch.frombuffer(bytearray(raw),dtype=torch.float16)[offset//2])
        # Remove only the embedded input projection to expose the plain Swin record.
        self.swin=RecoveredPackedSwin(raw[:8208]+raw[9232:],record_kind='swin32')

    def forward(self,color,padded_width,padded_height,frame_seed,*,color_scale,conditioning,window_batch=12):
        if window_batch<=0 or padded_width%8 or padded_height%8 or color.device!=self.input_project.device:
            raise ValueError('same-device input, eight-aligned geometry and positive batch required')
        from native_pre_fusion import active_pre_fusion
        fusion=active_pre_fusion()
        if fusion is not None and fusion.project_pack_enabled:
            noise=positional_noise(padded_width,padded_height,frame_seed,color.device)
            packed_input=fusion.project_pack(color,noise,self.input_project,padded_width,padded_height,
                                             color_scale,conditioning)
        else:
            features=single_color_features(color,padded_width,padded_height,frame_seed,
                                           color_scale=color_scale,conditioning=conditioning)
            # This original initial projection is FP16, not the FP8 policy used by FFNs.
            projected=project_input_features(features,self.input_project)
            packed_input=pack_image(projected)
        windows,mapping=gather_packed(packed_input,padded_width,padded_height,32,0,0)
        from native_grid_policy import run_windows
        values=run_windows(self.swin,windows,window_batch,'pre')
        packed=scatter_packed(values,mapping,padded_width,padded_height)
        pooled=average_pool2x2(unpack_image(packed,padded_width,padded_height,32))
        return {'skip':quantize_e4(packed),'outview':quantize_e4(pack_outview(pooled))}
