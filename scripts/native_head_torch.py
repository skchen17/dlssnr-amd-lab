"""Recovered output-head candidate; accepts real upstream features, never RGB substitutes.

Dynamic head geometry does NOT establish dynamic full-network geometry or HDR.
"""
import torch
from torch import nn
from native_swin_torch import RecoveredSwin32


class RecoveredHead32(nn.Module):
    native_graph_complete = False

    def __init__(self, packed_weights):
        super().__init__()
        self.cache_layout_enabled=False
        self._layout_cache={}
        self._layout_geometry=None
        self._layout_bytes=0
        self.block = RecoveredSwin32(packed_weights, record_kind='head32')
        halves = torch.frombuffer(bytearray(packed_weights), dtype=torch.float16)
        self.register_parameter('main_scale', nn.Parameter(halves[8272 // 2:8336 // 2].clone(), requires_grad=False))
        self.register_parameter('skip_scale', nn.Parameter(halves[8336 // 2:8400 // 2].clone(), requires_grad=False))
        k, c = torch.arange(32)[:, None], torch.arange(4)[None, :]
        kk = k % 16
        offset = 20784 + k // 16 * 512 + (c * 4 + kk % 8 // 2) * 16 + (kk % 2 + (kk >= 8) * 2) * 2
        self.register_parameter('tail', nn.Parameter(halves[offset // 2].clone(), requires_grad=False))
        if not all(bool(torch.isfinite(p).all()) for p in self.parameters()):
            raise ValueError('nonfinite output-head parameter')
        local = torch.arange(2048)
        pair, within, lane, cell = local & 1, (local >> 1) % 8, (local >> 4) % 32, local // 512
        order = torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])
        operation = cell * 8 + order[within]
        within = operation % 8
        source_bases = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7])
        source = source_bases[(operation >= 16).long() * 4 + within // 2]
        output = operation % 16 // 8 + (within & 1) * 2
        q = (lane & 19) | ((lane << 1) & 8) | ((lane >> 1) & 4)
        # middle.zwxy for lane bit16, followed by yxwz for lane bit4.
        selected = output ^ ((lane & 16) != 0).long() * 2 ^ ((lane & 4) != 0).long()
        source_lane = q ^ (selected * 8)
        load = source // 2
        for name, value in {'plane': load // 2, 'dy': source_lane // 16 + (load & 1) * 2,
                            'dx': source_lane // 4 % 4, 'channel': source_lane % 4 * 4 + (source & 1) * 2 + pair,
                            'skip_cell_x': cell % 2, 'skip_cell_y': cell // 2,
                            'skip_local': lane * 16 + torch.tensor([0, 4, 2, 6, 8, 12, 10, 14])[within] + pair,
                            'scale_index': (operation // 2 % 4) * 8 + lane % 4 * 2 + pair}.items():
            self.register_buffer(name, value.long())

    def _apply(self,fn,recurse=True):
        self.clear_layout_cache()
        return super()._apply(fn,recurse=recurse)

    def clear_layout_cache(self):
        self._layout_cache.clear();self._layout_geometry=None;self._layout_bytes=0

    def train(self,mode=True):
        self.clear_layout_cache()
        return super().train(mode)

    def _load_from_state_dict(self,*args,**kwargs):
        self.clear_layout_cache()
        return super()._load_from_state_dict(*args,**kwargs)

    def layout(self,ctas,padded_width,padded_height):
        """Pure geometry only. Int32 indices are safe under the explicit size gate."""
        gx=(padded_width+11)//8
        cx,cy=ctas[:,None]%gx,ctas[:,None]//gx
        mh,mw=padded_height//2,padded_width//2
        x,y=(-4+cx*8)//2+self.dx,(-4+cy*8)//2+self.dy
        main_valid=(x>=0)&(x<mw)&(y>=0)&(y<mh)
        main_index=(((self.plane*mh+y)*mw+x)*16+self.channel).clamp(0,padded_width*padded_height*8-1)
        sh,sw=padded_height//4,padded_width//4
        sx,sy=(-4+cx*8)//4+self.skip_cell_x,(-4+cy*8)//4+self.skip_cell_y
        skip_valid=(sx>=0)&(sx<sw)&(sy>=0)&(sy<sh)
        skip_index=((sy*sw+sx)*512+self.skip_local).clamp(0,padded_width*padded_height*32-1)
        return main_index,main_valid,skip_index,skip_valid

    def forward_range(self,main,skip,start,count,padded_width,padded_height):
        """Validated consecutive CTA range; cached entries contain NO image/weight values."""
        if (type(padded_width) is not int or type(padded_height) is not int or min(padded_width,padded_height)<=0
                or padded_width%8 or padded_height%8 or main.numel()!=padded_width*padded_height*8
                or skip.numel()!=padded_width*padded_height*32 or main.device!=skip.device):
            raise ValueError('invalid output-head feature geometry')
        total=((padded_width+11)//8)*((padded_height+11)//8)
        if type(start) is not int or type(count) is not int or start<0 or count<=0 or start+count>total:
            raise ValueError('invalid output-head CTA range')
        if not self.cache_layout_enabled or self.training or torch.is_grad_enabled():
            return self(main,skip,torch.arange(start,start+count,device=main.device),padded_width,padded_height)
        if padded_width*padded_height*32>=2**31:
            raise ValueError('head index exceeds reviewed int32 range')
        geometry=(padded_width,padded_height,main.device)
        if geometry!=self._layout_geometry:
            self.clear_layout_cache();self._layout_geometry=geometry
        key=(start,count)
        if key not in self._layout_cache:
            # 2 indices(int32) + 2 valid masks(bool) per packed window element.
            added=count*2048*10
            if self._layout_bytes+added>1300000000:
                raise ValueError('head geometry cache budget exceeded; no silent fallback')
            parts=self.layout(torch.arange(start,start+count,device=main.device),padded_width,padded_height)
            parts=(parts[0].int(),parts[1],parts[2].int(),parts[3])
            self._layout_cache[key]=parts;self._layout_bytes+=added
        fused=self.fuse(main,skip,None,padded_width,padded_height,layout=self._layout_cache[key])
        return self.project(fused)

    def fuse(self, main, skip, ctas, padded_width, padded_height,*,layout=None):
        if padded_width <= 0 or padded_height <= 0 or padded_width % 8 or padded_height % 8:
            raise ValueError('head requires padded feature dimensions divisible by eight')
        if main.numel() != padded_width * padded_height * 8 or skip.numel() != padded_width * padded_height * 32:
            raise ValueError('head requires both actual upstream feature buffers')
        ai,av,bi,bv=self.layout(ctas,padded_width,padded_height) if layout is None else layout
        a=main.flatten()[ai].masked_fill(~av,0)
        b=skip.flatten()[bi].masked_fill(~bv,0)
        first = (a * self.main_scale[self.scale_index]).half()
        return (first.float() + b.float() * self.skip_scale[self.scale_index].float()).half()

    def forward(self, main, skip, ctas, padded_width, padded_height):
        fused = self.fuse(main, skip, ctas, padded_width, padded_height)
        return self.project(fused)

    def project(self,fused):
        projected = self.block(fused)
        first = (projected[..., :16].float() @ self.tail[:16].float()).half()
        return (first.float() + projected[..., 16:].float() @ self.tail[16:].float()).half()


def compose_legacy_sdr_debug(residual, base, padded_width, padded_height):
    """Explicit OLD clamp/blend contract, never used as HDR implementation."""
    height, width, channels = base.shape
    gx, gy = (padded_width + 11) // 8, (padded_height + 11) // 8
    if width > padded_width or height > padded_height or channels != 4 or residual.shape != (gx * gy, 64, 4):
        raise ValueError('surface geometry mismatch')
    cta = torch.arange(gx * gy, device=residual.device)[:, None]
    token = torch.arange(64, device=residual.device)[None]
    group, within = token // 16, token % 16
    x = cta % gx * 8 - 4 + (group & 1) * 4 + within % 4
    y = cta // gx * 8 - 4 + (group >> 1) * 4 + within // 4
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    image = torch.empty_like(base)
    image[y[valid], x[valid]] = (base[y[valid], x[valid]].float() + residual[valid].float() * .25).clamp(0, 1).half()
    image[..., 3] = 1
    return image
