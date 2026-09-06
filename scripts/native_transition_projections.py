"""Original bottleneck boundary projections; spatial/fusion adapters stay explicit.

No implicit nearest-neighbour resize, fake skip, RGB placeholder or graph gap
substitution. These components are not a complete encoder/decoder boundary.
"""
import torch
from torch import nn
from native_swin_torch import decode_e4, quantize_e4
from native_vit1024 import matrix_indices
from native_split_swin512 import parameter, chunked_linear, channel_permutation


class EncoderFinalProjection(nn.Module):
    native_graph_complete = False
    def __init__(self, raw):
        super().__init__()
        if len(raw) != 524304:
            raise ValueError('expected original block30.layer4 record')
        parameter(self, 'weight', decode_e4(raw)[matrix_indices(512, 1024)])
        self.register_buffer('permutation', channel_permutation(512))

    def forward(self, actual_pooled_features):
        if actual_pooled_features.ndim != 3 or actual_pooled_features.shape[-1] != 512 or actual_pooled_features.dtype != torch.float16:
            raise ValueError('actual pooled FP16 logical features required; no RGB substitute')
        if actual_pooled_features.device != self.weight.device:
            raise ValueError('input/model device mismatch')
        return quantize_e4(chunked_linear(actual_pooled_features[..., self.permutation], self.weight))


class DecoderInputProjection(nn.Module):
    native_graph_complete = False
    spatial_upsample_verified = False
    fusion_adapter_verified = False
    def __init__(self, raw):
        super().__init__()
        if len(raw) != 525312:
            raise ValueError('expected original block39.layer0 record')
        parameter(self, 'weight', decode_e4(raw)[matrix_indices(1024, 512)])
        halves = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        parameter(self, 'residual_scale', halves[524288//2:])
        self.register_buffer('permutation', channel_permutation(1024))

    def forward(self, actual_bottleneck_features):
        if actual_bottleneck_features.ndim != 3 or actual_bottleneck_features.shape[-1] != 1024 or actual_bottleneck_features.dtype != torch.float16:
            raise ValueError('actual FP16 bottleneck features required')
        if actual_bottleneck_features.device != self.weight.device:
            raise ValueError('input/model device mismatch')
        return chunked_linear(actual_bottleneck_features[..., self.permutation], self.weight)

    def fuse(self, actual_bottleneck_features, actual_skip, width, height):
        """Learned 1x1 projection, original 2x replication and encoder residual.

        The padded bottleneck includes whole-feature boundary cells; crop only
        after expansion to the explicitly supplied encoder feature geometry.
        Captured-input checks, not this method, establish recovered semantics.
        """
        if width <= 0 or height <= 0 or width % 4 or height % 4:
            raise ValueError('explicit four-aligned encoder geometry required')
        expected = ((height//2+3)//4*4, (width//2+3)//4*4, 1024)
        if tuple(actual_bottleneck_features.shape) != expected:
            raise ValueError('bottleneck padding/encoder geometry mismatch')
        if (tuple(actual_skip.shape) != (height, width, 512)
                or actual_skip.dtype != torch.float16 or actual_skip.device != self.weight.device):
            raise ValueError('actual same-device FP16 encoder skip required')
        projected = self(actual_bottleneck_features)
        expanded = projected.repeat_interleave(2, 0).repeat_interleave(2, 1)[:height, :width]
        return quantize_e4(expanded + (actual_skip*self.residual_scale).half())
