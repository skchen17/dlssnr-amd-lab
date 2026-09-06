#include "output_head_full_common.hlsli"
ByteAddressBuffer RawA : register(t0);
ByteAddressBuffer Head : register(t1);
RWByteAddressBuffer Hidden : register(u0);
static const uint HiddenValues = 81u * 49u * 4u * 4u * 16u * 32u;

uint Activate(uint bits)
{
    float x = f16tof32(bits), clamped = clamp(x, -4.0f, 4.0f), absolute = abs(clamped);
    float first = f16tof32(FloatToHalfRN(mad(-0.055908203125f, absolute, 0.447265625f)));
    float second = f16tof32(FloatToHalfRN(mad(clamped, first, 0.89453125f)));
    return FloatToHalfRN(x * second);
}

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 4u; if (base >= HiddenValues) return;
    uint split = base / 32u, row = split % 16u; split /= 16u;
    uint group = split % 4u; split /= 4u; uint pass = split % 4u; uint cta = split / 4u;
    uint packed = 0u;
    [unroll] for (uint j = 0u; j < 4u; ++j) {
        uint column = (base % 32u) + j; float accumulator = 0.0f;
        uint tile = pass * 1024u + (column / 16u) * 512u;
        for (uint k = 0u; k < 32u; ++k)
            accumulator = mad(DecodeE4M3(LoadByte(RawA, cta * 2048u + PackedAIndex(group,row,k))),
                              DecodeE4M3(LoadByte(Head, PackedBIndex(tile,k,column % 16u))), accumulator);
        packed |= EncodeE4M3(Activate(FloatToHalfRN(accumulator))) << (j * 8u);
    }
    Hidden.Store(base, packed);
}
