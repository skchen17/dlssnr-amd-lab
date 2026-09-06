#include "output_head_full_common.hlsli"
ByteAddressBuffer First128 : register(t0);
ByteAddressBuffer Head : register(t1);
RWByteAddressBuffer Projection : register(u0);
static const uint OutputValues = 81u * 49u * 48u * 16u * 8u;

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 2u; if (base >= OutputValues) return;
    uint split = base / 8u, row = split % 16u; split /= 16u;
    uint mma = split % 48u, cta = split / 48u, group = mma / 12u, within = mma % 12u;
    uint outputWord = 0u;
    [unroll] for (uint j = 0u; j < 2u; ++j) {
        uint column = (base % 8u) + j, outputColumn = (within & 1u) * 8u + column;
        uint tile = uint(8400 + NR_ATTENTION_WEIGHT_SHIFT) + (within / 2u) * 512u; float accumulator = 0.0f;
        uint inputBase = (cta * 4u * 16u + group * 16u + row) * 32u;
        for (uint k = 0u; k < 32u; ++k) {
            uint activation = EncodeE4M3(LoadHalf(First128, (inputBase + Permute32(k)) * 2u));
            uint weight = LoadByte(Head, PackedBIndex(tile,k,outputColumn));
            accumulator = mad(DecodeE4M3(activation), DecodeE4M3(weight), accumulator);
        }
        outputWord |= FloatToHalfRN(accumulator) << (j * 16u);
    }
    Projection.Store(base * 2u, outputWord);
}
