#include "output_head_full_common.hlsli"
ByteAddressBuffer RawHalf : register(t0);
ByteAddressBuffer Hidden : register(t1);
ByteAddressBuffer Head : register(t2);
RWByteAddressBuffer First128 : register(u0);
static const uint OutputValues = 81u * 49u * 4u * 16u * 32u;

uint ResidualSeed(uint cta, uint group, uint row, uint column)
{
    uint mma = (column / 16u) * 2u + (column % 16u) / 8u;
    static const uint operationPairs[8] = {0u,2u,1u,3u,4u,6u,5u,7u};
    uint lane = (row % 8u) * 4u + (column % 8u) / 2u;
    uint element = (row >= 8u ? 2u : 0u) + (column & 1u);
    uint operation = operationPairs[mma * 2u + element / 2u], pairHalf = element & 1u;
    uint rawIndex = cta * 2048u + group * 512u + lane * 16u + operation * 2u + pairHalf;
    uint scaleIndex = 8208u + (mma * 4u + lane % 4u) * 4u + pairHalf * 2u;
    return FloatToHalfRN(f16tof32(LoadHalf(RawHalf, rawIndex * 2u)) * f16tof32(LoadHalf(Head, scaleIndex)));
}

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 2u; if (base >= OutputValues) return;
    uint split = base / 32u, row = split % 16u; split /= 16u;
    uint group = split % 4u, cta = split / 4u, outputWord = 0u;
    [unroll] for (uint j = 0u; j < 2u; ++j) {
        uint column = (base % 32u) + j;
        uint accumulatorBits = ResidualSeed(cta, group, row, column);
        for (uint pass = 0u; pass < 4u; ++pass) {
            float accumulator = f16tof32(accumulatorBits);
            uint hiddenBase = ((cta * 4u + pass) * 4u + group) * 512u + row * 32u;
            uint tile = 4096u + pass * 1024u + (column / 16u) * 512u;
            for (uint k = 0u; k < 32u; ++k)
                accumulator = mad(DecodeE4M3(LoadByte(Hidden, hiddenBase + Permute32(k))),
                                  DecodeE4M3(LoadByte(Head, PackedBIndex(tile,k,column % 16u))), accumulator);
            accumulatorBits = FloatToHalfRN(accumulator);
        }
        outputWord |= accumulatorBits << (j * 16u);
    }
    First128.Store(base * 2u, outputWord);
}
