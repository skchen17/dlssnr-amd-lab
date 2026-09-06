#include "output_head_full_common.hlsli"
ByteAddressBuffer QkScores : register(t0);
ByteAddressBuffer V : register(t1);
RWByteAddressBuffer Attention : register(u0);
static const uint QueryWorkItems = 81u * 49u * 64u;

uint ScoreOffset(uint cta, uint query, uint key)
{
    uint group = query / 16u, row = query % 16u;
    return cta * 4096u + (group * 8u + key / 8u) * 128u + row * 8u + key % 8u;
}

[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x; if (index >= QueryWorkItems) return;
    uint query = index % 64u, cta = index / 64u;
    float maximum = -asfloat(0x7f800000u);
    for (uint key = 0u; key < 64u; ++key)
        maximum = max(maximum, f16tof32(LoadHalf(QkScores, ScoreOffset(cta,query,key)*2u)));
    float denominator = 0.0f;
    for (uint key = 0u; key < 64u; ++key)
        denominator += exp(f16tof32(LoadHalf(QkScores, ScoreOffset(cta,query,key)*2u)) - maximum);
    uint probabilities[64];
    for (uint packedKey = 0u; packedKey < 64u; ++packedKey) {
        uint key = (packedKey / 32u) * 32u + Permute32(packedKey % 32u);
        float p = exp(f16tof32(LoadHalf(QkScores, ScoreOffset(cta,query,key)*2u)) - maximum) / denominator;
        probabilities[packedKey] = EncodeE4M3(FloatToHalfRN(p));
    }
    uint outputBase = index * 64u;
    for (uint featurePair = 0u; featurePair < 16u; ++featurePair) {
        uint word = 0u;
        [unroll] for (uint pair = 0u; pair < 2u; ++pair) {
            uint feature = featurePair * 2u + pair; float accumulator = 0.0f;
            for (uint key = 0u; key < 32u; ++key)
                accumulator = mad(DecodeE4M3(probabilities[key]),
                                  DecodeE4M3(LoadByte(V,(cta*64u+key)*32u+feature)), accumulator);
            accumulator = f16tof32(FloatToHalfRN(accumulator));
            for (uint key = 32u; key < 64u; ++key)
                accumulator = mad(DecodeE4M3(probabilities[key]),
                                  DecodeE4M3(LoadByte(V,(cta*64u+key)*32u+feature)), accumulator);
            word |= FloatToHalfRN(accumulator) << (pair * 16u);
        }
        Attention.Store(outputBase + featurePair * 4u, word);
    }
}
