#include "output_head_full_common.hlsli"
ByteAddressBuffer Q : register(t0);
ByteAddressBuffer K : register(t1);
ByteAddressBuffer Head : register(t2);
RWByteAddressBuffer Qk : register(u0);
static const uint Values = 81u * 49u * 32u * 16u * 8u;

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 2u; if (base >= Values) return;
    uint split = base / 8u, row = split % 16u; split /= 16u;
    uint mma = split % 32u, cta = split / 32u;
    uint queryGroup = mma / 8u, keyBlock = mma % 8u, query = queryGroup * 16u + row;
    uint outputWord = 0u;
    [unroll] for (uint j = 0u; j < 2u; ++j) {
        uint column = (base % 8u) + j, key = keyBlock * 8u + column;
        uint seedTile = uint(11472 + NR_ATTENTION_WEIGHT_SHIFT) + queryGroup * 2048u + (keyBlock / 2u) * 512u;
        uint lane = (row % 8u) * 4u + column / 2u;
        uint element = (row >= 8u ? 2u : 0u) + (column & 1u);
        uint seedOffset = seedTile + lane * 16u + (keyBlock & 1u) * 8u + element * 2u;
        float accumulator = f16tof32(LoadHalf(Head, seedOffset));
        uint qBase = (cta * 64u + query) * 32u, kBase = (cta * 64u + key) * 32u;
        for (uint channel = 0u; channel < 32u; ++channel)
            accumulator = mad(DecodeE4M3(LoadByte(Q,qBase+channel)),
                              DecodeE4M3(LoadByte(K,kBase+channel)), accumulator);
        outputWord |= FloatToHalfRN(accumulator) << (j * 16u);
    }
    Qk.Store(base * 2u, outputWord);
}
