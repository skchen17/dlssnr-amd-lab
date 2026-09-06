#include "output_head_full_common.hlsli"
ByteAddressBuffer Projection : register(t0);
RWByteAddressBuffer V : register(u0);
static const uint Values = 81u * 49u * 64u * 32u;

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 4u; if (base >= Values) return;
    uint featureBase = base % 32u, split = base / 32u;
    uint destinationKey = split % 64u, cta = split / 64u;
    uint keyChunk = destinationKey / 32u;
    uint sourceToken = keyChunk * 32u + Permute32(destinationKey % 32u);
    uint word = 0u;
    [unroll] for (uint j = 0u; j < 4u; ++j) {
        uint bits = LoadHalf(Projection, ProjectionIndex(cta,2u,sourceToken,featureBase+j) * 2u);
        word |= EncodeE4M3(bits) << (j * 8u);
    }
    V.Store(base, word);
}
