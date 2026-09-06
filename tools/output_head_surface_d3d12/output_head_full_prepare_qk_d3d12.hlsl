#include "output_head_full_common.hlsli"
ByteAddressBuffer Projection : register(t0);
ByteAddressBuffer Head : register(t1);
RWByteAddressBuffer Q : register(u0);
RWByteAddressBuffer K : register(u1);
static const uint WorkItems = 81u * 49u * 64u;

uint HalfMul(uint a, uint b) { return FloatToHalfRN(f16tof32(a) * f16tof32(b)); }
uint HalfAdd(uint a, uint b) { return FloatToHalfRN(f16tof32(a) + f16tof32(b)); }

[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x; if (index >= WorkItems) return;
    uint token = index % 64u, cta = index / 64u;
    uint qValues[32], kValues[32], qSquares[32], kSquares[32];
    for (uint channel = 0u; channel < 32u; ++channel) {
        qValues[channel] = LoadHalf(Projection, ProjectionIndex(cta,0u,token,channel) * 2u);
        kValues[channel] = LoadHalf(Projection, ProjectionIndex(cta,1u,token,channel) * 2u);
        qSquares[channel] = HalfMul(qValues[channel], qValues[channel]);
        kSquares[channel] = HalfMul(kValues[channel], kValues[channel]);
    }
    for (uint width = 16u; width != 0u; width >>= 1u)
        for (uint i = 0u; i < width; ++i) {
            qSquares[i] = HalfAdd(qSquares[2u*i], qSquares[2u*i+1u]);
            kSquares[i] = HalfAdd(kSquares[2u*i], kSquares[2u*i+1u]);
        }
    uint epsilonBits = FloatToHalfRN(6.199999916134402e-05f);
    uint qInverse = FloatToHalfRN(rsqrt(max(f16tof32(qSquares[0]), f16tof32(epsilonBits))));
    uint kInverse = FloatToHalfRN(rsqrt(max(f16tof32(kSquares[0]), f16tof32(epsilonBits))));
    uint scaleBits = FloatToHalfRN(asfloat(Head.Load(uint(19664 + NR_ATTENTION_WEIGHT_SHIFT))));
    uint outputBase = index * 32u;
    for (uint block = 0u; block < 8u; ++block) {
        uint qWord = 0u, kWord = 0u;
        [unroll] for (uint j = 0u; j < 4u; ++j) {
            uint packedChannel = block * 4u + j, source = Permute32(packedChannel);
            uint qNormalized = HalfMul(HalfMul(qValues[source], qInverse), scaleBits);
            uint kNormalized = HalfMul(kValues[source], kInverse);
            qWord |= EncodeE4M3(qNormalized) << (j * 8u);
            kWord |= EncodeE4M3(kNormalized) << (j * 8u);
        }
        Q.Store(outputBase + block * 4u, qWord); K.Store(outputBase + block * 4u, kWord);
    }
}
