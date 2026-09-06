#include "output_head_full_common.hlsli"
ByteAddressBuffer MainFeature : register(t0);
ByteAddressBuffer SkipFeature : register(t1);
ByteAddressBuffer Head : register(t2);
RWByteAddressBuffer FusedE4 : register(u0);
RWByteAddressBuffer FusedHalf : register(u1);
static const uint GridValues = 81u * 49u * 2048u;

uint ShuffledSourceLane(uint lane, uint output)
{
    uint q = (lane & 19u) | ((lane << 1u) & 8u) | ((lane >> 1u) & 4u);
    uint4 sources = uint4(q, q ^ 8u, q ^ 16u, q ^ 24u);
    uint4 middle = (lane & 16u) != 0u ? sources.zwxy : sources;
    uint4 swapped = middle.yxwz;
    return (lane & 4u) == 0u ? middle[output] : swapped[output];
}

uint FuseValue(uint ctaX, uint ctaY, uint operation, uint lane, uint pairHalf)
{
    uint cell = operation / 8u, within = operation % 8u, sourceSet = operation < 16u ? 0u : 1u;
    static const uint sourceBases[8] = {0u, 1u, 4u, 5u, 2u, 3u, 6u, 7u};
    uint source = sourceBases[sourceSet * 4u + within / 2u];
    uint parity = (operation % 16u) / 8u;
    uint sourceLane = ShuffledSourceLane(lane, parity + ((within & 1u) != 0u ? 2u : 0u));
    uint load = source / 2u, plane = load / 2u;
    int mainBaseY = (-4 + int(ctaY) * 8) / 2, mainBaseX = (-4 + int(ctaX) * 8) / 2;
    int y = mainBaseY + int(sourceLane / 16u) + ((load & 1u) != 0u ? 2 : 0);
    int x = mainBaseX + int((sourceLane / 4u) % 4u);
    uint mainCode = 0u;
    if (y >= 0 && y < 192 && x >= 0 && x < 320) {
        uint mainIndex = ((plane * 192u + uint(y)) * 320u + uint(x)) * 16u +
                         (sourceLane % 4u) * 4u + (source & 1u) * 2u + pairHalf;
        mainCode = LoadByte(MainFeature, mainIndex);
    }
    static const uint byteOrder[8] = {0u, 4u, 2u, 6u, 8u, 12u, 10u, 14u};
    int skipBaseY = (-4 + int(ctaY) * 8) / 4, skipBaseX = (-4 + int(ctaX) * 8) / 4;
    int skipY = skipBaseY + int(cell / 2u), skipX = skipBaseX + int(cell % 2u);
    uint skipCode = 0u;
    if (skipY >= 0 && skipY < 96 && skipX >= 0 && skipX < 160) {
        uint skipIndex = (uint(skipY) * 160u + uint(skipX)) * 512u + lane * 16u + byteOrder[within] + pairHalf;
        skipCode = LoadByte(SkipFeature, skipIndex);
    }
    uint scaleOffset = ((operation / 2u) % 4u) * 16u + (lane % 4u) * 4u + pairHalf * 2u;
    float mainScale = f16tof32(LoadHalf(Head, 8272u + scaleOffset));
    float skipScale = f16tof32(LoadHalf(Head, 8336u + scaleOffset));
    float roundedMain = f16tof32(FloatToHalfRN(DecodeE4M3(mainCode) * mainScale));
    return FloatToHalfRN(mad(DecodeE4M3(skipCode), skipScale, roundedMain));
}

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 4u; if (base >= GridValues) return;
    uint e4Word = 0u, half0 = 0u, half1 = 0u;
    static const uint conversionOrder[8] = {0u, 2u, 1u, 3u, 4u, 6u, 5u, 7u};
    [unroll] for (uint j = 0u; j < 4u; ++j) {
        uint index = base + j, cta = index / 2048u, local = index % 2048u;
        uint pairHalf = local & 1u, packedWithin = (local >> 1u) % 8u;
        uint lane = (local >> 4u) % 32u, cell = local / 512u;
        uint operation = cell * 8u + conversionOrder[packedWithin];
        uint bits = FuseValue(cta % 81u, cta / 81u, operation, lane, pairHalf);
        e4Word |= EncodeE4M3(bits) << (j * 8u);
        if (j < 2u) half0 |= bits << (j * 16u); else half1 |= bits << ((j - 2u) * 16u);
    }
    FusedE4.Store(base, e4Word); FusedHalf.Store(base * 2u, half0); FusedHalf.Store(base * 2u + 4u, half1);
}
