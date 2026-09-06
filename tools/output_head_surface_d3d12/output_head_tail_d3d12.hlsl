ByteAddressBuffer Projected : register(t0);
ByteAddressBuffer HeadWeights : register(t1);
RWByteAddressBuffer Residual : register(u0);

static const uint GridWidth = 81;
static const uint GridHeight = 49;
static const uint Tokens = 64;
static const uint Channels = 32;
static const uint WorkItems = GridWidth * GridHeight * Tokens;

uint LoadProjectedHalf(uint byteOffset)
{
    uint word = Projected.Load(byteOffset & ~3u);
    return (word >> ((byteOffset & 2u) * 8u)) & 0xffffu;
}

uint LoadWeightHalf(uint byteOffset)
{
    uint word = HeadWeights.Load(byteOffset & ~3u);
    return (word >> ((byteOffset & 2u) * 8u)) & 0xffffu;
}

float HalfValue(uint bits)
{
    return f16tof32(bits & 0xffffu);
}

uint FloatToHalfRN(float value)
{
    uint bits = asuint(value);
    uint sign = (bits >> 16) & 0x8000u;
    uint exponent = (bits >> 23) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return sign | (mantissa != 0 ? 0x7e00u : 0x7c00u);
    int halfExponent = int(exponent) - 127 + 15;
    if (halfExponent >= 31)
        return sign | 0x7c00u;
    if (halfExponent <= 0) {
        if (halfExponent < -10)
            return sign;
        mantissa |= 0x800000u;
        uint shift = uint(14 - halfExponent);
        uint rounded = mantissa >> shift;
        uint remainder = mantissa & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
            ++rounded;
        return sign | rounded;
    }
    uint result = sign | (uint(halfExponent) << 10) | (mantissa >> 13);
    uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u)))
        ++result;
    return result;
}

uint TailWeight(uint k, uint channel)
{
    uint tile = 20784u + (k / 16u) * 512u;
    uint kk = k % 16u;
    uint lane = channel * 4u + (kk % 8u) / 2u;
    uint element = (kk % 2u) + (kk >= 8u ? 2u : 0u);
    return LoadWeightHalf(tile + lane * 16u + element * 2u);
}

uint Project(uint cta, uint token, uint channel)
{
    float accumulator = 0.0f;
    uint baseElement = (cta * Tokens + token) * Channels;

    // Match the gfx1201 HIP lowering: packed pairs are consumed low-half first.
    for (uint k = 0; k < 16; k += 2) {
        float inputHigh = HalfValue(LoadProjectedHalf((baseElement + k + 1u) * 2u));
        float inputLow = HalfValue(LoadProjectedHalf((baseElement + k) * 2u));
        accumulator = mad(inputLow, HalfValue(TailWeight(k, channel)), accumulator);
        accumulator = mad(inputHigh, HalfValue(TailWeight(k + 1u, channel)), accumulator);
    }
    accumulator = HalfValue(FloatToHalfRN(accumulator));

    // The HIP compiler keeps the first two pairs as mix-FMAs, then lowers the
    // remaining six pair expressions to v_dot2_f32_f16 on gfx1201.
    for (uint k = 16; k < 20; k += 2) {
        float inputHigh = HalfValue(LoadProjectedHalf((baseElement + k + 1u) * 2u));
        float inputLow = HalfValue(LoadProjectedHalf((baseElement + k) * 2u));
        accumulator = mad(inputLow, HalfValue(TailWeight(k, channel)), accumulator);
        accumulator = mad(inputHigh, HalfValue(TailWeight(k + 1u, channel)), accumulator);
    }
    for (uint k = 20; k < 32; k += 2) {
        float inputLow = HalfValue(LoadProjectedHalf((baseElement + k) * 2u));
        float inputHigh = HalfValue(LoadProjectedHalf((baseElement + k + 1u) * 2u));
        float weightLow = HalfValue(TailWeight(k, channel));
        float weightHigh = HalfValue(TailWeight(k + 1u, channel));
        accumulator += inputLow * weightLow + inputHigh * weightHigh;
    }
    return FloatToHalfRN(accumulator);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    if (index >= WorkItems)
        return;
    uint token = index % Tokens;
    uint cta = index / Tokens;
    uint r = Project(cta, token, 0);
    uint g = Project(cta, token, 1);
    uint b = Project(cta, token, 2);
    uint a = Project(cta, token, 3);
    Residual.Store2(index * 8u, uint2(r | (g << 16), b | (a << 16)));
}
