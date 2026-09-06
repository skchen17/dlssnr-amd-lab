ByteAddressBuffer Residual : register(t0);
ByteAddressBuffer BaseSurface : register(t1);
RWByteAddressBuffer OutputSurface : register(u0);

static const uint GridWidth = 81;
static const uint GridHeight = 49;
static const uint Tokens = 64;
static const uint Width = 640;
static const uint Height = 360;
static const uint WorkItems = GridWidth * GridHeight * Tokens;

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

uint PackHalves(float low, float high)
{
    return FloatToHalfRN(low) | (FloatToHalfRN(high) << 16);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    if (index >= WorkItems)
        return;

    uint token = index % Tokens;
    uint cta = index / Tokens;
    uint group = token / 16;
    uint within = token % 16;
    int x = int((cta % GridWidth) * 8) - 4 +
            int((group & 1) * 4 + within % 4);
    int y = int((cta / GridWidth) * 8) - 4 +
            int((group >> 1) * 4 + within / 4);
    if (x < 0 || y < 0 || x >= int(Width) || y >= int(Height))
        return;

    uint residualOffset = index * 8;
    uint outputOffset = (uint(y) * Width + uint(x)) * 8;
    uint2 residual = Residual.Load2(residualOffset);
    uint2 base = BaseSurface.Load2(outputOffset);
    float r = clamp(HalfValue(base.x) + 0.25f * HalfValue(residual.x), 0.0f, 1.0f);
    float g = clamp(HalfValue(base.x >> 16) + 0.25f * HalfValue(residual.x >> 16), 0.0f, 1.0f);
    float b = clamp(HalfValue(base.y) + 0.25f * HalfValue(residual.y), 0.0f, 1.0f);
    OutputSurface.Store2(outputOffset, uint2(PackHalves(r, g),
                                              PackHalves(b, 1.0f)));
}
