#ifndef NR_ATTENTION_WEIGHT_SHIFT
#define NR_ATTENTION_WEIGHT_SHIFT 0
#endif

uint LoadByte(ByteAddressBuffer source, uint offset)
{
    uint word = source.Load(offset & ~3u);
    return (word >> ((offset & 3u) * 8u)) & 0xffu;
}

uint LoadHalf(ByteAddressBuffer source, uint offset)
{
    uint word = source.Load(offset & ~3u);
    return (word >> ((offset & 2u) * 8u)) & 0xffffu;
}

uint FloatToHalfRN(float value)
{
    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u;
    uint exponent = (bits >> 23) & 0xffu, mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return sign | (mantissa != 0u ? 0x7e00u : 0x7c00u);
    int halfExponent = int(exponent) - 112;
    if (halfExponent >= 31) return sign | 0x7c00u;
    if (halfExponent <= 0) {
        if (halfExponent < -10) return sign;
        mantissa |= 0x800000u;
        uint shift = uint(14 - halfExponent), rounded = mantissa >> shift;
        uint remainder = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return sign | rounded;
    }
    uint result = sign | (uint(halfExponent) << 10) | (mantissa >> 13);
    uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u))) ++result;
    return result;
}

uint RoundShiftRN(uint value, uint shift)
{
    uint base = value >> shift, remainder = value & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
    return base + uint(remainder > halfway || (remainder == halfway && (base & 1u)));
}

uint EncodeE4M3(uint halfBits)
{
    uint sign = (halfBits >> 8) & 0x80u, exponent = (halfBits >> 10) & 31u, mantissa = halfBits & 1023u;
    if (exponent == 31u && mantissa != 0u) return 0x7fu;
    uint code;
    if (exponent < 9u) code = min(RoundShiftRN(1024u + mantissa, 16u - exponent), 8u);
    else {
        uint rounded = RoundShiftRN(mantissa, 7u); bool carry = rounded >= 8u;
        code = ((exponent - 8u + uint(carry)) << 3) | (carry ? 0u : rounded);
        code = min(code, 126u);
    }
    return sign | code;
}

float DecodeE4M3(uint bits)
{
    uint magnitude = bits & 0x7fu, exponent = magnitude >> 3, mantissa = magnitude & 7u;
    float value = exponent == 0u ? float(mantissa) * exp2(-9.0f) :
        (magnitude == 127u ? asfloat(0x7fc00000u) :
         (1.0f + float(mantissa) * 0.125f) * exp2(float(int(exponent) - 7)));
    return (bits & 0x80u) != 0u ? -value : value;
}

uint Permute32(uint channel)
{
    uint block = channel / 16u, within = channel % 16u;
    return block * 16u + 2u * (within / 4u) + (within & 1u) + ((within & 2u) != 0u ? 8u : 0u);
}

uint PackedAIndex(uint group, uint row, uint k)
{
    uint lane = (row % 8u) * 4u + (k % 16u) / 4u;
    uint element = (k % 4u) + (row >= 8u ? 4u : 0u) + (k >= 16u ? 8u : 0u);
    return group * 512u + lane * 16u + element;
}

uint PackedBIndex(uint tile, uint k, uint column)
{
    uint lane = (column % 8u) * 4u + (k % 16u) / 4u;
    uint element = (k % 4u) + (k >= 16u ? 4u : 0u);
    return tile + lane * 16u + (column / 8u) * 8u + element;
}

uint ProjectionIndex(uint cta, uint segment, uint token, uint channel)
{
    uint group = token / 16u, row = token % 16u;
    uint mma = group * 12u + segment * 4u + (channel / 16u) * 2u + (channel % 16u) / 8u;
    return cta * 6144u + mma * 128u + row * 8u + channel % 8u;
}
