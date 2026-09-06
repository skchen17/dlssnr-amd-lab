#ifndef NR_ATTENTION_WEIGHT_SHIFT
#define NR_ATTENTION_WEIGHT_SHIFT 0
#endif
ByteAddressBuffer Attention : register(t0);
ByteAddressBuffer Residual : register(t1);
ByteAddressBuffer HeadWeights : register(t2);
RWByteAddressBuffer Projected : register(u0);

static const uint GridWidth = 81;
static const uint GridHeight = 49;
static const uint Tokens = 64;
static const uint Channels = 32;
static const uint OutputValues = GridWidth * GridHeight * Tokens * Channels;

uint LoadHalf(ByteAddressBuffer source, uint byteOffset)
{
    uint word = source.Load(byteOffset & ~3u);
    return (word >> ((byteOffset & 2u) * 8u)) & 0xffffu;
}

uint LoadHeadByte(uint byteOffset)
{
    uint word = HeadWeights.Load(byteOffset & ~3u);
    return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
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

uint RoundShiftRN(uint value, uint shift)
{
    uint base = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return base + uint(remainder > halfway || (remainder == halfway && (base & 1u)));
}

uint EncodeE4M3(uint halfBits)
{
    uint sign = (halfBits >> 8) & 0x80u;
    uint exponent = (halfBits >> 10) & 31u;
    uint mantissa = halfBits & 1023u;
    uint code;
    if (exponent == 31u && mantissa != 0u)
        return 0x7fu;
    if (exponent < 9u) {
        code = min(RoundShiftRN(1024u + mantissa, 16u - exponent), 8u);
    } else {
        uint rounded = RoundShiftRN(mantissa, 7u);
        bool carry = rounded >= 8u;
        code = ((exponent - 8u + uint(carry)) << 3) | (carry ? 0u : rounded);
        code = min(code, 126u);
    }
    return sign | code;
}

float DecodeE4M3(uint bits)
{
    uint magnitude = bits & 0x7fu;
    uint exponent = magnitude >> 3;
    uint mantissa = magnitude & 7u;
    float value = exponent == 0u
        ? float(mantissa) * exp2(-9.0f)
        : (magnitude == 127u ? asfloat(0x7fc00000u)
                             : (1.0f + float(mantissa) / 8.0f) * exp2(float(int(exponent) - 7)));
    return (bits & 0x80u) != 0u ? -value : value;
}

uint PermutedChannel(uint channel)
{
    uint block = channel / 16u;
    uint within = channel % 16u;
    return block * 16u + 2u * (within / 4u) + (within & 1u) + ((within & 2u) != 0u ? 8u : 0u);
}

uint WeightOffset(uint tile, uint k, uint column)
{
    uint lane = column * 4u + (k % 16u) / 4u;
    uint element = (k % 4u) + (k >= 16u ? 4u : 0u);
    return tile + lane * 16u + element;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    if (index >= OutputValues)
        return;
    uint outputChannel = index % Channels;
    uint tokenIndex = index / Channels;
    uint token = tokenIndex % Tokens;
    uint group = token / 16u;
    uint row = token % 16u;
    uint cta = tokenIndex / Tokens;
    uint baseElement = (cta * Tokens + token) * Channels;

    uint scaleBits = LoadHalf(HeadWeights, uint(20704 + NR_ATTENTION_WEIGHT_SHIFT) + outputChannel * 2u);
    float residualValue = HalfValue(LoadHalf(Residual, (baseElement + outputChannel) * 2u));
    float scaleValue = HalfValue(scaleBits);
    float accumulator = HalfValue(FloatToHalfRN(residualValue * scaleValue));

    uint tile = uint(19680 + NR_ATTENTION_WEIGHT_SHIFT) + (outputChannel / 16u) * 512u;
    uint column = outputChannel % 8u;
    uint nibbleHalf = (outputChannel % 16u) / 8u;
    for (uint k = 0; k < Channels; ++k) {
        uint sourceChannel = PermutedChannel(k);
        uint sourceBits = LoadHalf(Attention, (baseElement + sourceChannel) * 2u);
        float activation = DecodeE4M3(EncodeE4M3(sourceBits));
        uint weightBits = LoadHeadByte(WeightOffset(tile, k, column) + nibbleHalf * 8u);
        accumulator = mad(activation, DecodeE4M3(weightBits), accumulator);
    }
    uint outputBits = FloatToHalfRN(accumulator);
    uint byteOffset = index * 2u;
    uint address = byteOffset & ~3u;
    uint oldWord = Projected.Load(address);
    uint shift = (byteOffset & 2u) * 8u;
    uint mask = 0xffffu << shift;
    // Two adjacent threads share a dword, so use an atomic masked update.
    uint ignored;
    Projected.InterlockedAnd(address, ~mask, ignored);
    Projected.InterlockedOr(address, outputBits << shift, ignored);
}
