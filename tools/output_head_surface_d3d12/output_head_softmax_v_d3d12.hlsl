ByteAddressBuffer QkScores : register(t0);
ByteAddressBuffer PackedV : register(t1);
RWByteAddressBuffer Attention : register(u0);

static const uint GridWidth = 81;
static const uint GridHeight = 49;
static const uint Tokens = 64;
static const uint Channels = 32;
static const uint QueryWorkItems = GridWidth * GridHeight * Tokens;
static const uint QkValuesPerCta = 32 * 16 * 8;
static const uint PackedVPerCta = 64 * 32;
static const uint AttentionValuesPerCta = 64 * 32;

uint LoadHalf(ByteAddressBuffer source, uint byteOffset)
{
    uint word = source.Load(byteOffset & ~3u);
    return (word >> ((byteOffset & 2u) * 8u)) & 0xffffu;
}

uint LoadByte(ByteAddressBuffer source, uint byteOffset)
{
    uint word = source.Load(byteOffset & ~3u);
    return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

uint FloatToHalfRN(float value)
{
    uint bits = asuint(value);
    uint sign = (bits >> 16) & 0x8000u;
    uint exponent = (bits >> 23) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return sign | (mantissa != 0u ? 0x7e00u : 0x7c00u);
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

uint ScoreOffset(uint cta, uint query, uint key)
{
    uint group = query / 16u;
    uint row = query % 16u;
    return cta * QkValuesPerCta + (group * 8u + key / 8u) * 128u + row * 8u + key % 8u;
}

uint PackedVOffset(uint cta, uint key, uint feature)
{
    uint keyChunk = key / 32u;
    uint localKey = key % 32u;
    uint featureBlock = feature / 8u;
    uint column = feature % 8u;
    uint lane = column * 4u + (localKey % 16u) / 4u;
    uint element = (localKey % 4u) + (localKey >= 16u ? 4u : 0u);
    return cta * PackedVPerCta + (keyChunk * 4u + featureBlock) * 256u + lane * 8u + element;
}

uint Permute32(uint channel)
{
    uint block = channel / 16u;
    uint within = channel % 16u;
    return block * 16u + 2u * (within / 4u) + (within & 1u) + ((within & 2u) != 0u ? 8u : 0u);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    if (index >= QueryWorkItems)
        return;
    uint query = index % Tokens;
    uint cta = index / Tokens;

    float maximum = -asfloat(0x7f800000u);
    for (uint key = 0; key < Tokens; ++key)
        maximum = max(maximum, f16tof32(LoadHalf(QkScores, ScoreOffset(cta, query, key) * 2u)));

    float denominator = 0.0f;
    for (uint key = 0; key < Tokens; ++key)
        denominator += exp(f16tof32(LoadHalf(QkScores, ScoreOffset(cta, query, key) * 2u)) - maximum);

    uint probabilities[64];
    for (uint packedKey = 0; packedKey < Tokens; ++packedKey) {
        uint chunk = packedKey / 32u;
        uint key = chunk * 32u + Permute32(packedKey % 32u);
        float probability = exp(f16tof32(LoadHalf(QkScores, ScoreOffset(cta, query, key) * 2u)) - maximum) / denominator;
        probabilities[packedKey] = EncodeE4M3(FloatToHalfRN(probability));
    }

    uint outputBase = (cta * AttentionValuesPerCta + query * Channels) * 2u;
    for (uint featurePair = 0; featurePair < Channels / 2u; ++featurePair) {
        uint packedOutput = 0u;
        for (uint pair = 0; pair < 2u; ++pair) {
            uint feature = featurePair * 2u + pair;
            float accumulator = 0.0f;
            for (uint key = 0; key < 32u; ++key)
                accumulator = mad(DecodeE4M3(probabilities[key]),
                                  DecodeE4M3(LoadByte(PackedV, PackedVOffset(cta, key, feature))), accumulator);
            accumulator = f16tof32(FloatToHalfRN(accumulator));
            for (uint key = 32u; key < Tokens; ++key)
                accumulator = mad(DecodeE4M3(probabilities[key]),
                                  DecodeE4M3(LoadByte(PackedV, PackedVOffset(cta, key, feature))), accumulator);
            packedOutput |= FloatToHalfRN(accumulator) << (pair * 16u);
        }
        Attention.Store(outputBase + featurePair * 4u, packedOutput);
    }
}
