Texture2D<float4> sourceTexture : register(t0);
RWTexture2D<float4> outputTexture : register(u0);

// 3x3 RGB -> 8 ReLU features -> RGB residual.  The 256-float constant
// buffer is loaded from a separate, auditable model file by the host.
cbuffer ResidualModel : register(b0) {
    float4 convWeights0[27];
    float4 convWeights1[27];
    float4 hiddenBias0;
    float4 hiddenBias1;
    float4 outputWeights0[3];
    float4 outputWeights1[3];
    float4 outputBias;
    float4 modelParams;
};

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    sourceTexture.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;

    int2 p = int2(id.xy);
    int2 hi = int2(width - 1, height - 1);
    float4 h0 = hiddenBias0;
    float4 h1 = hiddenBias1;
    [unroll]
    for (int oy = -1; oy <= 1; ++oy) {
        [unroll]
        for (int ox = -1; ox <= 1; ++ox) {
            const int sampleIndex = (oy + 1) * 3 + ox + 1;
            float3 value = sourceTexture.Load(int3(clamp(p + int2(ox, oy), 0, hi), 0)).rgb;
            [unroll]
            for (int channel = 0; channel < 3; ++channel) {
                const int featureIndex = sampleIndex * 3 + channel;
                h0 += value[channel] * convWeights0[featureIndex];
                h1 += value[channel] * convWeights1[featureIndex];
            }
        }
    }
    h0 = max(h0, 0.0);
    h1 = max(h1, 0.0);
    float3 residual;
    [unroll]
    for (int channel = 0; channel < 3; ++channel)
        residual[channel] = dot(h0, outputWeights0[channel]) +
                            dot(h1, outputWeights1[channel]) + outputBias[channel];

    float4 center = sourceTexture.Load(int3(p, 0));
    outputTexture[id.xy] = float4(max(center.rgb + modelParams.x * residual, 0.0), center.a);
}
