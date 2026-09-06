Texture2D<float4> SourceTexture : register(t0);
SamplerState SourceSampler : register(s0);
RWTexture2D<float4> DestinationSurface : register(u0);

cbuffer CopyParameters : register(b0)
{
    float2 SourceOffset;
    float2 SourceExtent;
    float2 SourceInverseSize;
    float2 DestinationOffset;
    uint2 CopyDimensions;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= CopyDimensions.x ||
        dispatchThreadId.y >= CopyDimensions.y)
        return;

    float2 pixel = float2(dispatchThreadId.xy) + 0.5f;
    float2 normalized = pixel / float2(CopyDimensions);
    float2 uv = (SourceOffset + SourceExtent * normalized) * SourceInverseSize;
    int2 destination = int2(DestinationOffset) + int2(dispatchThreadId.xy);
    DestinationSurface[destination] = SourceTexture.SampleLevel(SourceSampler, uv, 0.0f);
}
