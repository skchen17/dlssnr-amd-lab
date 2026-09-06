Texture2D<float4> sourceTexture : register(t0);
RWTexture2D<float4> outputTexture : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    sourceTexture.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    int2 p = int2(id.xy);
    int2 hi = int2(width - 1, height - 1);
    float4 center = sourceTexture.Load(int3(p, 0));
    float3 laplacian = center.rgb * 4.0
        - sourceTexture.Load(int3(clamp(p + int2(-1, 0), 0, hi), 0)).rgb
        - sourceTexture.Load(int3(clamp(p + int2( 1, 0), 0, hi), 0)).rgb
        - sourceTexture.Load(int3(clamp(p + int2(0, -1), 0, hi), 0)).rgb
        - sourceTexture.Load(int3(clamp(p + int2(0,  1), 0, hi), 0)).rgb;
    // A bounded residual layer used only to validate dynamic neural-output plumbing.
    outputTexture[id.xy] = float4(max(center.rgb + 0.03125 * laplacian, 0.0), center.a);
}
