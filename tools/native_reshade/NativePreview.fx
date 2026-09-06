// Identity technique solely to establish ReShade's documented effect-chain boundary.
texture2D FrameColor : COLOR;
sampler2D FrameSampler { Texture = FrameColor; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; };
void VS(uint id : SV_VertexID, out float4 position : SV_Position, out float2 uv : TEXCOORD)
{
    uv = float2((id << 1) & 2, id & 2);
    position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 PS(float4 position : SV_Position, float2 uv : TEXCOORD) : SV_Target { return tex2D(FrameSampler, uv); }
technique NativePreviewCopy < enabled = true; >
{
    pass { VertexShader = VS; PixelShader = PS; }
}
