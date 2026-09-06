struct TileDescriptor {
    uint originX;
    uint originY;
    uint writeBeginX;
    uint writeBeginY;
    uint writeEndX;
    uint writeEndY;
};

cbuffer ResolutionConstants : register(b0) {
    uint frameWidth;
    uint frameHeight;
    uint batchTileCount;
    uint verticalHalo;
    uint tileBase;
};
StructuredBuffer<TileDescriptor> tiles : register(t0);
Texture2DArray<float4> networkWork : register(t1);
RWTexture2DArray<float4> networkOutputs : register(u0);

// Identity stand-in used only by the boundary self-test. The real graph writes
// the same 640x360 output-tile contract after consuming the 640x384 work tile.
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 640 || id.y >= 360 || id.z >= batchTileCount) return;
    networkOutputs[id] = networkWork.Load(int4(id.x, id.y + verticalHalo, id.z, 0));
}
