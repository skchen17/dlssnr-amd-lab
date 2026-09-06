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
Texture2DArray<float4> networkOutputs : register(t1);
RWTexture2D<float4> frameOutput : register(u0);

// Tiles overlap so every selected pixel has context.  Midpoint ownership from
// ResolutionPlan guarantees that this pass writes every frame pixel once.
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 640 || id.y >= 360 || id.z >= batchTileCount) return;
    TileDescriptor tile = tiles[tileBase + id.z];
    uint2 framePixel = uint2(tile.originX + id.x, tile.originY + id.y);
    if (framePixel.x >= frameWidth || framePixel.y >= frameHeight) return;
    if (framePixel.x < tile.writeBeginX || framePixel.x >= tile.writeEndX ||
        framePixel.y < tile.writeBeginY || framePixel.y >= tile.writeEndY) return;
    frameOutput[framePixel] = networkOutputs.Load(int4(id.xy, id.z, 0));
}
