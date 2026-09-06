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
Texture2D<float4> frameInput : register(t1);
RWTexture2DArray<float4> networkTiles : register(u0);

// Packs an arbitrary game surface into 640x384 compatibility tensors.  Border
// samples are clamped, including the 12 rows of context above and below the
// 360-row output region.
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 640 || id.y >= 384 || id.z >= batchTileCount) return;
    TileDescriptor tile = tiles[tileBase + id.z];
    int sourceX = int(tile.originX) + int(id.x);
    int sourceY = int(tile.originY) + int(id.y) - int(verticalHalo);
    sourceX = clamp(sourceX, 0, int(frameWidth) - 1);
    sourceY = clamp(sourceY, 0, int(frameHeight) - 1);
    networkTiles[id] = frameInput.Load(int3(sourceX, sourceY, 0));
}
