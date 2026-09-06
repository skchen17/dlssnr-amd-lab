#include "../output_head_surface_d3d12/output_head_full_common.hlsli"
ByteAddressBuffer Projected : register(t0);
RWByteAddressBuffer Output : register(u0);
cbuffer Config : register(b0) { uint Width, Height, GridX, GridY; int OriginX, OriginY; };

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint base = id.x * 4; if (base >= GridX * GridY * 2048) return;
    uint cta = base / 2048, local = base % 2048, cell = local / 512;
    int tileX = (int(cta % GridX) * 8 + OriginX) / 4 + int(cell % 2);
    int tileY = (int(cta / GridX) * 8 + OriginY) / 4 + int(cell / 2);
    if (tileX < 0 || tileX >= int(Width / 4) || tileY < 0 || tileY >= int(Height / 4)) return;
    uint lane = (local % 512) / 16, packed = 0;
    [unroll] for (uint j = 0; j < 4; ++j) {
        uint element = local % 16 + j;
        uint row = lane / 4 + ((element % 8) / 4) * 8;
        uint column = (lane % 4) * 2 + ((element % 4) / 2) * 8 + (element / 8) * 16 + (element & 1);
        uint h = LoadHalf(Projected, ((cta * 64 + cell * 16 + row) * 32 + column) * 2);
        packed |= EncodeE4M3(h) << (j * 8);
    }
    Output.Store((uint(tileY) * (Width / 4) + uint(tileX)) * 512 + local % 512, packed);
}
