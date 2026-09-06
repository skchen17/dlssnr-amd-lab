#include "../output_head_surface_d3d12/output_head_full_common.hlsli"
ByteAddressBuffer Input : register(t0);
RWByteAddressBuffer RawE4 : register(u0);
RWByteAddressBuffer RawHalf : register(u1);
cbuffer Config : register(b0) { uint Width, Height, GridX, GridY; int OriginX, OriginY; };

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint base = id.x * 4; if (base >= GridX * GridY * 2048) return;
    uint cta = base / 2048, local = base % 2048, cell = local / 512;
    int tileX = (int(cta % GridX) * 8 + OriginX) / 4 + int(cell % 2);
    int tileY = (int(cta / GridX) * 8 + OriginY) / 4 + int(cell / 2);
    uint packed = 0;
    if (tileX >= 0 && tileX < int(Width / 4) && tileY >= 0 && tileY < int(Height / 4))
        packed = Input.Load((uint(tileY) * (Width / 4) + uint(tileX)) * 512 + local % 512);
    RawE4.Store(base, packed);
    uint half0 = 0, half1 = 0;
    [unroll] for (uint j = 0; j < 4; ++j) {
        uint h = FloatToHalfRN(DecodeE4M3((packed >> (j * 8)) & 255));
        if (j < 2) half0 |= h << (j * 16); else half1 |= h << ((j - 2) * 16);
    }
    RawHalf.Store(base * 2, half0); RawHalf.Store(base * 2 + 4, half1);
}
