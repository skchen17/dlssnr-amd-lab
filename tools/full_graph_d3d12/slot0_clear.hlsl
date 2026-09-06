RWByteAddressBuffer activationArena : register(u0);

// Exact captured slot-0 contract: 108 groups * 256 lanes, one u32 per lane.
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    activationArena.Store(id.x * 4, 0);
}
