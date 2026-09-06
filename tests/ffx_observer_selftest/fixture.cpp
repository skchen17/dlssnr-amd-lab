#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include <cstdint>

static uint64_t counts[5]{};
using Callback = void (*)(const ffxDispatchDescUpscale*);
static Callback callback = nullptr;
extern "C" __declspec(dllexport) void Fixture_SetCallback(Callback value) { callback=value; }
extern "C" __declspec(dllexport) uint64_t Fixture_Count(unsigned i) { return i<5 ? counts[i] : 0; }
extern "C" ffxReturnCode_t ffxCreateContext(ffxContext* c, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*) {
    ++counts[0]; *c=reinterpret_cast<void*>(0x12345678); return 0;
}
extern "C" ffxReturnCode_t ffxDestroyContext(ffxContext* c, const ffxAllocationCallbacks*) {
    ++counts[1]; *c=nullptr; return 29;
}
extern "C" ffxReturnCode_t ffxConfigure(ffxContext*, const ffxConfigureDescHeader*) { ++counts[2]; return GetLastError()==0xCAFE ? 23 : 99; }
extern "C" ffxReturnCode_t ffxQuery(ffxContext*, ffxQueryDescHeader* d) {
    ++counts[3]; if (d) d->type=0x11223344; return 31;
}
extern "C" ffxReturnCode_t ffxDispatch(ffxContext*, const ffxDispatchDescHeader* d) {
    ++counts[4];
    ffxApiHeader h{}; SIZE_T n=0;
    if (!d || !ReadProcessMemory(GetCurrentProcess(),d,&h,sizeof(h),&n) || n!=sizeof(h)) return 47;
    if (h.type!=FFX_API_DISPATCH_DESC_TYPE_UPSCALE) return 41;
    ffxDispatchDescUpscale body{};
    if (!ReadProcessMemory(GetCurrentProcess(),d,&body,sizeof(body),&n) || n!=sizeof(body)) return 43;
    if (callback && body.commandList) callback(reinterpret_cast<const ffxDispatchDescUpscale*>(d));
    SetLastError(0xFACE);
    return 37;
}
