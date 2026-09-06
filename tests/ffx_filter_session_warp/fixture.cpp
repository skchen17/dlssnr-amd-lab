#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"

extern "C" __declspec(dllexport) ffxReturnCode_t ffxCreateContext(
    ffxContext* context, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*) {
    *context = reinterpret_cast<void*>(0x46584f4b);
    return 0;
}
extern "C" __declspec(dllexport) ffxReturnCode_t ffxDestroyContext(
    ffxContext* context, const ffxAllocationCallbacks*) {
    *context = nullptr;
    return 0;
}
extern "C" __declspec(dllexport) ffxReturnCode_t ffxDispatch(
    ffxContext*, const ffxDispatchDescHeader* desc) {
    return desc && desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE ? 0 : 1;
}
