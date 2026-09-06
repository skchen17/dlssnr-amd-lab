#pragma once
#include <cstdint>

// Private compatibility-host API, not NVIDIA NVAPI. The host opts a list into
// the captured 640x360/frame1 head contract. State values are D3D12_RESOURCE_STATES.
// No state inference: callers supply states at each intercepted head invocation.
struct NvapiAmdHeadResourceV1 {
    void* resource;
    uint32_t state;
};
struct NvapiAmdHeadSessionV1 {
    uint32_t struct_size;
    uint32_t contract; // 1: captured model310.8, 640x360, no history, fixed composition
    void* command_list;
    void* queue;
    void* completion_fence;
    uint64_t completion_value;
    const NvapiAmdHeadResourceV1* resources;
    uint32_t resource_count;
};
struct NvapiAmdHeadDiagnosticsV1 {
    uint32_t struct_size;
    uint32_t active_sessions;
    uint64_t sessions_created;
    uint64_t sessions_released;
    uint64_t head_records;
    uint64_t rejected_records;
};
// Begin holds COM references, owns GPU scratch and preloads shader PSOs. Submit
// this list on the supplied queue, then queue->Signal(fence, completion_value).
// Release rejects an incomplete fence. Never CPU-signal that fence to release
// pending work. One registered list has one scratch set and is recorded serially.
// Resources must retain the declared states at every invocation; Record restores
// those states. Host must rebind subsequent compute root/PSO state.
using NvapiAmdBeginHeadD3D12_t = int(__cdecl*)(const NvapiAmdHeadSessionV1*);
using NvapiAmdReleaseHeadD3D12_t = int(__cdecl*)(void* command_list);
// Deferred sessions pass null queue/fence and value 0 to Begin. The interceptor
// attaches the actual queue and an unsignaled fence at ExecuteCommandLists.
using NvapiAmdSealHeadD3D12_t = int(__cdecl*)(void* command_list, void* queue,
                                          void* fence, uint64_t completion_value);
using NvapiAmdGetHeadD3D12Diagnostics_t = int(__cdecl*)(NvapiAmdHeadDiagnosticsV1*);
