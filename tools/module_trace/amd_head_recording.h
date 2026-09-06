#pragma once
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <wrl/client.h>
#include "../nvapi_amd/nvapi_amd_d3d12_head.h"

// Included after the trace's resource resolvers and Log declaration. Only the
// opt-in fixed-contract path consumes this conservative legacy-barrier tracker.
namespace auto_head {
using Microsoft::WRL::ComPtr;
struct State { D3D12_RESOURCE_STATES state; bool valid; };
struct Pending {
    ComPtr<ID3D12Fence> fence;
    NvapiAmdReleaseHeadD3D12_t release = nullptr;
    NvapiAmdSealHeadD3D12_t seal = nullptr;
    bool submitted = false;
    std::vector<NvapiAmdHeadResourceV1> bindings;
};
inline std::recursive_mutex mutex;
inline std::unordered_map<ID3D12GraphicsCommandList*, std::unordered_map<ID3D12Resource*, State>> states;
inline std::unordered_map<ID3D12GraphicsCommandList*, Pending> pending;
inline std::unordered_set<ID3D12GraphicsCommandList*> closed;
inline thread_local bool internal = false;
inline uint64_t begun = 0, retired = 0, rejected = 0;
inline bool Enabled() {
    char value[8]{}; return GetEnvironmentVariableA("MODULE_TRACE_AMD_D3D12_HEAD", value, 8) && strcmp(value, "1") == 0;
}
struct InternalScope {
    bool previous = internal;
    InternalScope() { internal = true; }
    ~InternalScope() { internal = previous; }
};
inline void Collect() {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    for (auto it = pending.begin(); it != pending.end();) {
        if (it->second.submitted && it->second.fence->GetCompletedValue() == 1 && it->second.release(it->first)) {
            Log("{\"ev\":\"amd_head_retired\",\"list\":\"0x%llx\"}", (unsigned long long)(uintptr_t)it->first);
            ++retired; it = pending.erase(it);
        } else ++it;
    }
}
inline void Barriers(ID3D12GraphicsCommandList* list, UINT count, const D3D12_RESOURCE_BARRIER* barriers) {
    if (!Enabled() || internal || !barriers) return;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto& map = states[list];
    for (UINT i = 0; i < count; ++i) {
        const auto& b = barriers[i];
        if (b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
            map[b.Transition.pResource] = {b.Transition.StateAfter,
                b.Flags == D3D12_RESOURCE_BARRIER_FLAG_NONE && b.Transition.Subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES};
        } else if (b.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
            // Heap aliasing makes earlier whole-resource assumptions invalid.
            for (auto& [resource, state] : map) state.valid = false;
        }
    }
}
inline void CopyUse(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    if (!Enabled() || internal || !resource) return;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto& map = states[list]; auto found = map.find(resource);
    if (found != map.end() && found->second.state == D3D12_RESOURCE_STATE_COMMON) found->second.valid = false;
}
inline void DispatchUse(ID3D12GraphicsCommandList* list) {
    if (!Enabled() || internal) return;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    for (auto& [r, state] : states[list]) if (state.state == D3D12_RESOURCE_STATE_COMMON) state.valid = false;
}
inline bool ResetAllowed(ID3D12GraphicsCommandList* list) {
    if (!Enabled()) return true;
    Collect();
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return pending.find(list) == pending.end();
}
inline void ResetStates(ID3D12GraphicsCommandList* list) {
    std::lock_guard<std::recursive_mutex> guard(mutex); states.erase(list); closed.erase(list);
}
inline void Closed(ID3D12GraphicsCommandList* list) { std::lock_guard<std::recursive_mutex> guard(mutex); closed.insert(list); }
// Called immediately before the actual backend launch, not by a synthetic host
// registration API. Unknown resources/states fail closed without HIP fallback.
inline bool Prepare(ID3D12GraphicsCommandList* list, const void* params, uint32_t size, void* backendFunction) {
    Collect();
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (!list || !params || size != 184 || !backendFunction || closed.count(list)) return false;
    auto pointer = [&](size_t offset) { uint64_t p; memcpy(&p, (const uint8_t*)params + offset, 8); return p; };
    const auto main = ResolveBufferAddress(pointer(0)), skip = ResolveBufferAddress(pointer(8)), model = ResolveBufferAddress(pointer(24));
    ID3D12Resource* base = FindObjectResource(pointer(56));
    ID3D12Resource* output = FindObjectResource(pointer(16));
    if (!main.resource || skip.resource != main.resource || !model.resource || !base || !output) { ++rejected; return false; }
    auto foundStates = states.find(list);
    if (foundStates == states.end()) { ++rejected; return false; }
    std::vector<NvapiAmdHeadResourceV1> resources;
    for (auto* resource : {main.resource, model.resource, base, output}) {
        const auto found = foundStates->second.find(resource);
        if (found == foundStates->second.end() || !found->second.valid) { ++rejected; return false; }
        resources.push_back({resource, uint32_t(found->second.state)});
    }
    auto existing = pending.find(list);
    if (existing != pending.end()) {
        if (existing->second.submitted) return false;
        for (size_t i = 0; i < resources.size(); ++i)
            if (resources[i].resource != existing->second.bindings[i].resource || resources[i].state != existing->second.bindings[i].state) { ++rejected; return false; }
        return true;
    }
    HMODULE backend = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(backendFunction), &backend)) return false;
    auto begin = reinterpret_cast<NvapiAmdBeginHeadD3D12_t>(g_realGetProcAddress(backend, "NvapiAmd_BeginHeadD3D12"));
    auto release = reinterpret_cast<NvapiAmdReleaseHeadD3D12_t>(g_realGetProcAddress(backend, "NvapiAmd_ReleaseHeadD3D12"));
    auto seal = reinterpret_cast<NvapiAmdSealHeadD3D12_t>(g_realGetProcAddress(backend, "NvapiAmd_SealHeadD3D12"));
    if (!begin || !release || !seal) return false;
    ComPtr<ID3D12Device> device; ComPtr<ID3D12Fence> fence;
    if (FAILED(list->GetDevice(IID_PPV_ARGS(&device))) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    NvapiAmdHeadSessionV1 config{sizeof(config), 1, list, nullptr, nullptr, 0, resources.data(), uint32_t(resources.size())};
    InternalScope suppress;
    if (!begin(&config)) { ++rejected; return false; }
    pending.emplace(list, Pending{fence, release, seal, false, resources});
    ++begun;
    Log("{\"ev\":\"amd_head_auto_begin\",\"list\":\"0x%llx\",\"resources\":4}", (unsigned long long)(uintptr_t)list);
    return true;
}
inline bool BeforeSubmit(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists) {
    if (!Enabled()) return true;
    Collect();
    std::lock_guard<std::recursive_mutex> guard(mutex);
    // Seal is irreversible. Preflight the complete batch before attaching any
    // fence; this fixed-contract bridge permits only one head list per submit.
    // In particular, a duplicated list must never be sealed and then rejected.
    unsigned headLists = 0;
    for (UINT i = 0; i < count; ++i) {
        const auto found = pending.find(static_cast<ID3D12GraphicsCommandList*>(lists[i]));
        if (found == pending.end()) continue;
        if (++headLists > 1 || found->second.submitted || !closed.count(found->first)) {
            ++rejected; Log("{\"ev\":\"amd_head_submit_rejected\",\"reason\":\"batch_preflight\"}"); return false;
        }
    }
    for (UINT i = 0; i < count; ++i) {
        const auto found = pending.find(static_cast<ID3D12GraphicsCommandList*>(lists[i]));
        if (found == pending.end()) continue;
        if (found->second.submitted || !found->second.seal(found->first, queue, found->second.fence.Get(), 1)) {
            ++rejected; Log("{\"ev\":\"amd_head_submit_rejected\"}"); return false;
        }
        found->second.submitted = true;
    }
    return true;
}
inline void AfterSubmit(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists) {
    if (!Enabled()) return;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    for (UINT i = 0; i < count; ++i) {
        const auto found = pending.find(static_cast<ID3D12GraphicsCommandList*>(lists[i]));
        if (found == pending.end()) continue;
        HRESULT status = queue->Signal(found->second.fence.Get(), 1);
        Log("{\"ev\":\"amd_head_queue_signal\",\"list\":\"0x%llx\",\"status\":\"0x%08X\"}",
            (unsigned long long)(uintptr_t)found->first, unsigned(status));
    }
}
} // namespace auto_head
