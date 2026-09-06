#pragma once
#include <windows.h>
#include <cstdint>

// No automatic DllMain hooks or game deployment. Attach/detach are allowed only
// at a caller-established quiescent boundary. The observer is pinned on attach.
// Profile 0: generic 16-byte descriptor headers ONLY (default for real game).
// Profile 1: caller asserts the pinned SDK 1.1.3 body ABI (synthetic validation).
struct FfxObserverConfigV1 {
    uint32_t size;
    uint32_t profile;
    HMODULE importing_module;
    const wchar_t* log_path;
    uint32_t event_limit;
};
struct FfxObserverStatsV1 {
    uint32_t size;
    uint32_t patched_imports;
    uint64_t forwarded_calls;
    uint64_t emitted_events;
    uint64_t dropped_events;
    uint64_t active_calls;
};
using FfxObserverAttach_t = int (__cdecl*)(const FfxObserverConfigV1*);
using FfxObserverDetach_t = int (__cdecl*)();
using FfxObserverStats_t = int (__cdecl*)(FfxObserverStatsV1*);

// Pointer-free remote entry arguments. Launcher holds the application's primary
// thread at its restored PE entry point. Profile stays header-only for bootstrap.
struct FfxObserverBootstrapV2 {
    uint32_t size;
    uint32_t version;
    uint32_t profile;
    uint32_t event_limit;
    wchar_t log_path[1024];
};
