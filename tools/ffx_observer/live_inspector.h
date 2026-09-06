#pragma once
#include <windows.h>
#include <cstdint>
struct FfxLiveConfigV1 {
    uint32_t size, version, sample_limit, reserved;
    uint64_t expected_observer_base;
    wchar_t log_path[1024];
};
struct FfxLiveTestConfigV1 { FfxLiveConfigV1 common; uint64_t original_dispatch; };
