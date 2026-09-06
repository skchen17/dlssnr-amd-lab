#pragma once
#include <windows.h>
#include <cstdint>
struct FfxSessionConfigV1 {
    uint32_t size,version,sample_limit,capture_enabled;
    uint64_t expected_dispatch_module,expected_context_module;
    wchar_t log_path[1024];
};
struct FfxSessionTestConfigV1 {
    FfxSessionConfigV1 common;
    uint64_t original_create,original_destroy,original_dispatch;
};
// Config version 2 starts idle and requires capture_enabled=0. Version 1 is legacy.
struct FfxSessionCommandV1 {uint32_t size,command;}; // 1/2 capture, 3/4 observe, 5/6 output, 7/8 roundtrip, 9/10 patch, 11/12 fixed filter, 13/14 weighted network
// 15 static network preview, 16 preview stop, 17 status, 18 static input preview.
