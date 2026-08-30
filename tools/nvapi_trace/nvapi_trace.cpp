// nvapi_trace — Phase 4: nvapi64.dll replacement shim.
//
// Interposes nvapi_QueryInterface, resolves every requested function pointer from
// the real nvapi64.dll, wraps each in a forwarding trampoline, and records every
// call (timestamp / thread / id / name / return value / arg1 magic+readability).
// Binary payload pointers are never copied; only size-independent facts (magic
// bytes behind a SEH-protected read) are recorded.
//
// Log: JSONL at %NVAPI_TRACE_LOG% (default: .\nvapi_trace.jsonl).
// Real dll search order: %NVAPI_REAL_DLL% -> <this dir>\nvapi64_real.dll
//   -> <system32>\nvapi64.dll. If none exists the shim still records resolutions
//   (returns NULL) so callers' intent is captured.
//
// Usage: place build\nvapi64.dll next to the target executable; keep the real one
// as nvapi64_real.dll (or set NVAPI_REAL_DLL). See docs/RESEARCH.md for legality.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>

#include "nvapi_ids.h"

typedef void* (*RealQueryInterface)(uint32_t);
typedef uint64_t (*NvFn4)(uint64_t, uint64_t, uint64_t, uint64_t);

struct StubCtx {
    uint32_t id;
    void* realFn;
    void* stub;   // forwarding trampoline returned to callers
};

static CRITICAL_SECTION g_cs;
static FILE* g_log = nullptr;
static RealQueryInterface g_realQuery = nullptr;
static HMODULE g_realDll = nullptr;
static bool g_logInitDone = false;
static bool g_backendInitDone = false;
static std::map<uint32_t, StubCtx*> g_stubs;   // id -> ctx (stub code lives next to it)
static LARGE_INTEGER g_qpcFreq;

static void Log(const char* fmt, ...) {
    if (!g_log) return;
    EnterCriticalSection(&g_cs);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
    LeaveCriticalSection(&g_cs);
}

static double NowMs() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)g_qpcFreq.QuadPart;
}

static const char* NameFor(uint32_t id, const char** src) {
    for (auto& e : kNvApiKnownIds)
        if (e.id == id) { if (src) *src = e.source; return e.name; }
    for (auto& e : kNvApiCommunityIds)
        if (e.id == id) { if (src) *src = e.source; return e.name; }
    if (src) *src = "unknown";
    return nullptr;
}

// DllMain-safe: log file + timer only. The real nvapi64.dll is loaded lazily on
// first query (loading third-party DLLs under the loader lock is a deadlock risk).
static void LogInit() {
    if (g_logInitDone) return;
    g_logInitDone = true;
    InitializeCriticalSection(&g_cs);
    QueryPerformanceFrequency(&g_qpcFreq);
    const char* logPath = getenv("NVAPI_TRACE_LOG");
    if (!logPath || !*logPath) logPath = "nvapi_trace.jsonl";
    g_log = _fsopen(logPath, "a", _SH_DENYNO);   // shared: readers may tail while we log
}

static void BackendInit() {
    if (g_backendInitDone) return;
    g_backendInitDone = true;

    char selfPath[1024] = {0};
    HMODULE self = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&BackendInit, &self);
    GetModuleFileNameA(self, selfPath, _countof(selfPath));

    // locate the real nvapi64.dll
    char realPath[1200] = {0};
    const char* env = getenv("NVAPI_REAL_DLL");
    if (env && *env && GetFileAttributesA(env) != INVALID_FILE_ATTRIBUTES) {
        strcpy_s(realPath, env);
    } else {
        // nvapi64_real.dll next to this shim
        char dir[1024];
        strcpy_s(dir, selfPath);
        char* slash = strrchr(dir, '\\');
        if (slash) *slash = 0;
        snprintf(realPath, sizeof(realPath), "%s\\nvapi64_real.dll", dir);
        if (GetFileAttributesA(realPath) == INVALID_FILE_ATTRIBUTES) {
            char sys[MAX_PATH];
            GetSystemDirectoryA(sys, MAX_PATH);
            snprintf(realPath, sizeof(realPath), "%s\\nvapi64.dll", sys);
        }
    }
    g_realDll = LoadLibraryA(realPath);
    if (g_realDll) g_realQuery = (RealQueryInterface)GetProcAddress(g_realDll, "nvapi_QueryInterface");

    Log("{\"ev\":\"backend\",\"ts\":%.3f,\"pid\":%lu,\"shim\":\"%s\",\"real\":\"%s\",\"real_loaded\":%s,\"real_query\":%s}",
        NowMs(), GetCurrentProcessId(), selfPath, realPath,
        g_realDll ? "true" : "false", g_realQuery ? "true" : "false");
}

// Readability-checked magic peek: records first 4 bytes of arg1 if committed and
// readable (never copies more). Uses VirtualQuery instead of SEH for portability.
static void PeekArg1(uint64_t a1, char* out, size_t n) {
    out[0] = 0;
    if (!a1) { snprintf(out, n, "null"); return; }
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery((LPCVOID)a1, &mbi, sizeof(mbi)) < sizeof(mbi) ||
        mbi.State != MEM_COMMIT ||
        (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
        !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                         PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                         PAGE_EXECUTE_WRITECOPY))) {
        snprintf(out, n, "unreadable");
        return;
    }
    if ((uintptr_t)mbi.BaseAddress + mbi.RegionSize < (uintptr_t)a1 + 4) {
        snprintf(out, n, "unreadable");
        return;
    }
    uint32_t m = *(volatile uint32_t*)(uintptr_t)a1;
    snprintf(out, n, "0x%08x", m);
}

// Central dispatcher: called by every trampoline. Forwards up to 4 register args;
// additional stack args ride along untouched (pure jmp chain, no stack delta).
extern "C" uint64_t NvapiTrampolineDispatch(StubCtx* ctx, uint64_t a1, uint64_t a2,
                                            uint64_t a3, uint64_t a4) {
    double t0 = NowMs();
    DWORD tid = GetCurrentThreadId();
    const char* src = nullptr;
    const char* name = NameFor(ctx->id, &src);
    char magic[32];
    PeekArg1(a1, magic, sizeof(magic));
    Log("{\"ev\":\"call\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08x\",\"name\":\"%s\",\"src\":\"%s\",\"arg1\":%llu,\"arg1_magic\":\"%s\"}",
        t0, tid, ctx->id, name ? name : "", src, (unsigned long long)a1, magic);

    uint64_t ret = 0;
    if (ctx->realFn) {
        ret = ((NvFn4)ctx->realFn)(a1, a2, a3, a4);
    } else {
        ret = 0xFFFFFFFF;   // NVAPI_GENERIC_ERROR-ish sentinel when no real dll
    }
    Log("{\"ev\":\"ret\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08x\",\"ret\":%lld,\"ret_hex\":\"0x%llx\",\"dt_ms\":%.3f}",
        NowMs(), tid, ctx->id, (long long)(int32_t)ret, (unsigned long long)ret, NowMs() - t0);
    return ret;
}

// Thunk: stubs land here with rax=&StubCtx; preserves rcx..r9 and re-slots args.
// All jumps use absolute indirection (rel32 cannot span VirtualAlloc<->image gap).
//   mov r10, rcx         ; original arg1
//   mov rcx, rax         ; ctx
//   mov rdx, r10         ; arg1 -> param2
//   mov r11, imm64       ; &NvapiTrampolineDispatch
//   jmp r11
static unsigned char* g_thunk = nullptr;   // RWX copy built at first stub creation

static void* MakeStub(StubCtx* ctx) {
    // stub: mov rax, imm64(ctx) ; mov r11, imm64(thunk) ; jmp r11
    static unsigned char* pool = nullptr;
    static size_t poolUsed = 0, poolCap = 0;
    if (!g_thunk) {
        g_thunk = (unsigned char*)VirtualAlloc(nullptr, 4096, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
        unsigned char* t = g_thunk;
        size_t n = 0;
        t[n++] = 0x4C; t[n++] = 0x8B; t[n++] = 0xD1;   // mov r10, rcx
        t[n++] = 0x48; t[n++] = 0x8B; t[n++] = 0xC8;   // mov rcx, rax
        t[n++] = 0x49; t[n++] = 0x8B; t[n++] = 0xD2;   // mov rdx, r10
        t[n++] = 0x49; t[n++] = 0xBB;                   // mov r11, imm64
        void* disp = (void*)&NvapiTrampolineDispatch;
        memcpy(t + n, &disp, 8); n += 8;
        t[n++] = 0x41; t[n++] = 0xFF; t[n++] = 0xE3;   // jmp r11
    }
    if (!pool || poolUsed + 48 > poolCap) {
        poolCap = 4096;
        pool = (unsigned char*)VirtualAlloc(nullptr, poolCap, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
        poolUsed = 0;
    }
    unsigned char* s = pool + poolUsed;
    poolUsed += 48;
    size_t n = 0;
    s[n++] = 0x48; s[n++] = 0xB8;                       // mov rax, imm64
    memcpy(s + n, &ctx, 8); n += 8;
    s[n++] = 0x49; s[n++] = 0xBB;                       // mov r11, imm64
    void* thunk = g_thunk;
    memcpy(s + n, &thunk, 8); n += 8;
    s[n++] = 0x41; s[n++] = 0xFF; s[n++] = 0xE3;        // jmp r11
    return s;
}

extern "C" __declspec(dllexport) void* __cdecl nvapi_QueryInterface(uint32_t id) {
    LogInit();
    BackendInit();
    void* real = g_realQuery ? g_realQuery(id) : nullptr;
    const char* src = nullptr;
    const char* name = NameFor(id, &src);
    bool cuFamily = (src && strcmp(src, "community") == 0);

    EnterCriticalSection(&g_cs);
    auto it = g_stubs.find(id);
    StubCtx* ctx = (it != g_stubs.end()) ? it->second : nullptr;
    if (ctx) ctx->realFn = real;   // refresh (real ptr may change across init)
    LeaveCriticalSection(&g_cs);

    if (ctx) {
        Log("{\"ev\":\"resolve_again\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08x\",\"real_ptr\":%llu}",
            NowMs(), GetCurrentThreadId(), id, (unsigned long long)(uintptr_t)real);
        return ctx->stub;
    }

    ctx = new StubCtx{ id, real, nullptr };
    ctx->stub = MakeStub(ctx);
    EnterCriticalSection(&g_cs);
    g_stubs[id] = ctx;
    LeaveCriticalSection(&g_cs);
    Log("{\"ev\":\"resolve\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08x\",\"name\":\"%s\",\"src\":\"%s\",\"real_ptr\":%llu,\"stub\":%llu,\"cu_family\":%s}",
        NowMs(), GetCurrentThreadId(), id, name ? name : "", src,
        (unsigned long long)(uintptr_t)real, (unsigned long long)(uintptr_t)ctx->stub,
        cuFamily ? "true" : "false");
    return ctx->stub;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(GetModuleHandleA(nullptr));
        LogInit();   // only log/timer setup here; backend init is lazy
    }
    return TRUE;
}
