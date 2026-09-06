// nvapi_trace — Phase 4: nvapi64.dll replacement shim.
//
// Interposes nvapi_QueryInterface, resolves every requested function pointer from
// the real nvapi64.dll, wraps each in a forwarding trampoline, and records every
// call (timestamp / thread / id / name / return value / arg1 magic+readability).
// Binary payload pointers are never copied; only size-independent facts (magic
// bytes behind a SEH-protected read) are recorded.
//
// Round 2 Phase F: the "stack args ride along" assumption is now validated by
// tests/nvapi_trampoline_test (1..10 args, byte-exact). The trampoline builder
// is exported as NvapiTrace_MakeTrampoline for that self-test.
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
#include <initializer_list>
#include <map>

#include "nvapi_ids.h"

typedef void* (*RealQueryInterface)(uint32_t);

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
    char logPath[1024] = "nvapi_trace.jsonl";
    DWORD n = GetEnvironmentVariableA("NVAPI_TRACE_LOG", logPath, sizeof(logPath));
    if (n == 0 || n >= sizeof(logPath)) strcpy(logPath, "nvapi_trace.jsonl");
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

// Central log point: called by the trampoline BEFORE it jumps to the real
// function. Round 1 wrapped the call in a C dispatcher (NvFn4 cast, "stack
// args ride along"); tests/nvapi_trampoline_test DISPROVED that for args 5+
// (the dispatcher's frame sits between caller and callee, so stack args were
// read from the wrong place). Round 2 fix: the trampoline only logs here,
// then restores rcx/rdx/r8/r9 and does a pure `jmp` to the real function,
// leaving the caller's stack arguments untouched. Consequence: return values
// are no longer logged (byte-exact passthrough beats observability).
// xmm/vector args are not preserved (nvapi surface is int/ptr; documented).
extern "C" void NvapiTrace_LogEntry(StubCtx* ctx, uint64_t a1) {
    double t0 = NowMs();
    DWORD tid = GetCurrentThreadId();
    const char* src = nullptr;
    const char* name = NameFor(ctx->id, &src);
    char magic[32];
    PeekArg1(a1, magic, sizeof(magic));
    Log("{\"ev\":\"call\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08x\",\"name\":\"%s\",\"src\":\"%s\",\"arg1\":%llu,\"arg1_magic\":\"%s\",\"real\":%s}",
        t0, tid, ctx->id, name ? name : "", src, (unsigned long long)a1, magic,
        ctx->realFn ? "true" : "false");
}

// Thunk: stubs land here with rax=&StubCtx. Byte-exact passthrough for any
// argument count (validated by tests/nvapi_trampoline_test, 1..10 args):
//   sub rsp, 0x58               ; 0x28 home for the log call + 5 save slots
//                               ; + 8 alignment (entry rsp == 8 mod 16)
//   mov [rsp+0x28], rcx         ; stash arg1..arg4 + ctx below the stack args
//   mov [rsp+0x30], rdx
//   mov [rsp+0x38], r8
//   mov [rsp+0x40], r9
//   mov [rsp+0x48], rax
//   mov rdx, rcx                ; LogEntry(ctx, a1)
//   mov rcx, rax
//   mov r11, imm64              ; &NvapiTrace_LogEntry (abs: alloc<->image gap)
//   call r11
//   mov rcx/rdx/r8/r9/rax back from the save slots
//   mov r11, [rax+8]            ; StubCtx.realFn
//   test r11,r11 / jnz
//   mov rax, 0xFFFFFFFF         ; sentinel when no real dll, then ret
//   add rsp, 0x58               ; rsp == entry rsp -> stack args untouched
//   jmp r11                     ; real fn returns straight to the caller
static unsigned char* g_thunk = nullptr;   // RWX copy built at first stub creation

static void EmitBytes(unsigned char* t, size_t& n, std::initializer_list<unsigned char> b) {
    for (auto x : b) t[n++] = x;
}

static void* MakeStub(StubCtx* ctx) {
    // stub: mov rax, imm64(ctx) ; mov r11, imm64(thunk) ; jmp r11
    static unsigned char* pool = nullptr;
    static size_t poolUsed = 0, poolCap = 0;
    if (!g_thunk) {
        g_thunk = (unsigned char*)VirtualAlloc(nullptr, 4096, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
        unsigned char* t = g_thunk;
        size_t n = 0;
        EmitBytes(t, n, {0x48,0x81,0xEC,0x58,0x00,0x00,0x00});   // sub rsp, 0x58
        EmitBytes(t, n, {0x48,0x89,0x4C,0x24,0x28});             // mov [rsp+0x28], rcx
        EmitBytes(t, n, {0x48,0x89,0x54,0x24,0x30});             // mov [rsp+0x30], rdx
        EmitBytes(t, n, {0x4C,0x89,0x44,0x24,0x38});             // mov [rsp+0x38], r8
        EmitBytes(t, n, {0x4C,0x89,0x4C,0x24,0x40});             // mov [rsp+0x40], r9
        EmitBytes(t, n, {0x48,0x89,0x44,0x24,0x48});             // mov [rsp+0x48], rax
        EmitBytes(t, n, {0x48,0x8B,0xD1});                       // mov rdx, rcx
        EmitBytes(t, n, {0x48,0x8B,0xC8});                       // mov rcx, rax
        EmitBytes(t, n, {0x49,0xBB});                            // mov r11, imm64
        void* logFn = (void*)&NvapiTrace_LogEntry;
        memcpy(t + n, &logFn, 8); n += 8;
        EmitBytes(t, n, {0x41,0xFF,0xD3});                       // call r11
        EmitBytes(t, n, {0x48,0x8B,0x4C,0x24,0x28});             // mov rcx, [rsp+0x28]
        EmitBytes(t, n, {0x48,0x8B,0x54,0x24,0x30});             // mov rdx, [rsp+0x30]
        EmitBytes(t, n, {0x4C,0x8B,0x44,0x24,0x38});             // mov r8,  [rsp+0x38]
        EmitBytes(t, n, {0x4C,0x8B,0x4C,0x24,0x40});             // mov r9,  [rsp+0x40]
        EmitBytes(t, n, {0x48,0x8B,0x44,0x24,0x48});             // mov rax, [rsp+0x48]
        EmitBytes(t, n, {0x4C,0x8B,0x58,0x08});                  // mov r11, [rax+8] (realFn)
        EmitBytes(t, n, {0x4D,0x85,0xDB});                       // test r11, r11
        EmitBytes(t, n, {0x75,0x0F});                            // jnz +15 (skip sentinel)
        EmitBytes(t, n, {0x48,0xC7,0xC0,0xFF,0xFF,0xFF,0xFF});   // mov rax, 0xFFFFFFFF
        EmitBytes(t, n, {0x48,0x81,0xC4,0x58,0x00,0x00,0x00});   // add rsp, 0x58
        EmitBytes(t, n, {0xC3});                                 // ret
        EmitBytes(t, n, {0x48,0x81,0xC4,0x58,0x00,0x00,0x00});   // add rsp, 0x58
        EmitBytes(t, n, {0x41,0xFF,0xE3});                       // jmp r11
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

// ---------------------------------------------------------------------------
// Round 2 Phase F — ABI self-test hook.
// Lets tests/nvapi_trampoline_test wrap an arbitrary function in the exact
// same stub/thunk chain that nvapi_QueryInterface hands out, then verify all
// arguments (1..10) + the return value arrive byte-exact.
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport) void* __cdecl NvapiTrace_MakeTrampoline(
        uint32_t id, void* realFn) {
    LogInit();
    StubCtx* ctx = new StubCtx{ id, realFn, nullptr };
    ctx->stub = MakeStub(ctx);
    EnterCriticalSection(&g_cs);
    g_stubs[id] = ctx;
    LeaveCriticalSection(&g_cs);
    return ctx->stub;
}
