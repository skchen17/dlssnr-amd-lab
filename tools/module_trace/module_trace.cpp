// module_trace — Phase 4: module-load recorder.
//
// Interposes LoadLibraryExW/A + LoadLibraryW/A + FreeLibrary via IAT patching of
// every loaded module (no third-party hooking library needed), and logs every
// module load/unload as JSONL. Useful for observing which DLLs the DLSSNR host
// pulls in (nvngx_*, nvcuda, nvapi, dxcore, dxgi, d3d12, d3d12core, etc.).
//
// Log: %MODULE_TRACE_LOG% (default: .\module_trace.log).
// Inject by loading this DLL into the target process early (launcher/injector);
// see docs/CALLGRAPH.md runbook.
//
// Note: IAT patching catches standard LoadLibrary* calls. Direct ntdll
// LdrLoadDll usage (rare in game code paths) is NOT intercepted and is recorded
// here as a known limitation rather than silently ignored.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <share.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#pragma comment(lib, "psapi.lib")

static CRITICAL_SECTION g_cs;
static FILE* g_log = nullptr;
static LARGE_INTEGER g_freq;
static bool g_ready = false;

typedef HMODULE(WINAPI* LoadLibraryExW_t)(LPCWSTR, HANDLE, DWORD);
typedef HMODULE(WINAPI* LoadLibraryExA_t)(LPCSTR, HANDLE, DWORD);
typedef HMODULE(WINAPI* LoadLibraryW_t)(LPCWSTR);
typedef HMODULE(WINAPI* LoadLibraryA_t)(LPCSTR);
typedef BOOL(WINAPI* FreeLibrary_t)(HMODULE);

static LoadLibraryExW_t g_realLoadLibraryExW = LoadLibraryExW;
static LoadLibraryExA_t g_realLoadLibraryExA = LoadLibraryExA;
static LoadLibraryW_t g_realLoadLibraryW = LoadLibraryW;
static LoadLibraryA_t g_realLoadLibraryA = LoadLibraryA;
static FreeLibrary_t g_realFreeLibrary = FreeLibrary;

static double NowMs() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)g_freq.QuadPart;
}

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

static void LogWide(const char* ev, const wchar_t* wpath, HMODULE res) {
    char path[1024] = {0};
    if (wpath) WideCharToMultiByte(CP_UTF8, 0, wpath, -1, path, sizeof(path) - 1, nullptr, nullptr);
    Log("{\"ev\":\"%s\",\"ts\":%.3f,\"tid\":%lu,\"path\":\"%s\",\"result\":\"0x%llx\"}",
        ev, NowMs(), GetCurrentThreadId(), path, (unsigned long long)(uintptr_t)res);
}
static void LogNarrow(const char* ev, const char* path, HMODULE res) {
    Log("{\"ev\":\"%s\",\"ts\":%.3f,\"tid\":%lu,\"path\":\"%s\",\"result\":\"0x%llx\"}",
        ev, NowMs(), GetCurrentThreadId(), path ? path : "", (unsigned long long)(uintptr_t)res);
}

static HMODULE WINAPI HookLoadLibraryExW(LPCWSTR p, HANDLE f, DWORD flags) {
    HMODULE r = g_realLoadLibraryExW(p, f, flags);
    LogWide("load", p, r);
    return r;
}
static HMODULE WINAPI HookLoadLibraryExA(LPCSTR p, HANDLE f, DWORD flags) {
    HMODULE r = g_realLoadLibraryExA(p, f, flags);
    LogNarrow("load", p, r);
    return r;
}
static HMODULE WINAPI HookLoadLibraryW(LPCWSTR p) {
    HMODULE r = g_realLoadLibraryW(p);
    LogWide("load", p, r);
    return r;
}
static HMODULE WINAPI HookLoadLibraryA(LPCSTR p) {
    HMODULE r = g_realLoadLibraryA(p);
    LogNarrow("load", p, r);
    return r;
}
static BOOL WINAPI HookFreeLibrary(HMODULE m) {
    char name[512] = "?";
    GetModuleFileNameA(m, name, sizeof(name));
    BOOL r = g_realFreeLibrary(m);
    Log("{\"ev\":\"free\",\"ts\":%.3f,\"tid\":%lu,\"path\":\"%s\",\"ok\":%s}",
        NowMs(), GetCurrentThreadId(), name, r ? "true" : "false");
    return r;
}

// Patch one module's IAT entries that point at the real loader APIs.
static void PatchModuleIAT(HMODULE mod, HMODULE self) {
    if (mod == self) return;
    // guard against non-image memory (SEH is unreliable in this toolchain)
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery((LPCVOID)mod, &mbi, sizeof(mbi)) < sizeof(mbi) ||
        mbi.State != MEM_COMMIT) return;
    auto dos = (IMAGE_DOS_HEADER*)mod;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto nt = (IMAGE_NT_HEADERS*)((BYTE*)mod + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    auto dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir->VirtualAddress) return;
    auto imp = (IMAGE_IMPORT_DESCRIPTOR*)((BYTE*)mod + dir->VirtualAddress);
    for (; imp->Name; ++imp) {
        const char* dllName = (const char*)((BYTE*)mod + imp->Name);
        bool isKernel = _stricmp(dllName, "kernel32.dll") == 0 ||
                        _stricmp(dllName, "KERNELBASE.dll") == 0;
        if (!isKernel) continue;
        auto thunk = (IMAGE_THUNK_DATA*)((BYTE*)mod +
                       (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        auto iat = (IMAGE_THUNK_DATA*)((BYTE*)mod + imp->FirstThunk);
        for (; thunk->u1.AddressOfData; ++thunk, ++iat) {
            if (thunk->u1.Ordinal & IMAGE_ORDINAL_FLAG) continue;
            auto byName = (IMAGE_IMPORT_BY_NAME*)((BYTE*)mod + thunk->u1.AddressOfData);
            void* target = nullptr;
            if (!strcmp(byName->Name, "LoadLibraryExW")) target = (void*)HookLoadLibraryExW;
            else if (!strcmp(byName->Name, "LoadLibraryExA")) target = (void*)HookLoadLibraryExA;
            else if (!strcmp(byName->Name, "LoadLibraryW")) target = (void*)HookLoadLibraryW;
            else if (!strcmp(byName->Name, "LoadLibraryA")) target = (void*)HookLoadLibraryA;
            else if (!strcmp(byName->Name, "FreeLibrary")) target = (void*)HookFreeLibrary;
            if (!target) continue;
            DWORD old = 0;
            if (VirtualProtect(&iat->u1.Function, sizeof(void*), PAGE_READWRITE, &old)) {
                iat->u1.Function = (ULONGLONG)(uintptr_t)target;
                VirtualProtect(&iat->u1.Function, sizeof(void*), old, &old);
            }
        }
    }
}

static void PatchAllModules(HMODULE self) {
    HMODULE mods[1024];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return;
    int n = needed / sizeof(HMODULE);
    for (int i = 0; i < n; ++i) PatchModuleIAT(mods[i], self);
}

static DWORD WINAPI InitThread(LPVOID param) {
    HMODULE self = (HMODULE)param;
    // capture already-loaded modules as baseline
    HMODULE mods[1024];
    DWORD needed = 0;
    if (EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed)) {
        int n = needed / sizeof(HMODULE);
        for (int i = 0; i < n; ++i) {
            char name[1024] = "?";
            GetModuleFileNameA(mods[i], name, sizeof(name));
            Log("{\"ev\":\"baseline\",\"ts\":%.3f,\"path\":\"%s\"}", NowMs(), name);
        }
    }
    PatchAllModules(self);
    Log("{\"ev\":\"hooks_installed\",\"ts\":%.3f,\"pid\":%lu}", NowMs(), GetCurrentProcessId());
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls((HMODULE)inst);
        InitializeCriticalSection(&g_cs);
        QueryPerformanceFrequency(&g_freq);
        const char* logPath = getenv("MODULE_TRACE_LOG");
        if (!logPath || !*logPath) logPath = "module_trace.log";
        g_log = _fsopen(logPath, "a", _SH_DENYNO);   // shared: readers may tail while we log
        g_ready = true;
        Log("{\"ev\":\"attach\",\"ts\":%.3f,\"pid\":%lu}", NowMs(), GetCurrentProcessId());
        // never do heavy work in DllMain: defer
        HANDLE t = CreateThread(nullptr, 0, InitThread, (LPVOID)inst, 0, nullptr);
        if (t) CloseHandle(t);
    }
    return TRUE;
}

// marker export so injectors can verify this is the right dll
extern "C" __declspec(dllexport) const char* module_trace_version() {
    return "dlssnr-amd-lab module_trace 1.0";
}
