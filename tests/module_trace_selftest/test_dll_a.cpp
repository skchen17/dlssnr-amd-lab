// test_dll_a — module_trace selftest fixture A.
// Loaded by selftest_host AFTER hooks are installed; its inner LoadLibraryA
// must go through module_trace's hook (proves fresh modules get IAT-patched).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstring>

// argv-style path of this DLL comes in as <dir>\test_dll_a.dll; derive
// test_dll_b.dll next to it.
extern "C" __declspec(dllexport) HMODULE WINAPI TestDllA_LoadInner(LPCSTR selfPath) {
    char buf[1024];
    strncpy(buf, selfPath, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    char* slash = strrchr(buf, '\\');
    if (!slash) slash = strrchr(buf, '/');
    if (slash) strcpy(slash + 1, "test_dll_b.dll");
    else strcpy(buf, "test_dll_b.dll");
    HMODULE b = LoadLibraryA(buf);
    if (b) {
        typedef int(WINAPI* Ping_t)();
        auto ping = (Ping_t)GetProcAddress(b, "TestDllB_Ping");
        if (ping) printf("dll_a: dll_b ping=%d\n", ping());
    }
    return b;
}

extern "C" __declspec(dllexport) int WINAPI TestDllA_Marker() {
    return 0x42;
}

// ordinal-1 export for the by-ordinal GetProcAddress check
#pragma comment(linker, "/EXPORT:TestDllA_OrdOnly,@1")
extern "C" int WINAPI TestDllA_OrdOnly() { return 1; }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
