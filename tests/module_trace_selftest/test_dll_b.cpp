// test_dll_b — module_trace selftest fixture B.
// Loaded from INSIDE test_dll_a to prove that a module loaded after the
// hooks were installed also gets its IAT patched / its load logged.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern "C" __declspec(dllexport) int WINAPI TestDllB_Ping() {
    return 7;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
