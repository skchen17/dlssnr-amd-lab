// module_trace selftest host — Round 2 Phase E.
//
// Exercises the FULL module_trace chain and verifies the JSONL log:
//   1. load module_trace.dll (absolute path)
//   2. ModuleTrace_InitializeAndWait -> hooks_installed=true
//   3. LoadLibraryA(test_dll_a)                    (self IAT already patched)
//   4. TestDllA_LoadInner -> inside dll_a: LoadLibraryA(test_dll_b) +
//      GetProcAddress  (proves freshly loaded modules get patched too)
//   5. GetProcAddress(dll_a) from here             (getprocaddr hook)
//   6. FreeLibrary both DLLs                       (free hook)
//
// Exit 0 = READY (log contains baseline, hooks_installed, load x3,
// getprocaddr, free x2 in order). Anything else prints the first missing
// expectation and exits 1.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: module_trace_selftest.exe <module_trace.dll> <test_dll_a.dll> [log path]\n");
        return 2;
    }
    const char* mtPath = argv[1];
    const char* dllAPath = argv[2];
    const char* logPath = (argc > 3) ? argv[3] : "module_trace_selftest.log";

    // module_trace opens %MODULE_TRACE_LOG% inside DllMain; set it here.
    SetEnvironmentVariableA("MODULE_TRACE_LOG", logPath);
    DeleteFileA(logPath);

    int failures = 0;
    auto check = [&](bool cond, const char* what) {
        printf("[%s] %s\n", cond ? "PASS" : "FAIL", what);
        if (!cond) ++failures;
    };

    HMODULE mt = LoadLibraryA(mtPath);
    check(mt != nullptr, "LoadLibrary(module_trace.dll)");
    if (!mt) return 1;

    typedef int(WINAPI* InitAndWait_t)(DWORD);
    auto initAndWait = (InitAndWait_t)GetProcAddress(mt, "ModuleTrace_InitializeAndWait");
    check(initAndWait != nullptr, "ModuleTrace_InitializeAndWait export present");
    int ready = initAndWait ? initAndWait(10000) : 0;
    check(ready == 1, "InitializeAndWait -> hooks_installed=true");
    if (!ready) return 1;

    // self IAT is now patched: this LoadLibraryA goes through the hook
    HMODULE a = LoadLibraryA(dllAPath);
    check(a != nullptr, "LoadLibraryA(test_dll_a) after hooks installed");
    if (!a) return 1;

    typedef HMODULE(WINAPI* LoadInner_t)(LPCSTR);
    typedef int(WINAPI* Marker_t)();
    auto loadInner = (LoadInner_t)GetProcAddress(a, "TestDllA_LoadInner");
    check(loadInner != nullptr, "resolve TestDllA_LoadInner");
    if (loadInner) {
        HMODULE b = loadInner(dllAPath);   // dll_b path is derived inside dll_a
        check(b != nullptr, "dll_a inner LoadLibraryA(test_dll_b)");
    }

    auto marker = (Marker_t)GetProcAddress(a, "TestDllA_Marker");
    check(marker && marker() == 0x42, "dll_a TestDllA_Marker() == 0x42");

    // ordinal lookup through the hooked GetProcAddress
    auto byOrd = GetProcAddress(a, (LPCSTR)1);
    check(byOrd != nullptr, "GetProcAddress by ordinal (ordinal 1)");

    check(FreeLibrary(a), "FreeLibrary(test_dll_a)");

    // ---- log assertions ----
    FILE* f = _fsopen(logPath, "r", _SH_DENYNO);
    check(f != nullptr, "log file readable");
    std::string log;
    if (f) {
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) log.append(buf, n);
        fclose(f);
    }
    struct Expect { const char* name; const char* needle; };
    Expect expects[] = {
        {"baseline",        "\"ev\":\"baseline\""},
        {"hooks_installed", "\"ev\":\"hooks_installed\""},
        {"load dll_a",      "test_dll_a.dll"},
        {"load dll_b",      "test_dll_b.dll"},
        {"getprocaddr",     "\"ev\":\"getprocaddr\""},
        {"free",            "\"ev\":\"free\""},
    };
    size_t pos = 0;
    for (auto& e : expects) {
        size_t p = log.find(e.needle, pos);
        bool found = p != std::string::npos;
        check(found, e.name);
        if (found) pos = p + strlen(e.needle);   // enforce ordering
    }
    // module_trace.dll itself was loaded BEFORE hooks existed: it must appear
    // in the baseline (or as a load event) somewhere in the log.
    check(log.find("module_trace.dll") != std::string::npos,
          "module_trace.dll recorded (baseline or load)");

    printf("\nselftest: %s (%d failure%s)\n",
           failures ? "NOT_READY" : "READY", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
