// nvapi_trampoline_test — Round 2 Phase F: ABI self-proof for the nvapi_trace
// forwarding trampoline.
//
// Round 1 shipped the assumption "stack args ride along" (tools/nvapi_trace,
// NvFn4 casts only the first 4 register args). This test proves or disproves
// it: for N = 1..10 arguments, wrap a capture function in the SAME stub/thunk
// chain nvapi_QueryInterface hands out, call it with per-position bit
// patterns, and demand every argument + the return value arrive BYTE-EXACT.
//
// Exit 0 = all byte-exact (>= 8 and 10 arg cases mandatory). Exit 1 = any
// mismatch (then the plan's fallback applies: per-function typed shims).
//
// IMPORTANT ABI note (discovered by this very test): the stub is a `jmp` chain
// into a C dispatcher whose spilled parameters occupy the CALLER's 32-byte
// home space. x64 callers are REQUIRED to reserve it; a non-conformant direct
// call (no home space) destroys stack args 5+ (reproduced, n>=3 all corrupt).
// Real nvapi callers (driver/NGX/loader) conform, so the measurements below
// use variadic caller stubs, which the compiler emits with reserved home space.
// Production caveat stays documented: trampoline correctness depends on
// ABI-conformant callers.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>

typedef void* (*MakeTrampoline_t)(uint32_t id, void* realFn);

static uint64_t g_captured[16];

// capture functions: record all N args, return a distinctive pattern
extern "C" uint64_t Capture1(uint64_t a1) {
    g_captured[0] = a1; return 0xA100000000000001ull;
}
extern "C" uint64_t Capture2(uint64_t a1, uint64_t a2) {
    g_captured[0] = a1; g_captured[1] = a2; return 0xA200000000000002ull;
}
extern "C" uint64_t Capture3(uint64_t a1, uint64_t a2, uint64_t a3) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3;
    return 0xA300000000000003ull;
}
extern "C" uint64_t Capture4(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    return 0xA400000000000004ull;
}
extern "C" uint64_t Capture5(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                             uint64_t a5) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    g_captured[4] = a5; return 0xA500000000000005ull;
}
extern "C" uint64_t Capture6(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                             uint64_t a5, uint64_t a6) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    g_captured[4] = a5; g_captured[5] = a6; return 0xA600000000000006ull;
}
extern "C" uint64_t Capture7(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                             uint64_t a5, uint64_t a6, uint64_t a7) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    g_captured[4] = a5; g_captured[5] = a6; g_captured[6] = a7;
    return 0xA700000000000007ull;
}
extern "C" uint64_t Capture8(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                             uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    g_captured[4] = a5; g_captured[5] = a6; g_captured[6] = a7; g_captured[7] = a8;
    return 0xA800000000000008ull;
}
extern "C" uint64_t Capture9(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                             uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8,
                             uint64_t a9) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    g_captured[4] = a5; g_captured[5] = a6; g_captured[6] = a7; g_captured[7] = a8;
    g_captured[8] = a9; return 0xA900000000000009ull;
}
extern "C" uint64_t Capture10(uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                              uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8,
                              uint64_t a9, uint64_t a10) {
    g_captured[0] = a1; g_captured[1] = a2; g_captured[2] = a3; g_captured[3] = a4;
    g_captured[4] = a5; g_captured[5] = a6; g_captured[6] = a7; g_captured[7] = a8;
    g_captured[8] = a9; g_captured[9] = a10; return 0xAA0000000000000Aull;
}

typedef uint64_t (*Fn1)(uint64_t);
typedef uint64_t (*Fn2)(uint64_t, uint64_t);
typedef uint64_t (*Fn3)(uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn4)(uint64_t, uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn5)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn6)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn7)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn8)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn9)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
typedef uint64_t (*Fn10)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

// Variadic caller stubs: the compiler reserves the 32-byte home space (as
// every ABI-conformant caller must), then forwards everything to the target.
extern "C" uint64_t CallStubV(void* target, int n, ...) {
    va_list ap;
    va_start(ap, n);
    uint64_t a[10];
    for (int i = 0; i < n && i < 10; ++i) a[i] = va_arg(ap, uint64_t);
    va_end(ap);
    switch (n) {
        case 1:  return ((Fn1)target)(a[0]);
        case 2:  return ((Fn2)target)(a[0], a[1]);
        case 3:  return ((Fn3)target)(a[0], a[1], a[2]);
        case 4:  return ((Fn4)target)(a[0], a[1], a[2], a[3]);
        case 5:  return ((Fn5)target)(a[0], a[1], a[2], a[3], a[4]);
        case 6:  return ((Fn6)target)(a[0], a[1], a[2], a[3], a[4], a[5]);
        case 7:  return ((Fn7)target)(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
        case 8:  return ((Fn8)target)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        case 9:  return ((Fn9)target)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
        case 10: return ((Fn10)target)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
    }
    return 0;
}

// per-position bit patterns (no two equal, stack args distinguishable)
static uint64_t Pattern(int n, int slot) {
    return 0x1000000000000000ull * (uint64_t)(n) |
           0x00D0C0B0A0908070ull ^ (0x0102040810204080ull >> (slot & 7)) |
           ((uint64_t)(slot + 1) << 40);
}
static uint64_t ExpectedRet(int n) { return 0xA000000000000000ull | ((uint64_t)n << 56) | (uint64_t)n; }

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: nvapi_trampoline_test.exe <nvapi64.dll shim path>\n");
        return 2;
    }
    SetEnvironmentVariableA("NVAPI_TRACE_LOG", "nvapi_trampoline_test.jsonl");
    DeleteFileA("nvapi_trampoline_test.jsonl");
    setvbuf(stdout, nullptr, _IONBF, 0);   // survive crashes: keep output

    HMODULE shim = LoadLibraryA(argv[1]);
    if (!shim) { printf("FAIL: LoadLibrary(%s) lasterr=%lu\n", argv[1], GetLastError()); return 1; }
    auto make = (MakeTrampoline_t)GetProcAddress(shim, "NvapiTrace_MakeTrampoline");
    if (!make) { printf("FAIL: NvapiTrace_MakeTrampoline export missing\n"); return 1; }

    void* realFns[11] = { nullptr, (void*)Capture1, (void*)Capture2, (void*)Capture3,
                          (void*)Capture4, (void*)Capture5, (void*)Capture6,
                          (void*)Capture7, (void*)Capture8, (void*)Capture9,
                          (void*)Capture10 };
    int failures = 0;
    for (int n = 1; n <= 10; ++n) {
        memset(g_captured, 0, sizeof(g_captured));
        void* stub = make(0xF0000000u | (uint32_t)n, realFns[n]);
        uint64_t expected[10];
        for (int s = 0; s < n; ++s) expected[s] = Pattern(n, s);

        uint64_t ret = 0;
        switch (n) {
            case 1:  ret = CallStubV(stub, n, expected[0]); break;
            case 2:  ret = CallStubV(stub, n, expected[0], expected[1]); break;
            case 3:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2]); break;
            case 4:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3]); break;
            case 5:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3], expected[4]); break;
            case 6:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3], expected[4], expected[5]); break;
            case 7:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3], expected[4], expected[5], expected[6]); break;
            case 8:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3], expected[4], expected[5], expected[6], expected[7]); break;
            case 9:  ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3], expected[4], expected[5], expected[6], expected[7], expected[8]); break;
            case 10: ret = CallStubV(stub, n, expected[0], expected[1], expected[2], expected[3], expected[4], expected[5], expected[6], expected[7], expected[8], expected[9]); break;
        }

        int bad = 0;
        for (int s = 0; s < n; ++s) {
            if (memcmp(&g_captured[s], &expected[s], 8) != 0) {
                printf("  arg%d mismatch: got 0x%016llx want 0x%016llx\n",
                       s + 1, (unsigned long long)g_captured[s], (unsigned long long)expected[s]);
                ++bad;
            }
        }
        uint64_t wantRet = ExpectedRet(n);
        if (memcmp(&ret, &wantRet, 8) != 0) {
            printf("  ret mismatch: got 0x%016llx want 0x%016llx\n",
                   (unsigned long long)ret, (unsigned long long)ExpectedRet(n));
            ++bad;
        }
        printf("[%s] %2d-arg trampoline byte-exact (%d mismatches)\n",
               bad ? "FAIL" : "PASS", n, bad);
        if (bad) ++failures;
    }

    printf("\nnvapi trampoline self-proof: %s (%d failing widths)\n",
           failures ? "DISPROVED — fallback to typed shims required" : "PROVED", failures);
    return failures ? 1 : 0;
}
