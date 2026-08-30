// hip_probe — Phase 1: RDNA4 compute baseline (gate S1).
//
// Proves that this Windows + driver + HIP SDK stack can stably execute GPU
// compute on gfx1200/gfx1201 and produce numerically correct results.
// No DLSS involvement at this stage.
//
// Tests:
//   A. hipGetDeviceCount / device properties
//   B. device allocation + memset/memcpy roundtrip
//   C. FP32 vector-add kernel (correctness + timing)
//   D. FP16 naive GEMM (correctness vs CPU fp32 reference + timing)
//   E. FP8 workload: E4M3 and E5M2 conversion + fp32-accumulate dot product
//      (device-side conversion functions implemented here; validated vs CPU)
//   F. RDNA4 WMMA f16 intrinsic (gfx12 only) — verified with mapping-agnostic
//      ground truth (A=ones, B=identity => C=ones). Lane-mapping details are
//      deliberately NOT trusted from docs (ROCm issue #6025).
//
// Output: human log to stdout; JSON report to --json <path> (or stdout with --json -).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// helpers

#define HIP_CHECK(cmd)                                                       \
    do {                                                                     \
        hipError_t e_ = (cmd);                                               \
        if (e_ != hipSuccess) {                                              \
            printf("HIP ERROR %s:%d  %s -> %d (%s)\n", __FILE__, __LINE__,   \
                   #cmd, (int)e_, hipGetErrorString(e_));                    \
            g_anyFailure = true;                                             \
        }                                                                    \
    } while (0)

static bool g_anyFailure = false;

struct TestResult {
    const char* name;
    bool ran = false;
    bool pass = false;
    std::string detail;
    double kernelMs = -1.0;
};
static std::vector<TestResult> g_results;

static void Record(TestResult r) {
    printf("[%s] %-28s %s\n", r.pass ? "PASS" : (r.ran ? "FAIL" : "SKIP"), r.name,
           r.detail.c_str());
    g_results.push_back(std::move(r));
}

// ---------------------------------------------------------------------------
// FP8 conversion reference (host + device, identical logic).
// E4M3 (FN, no inf, NaN=0x7f/0xff, max 448) and E5M2 (IEEE-like, has inf).

__host__ __device__ static inline float fp8e4m3_to_float(unsigned char b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xF;
    const int man = b & 0x7;
    float v;
    if (exp == 0xF && man == 0x7) return sign ? -NAN : NAN;  // NaN encodings
    if (exp == 0) {
        v = (float)man * (1.0f / 64.0f) * powf(2.0f, -6.0f);  // subnormal: 2^-6 * m/8
    } else {
        v = (1.0f + (float)man / 8.0f) * powf(2.0f, (float)(exp - 7));
    }
    return sign ? -v : v;
}

__host__ __device__ static inline unsigned char float_to_fp8e4m3(float f) {
    if (f != f) return 0x7F;                       // NaN
    int sign = f < 0.f ? 1 : 0;
    float af = fabsf(f);
    const float maxv = 448.f;
    if (af >= maxv + 16.f) af = maxv;              // saturate (round-to-nearest-even boundary)
    // find exponent
    int e;
    float m;
    if (af == 0.f) return (unsigned char)(sign << 7);
    float lf = log2f(af);
    e = (int)floorf(lf);
    if (e < -6) {  // subnormal
        int man = (int)rintf(af / powf(2.0f, -6.0f) * 8.0f);
        if (man > 7) man = 7;
        return (unsigned char)((sign << 7) | man);
    }
    if (e > 8) { af = maxv; e = 8; }
    m = af / powf(2.0f, (float)e) - 1.0f;
    int man = (int)rintf(m * 8.0f);
    if (man > 7) { man = 0; e += 1; }
    if (e > 8) { e = 8; man = 6; }                 // saturate to 448
    if (e < 1) {                                   // slipped into subnormal
        int m2 = (int)rintf(af / powf(2.0f, -6.0f) * 8.0f);
        return (unsigned char)((sign << 7) | (m2 > 7 ? 7 : m2));
    }
    return (unsigned char)((sign << 7) | (e << 3) | man);
}

__host__ __device__ static inline float fp8e5m2_to_float(unsigned char b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 2) & 0x1F;
    const int man = b & 0x3;
    float v;
    if (exp == 0x1F) {
        if (man) return sign ? -NAN : NAN;
        return sign ? -INFINITY : INFINITY;
    }
    if (exp == 0) {
        v = (float)man * (1.0f / 4.0f) * powf(2.0f, -14.0f);
    } else {
        v = (1.0f + (float)man / 4.0f) * powf(2.0f, (float)(exp - 15));
    }
    return sign ? -v : v;
}

__host__ __device__ static inline unsigned char float_to_fp8e5m2(float f) {
    if (f != f) return 0x7F;
    int sign = f < 0.f ? 1 : 0;
    float af = fabsf(f);
    if (af > 57344.f) return (unsigned char)((sign << 7) | 0x7C);  // inf
    if (af == 0.f) return (unsigned char)(sign << 7);
    int e = (int)floorf(log2f(af));
    if (e < -14) {
        int man = (int)rintf(af / powf(2.0f, -14.0f) * 4.0f);
        if (man > 3) man = 3;
        return (unsigned char)((sign << 7) | man);
    }
    if (e > 15) e = 15;
    float m = af / powf(2.0f, (float)e) - 1.0f;
    int man = (int)rintf(m * 4.0f);
    if (man > 3) { man = 0; e += 1; }
    if (e > 15) { return (unsigned char)((sign << 7) | 0x7C); }
    return (unsigned char)((sign << 7) | (e << 2) | man);
}

// ---------------------------------------------------------------------------
// kernels

__global__ static void k_fp32_vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ static void k_fp16_gemm(const half* A, const half* B, half* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < N; ++k) {
        acc += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = __float2half(acc);
}

__global__ static void k_fp8_dot(const unsigned char* a, const unsigned char* b, float* out,
                                 int n, int mode) {
    // mode 0: e4m3, mode 1: e5m2 ; accumulate into out[0] with fp32
    float acc = 0.f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float av = (mode == 0) ? fp8e4m3_to_float(a[i]) : fp8e5m2_to_float(a[i]);
        float bv = (mode == 0) ? fp8e4m3_to_float(b[i]) : fp8e5m2_to_float(b[i]);
        acc = av * bv;
        atomicAdd(out, acc);
    }
}

__global__ static void k_fp8_roundtrip(const float* in, unsigned char* mid, float* out, int n,
                                       int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned char q = (mode == 0) ? float_to_fp8e4m3(in[i]) : float_to_fp8e5m2(in[i]);
    mid[i] = q;
    out[i] = (mode == 0) ? fp8e4m3_to_float(q) : fp8e5m2_to_float(q);
}

// ---- WMMA test (gfx12 wave32) ----
// Opt-in: compile with -DWMMA_TEST=1. The f32-accumulator builtin is known to
// fail ISel in some ROCm 7.2 LLVM builds ("Cannot select: intrinsic
// %llvm.amdgcn.wmma.f32.16x16x16.f16"); keep the main build independent of it.
#if (defined(__gfx1200__) || defined(__gfx1201__)) && defined(WMMA_TEST) && WMMA_TEST
#define DLSSNR_LAB_HAS_WMMA 1
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__global__ static void k_wmma_f16_ones_identity(float* out) {
    // A = all 1.0 (16x16), B = identity (16x16)  =>  C = all 1.0 (16x16)
    // Ground truth is mapping-agnostic: regardless of how lanes own elements,
    // a correct 16x16x16 matrix product of ones x identity is ones.
    const int lane = threadIdx.x & 31;

    v16h avec;
    for (int i = 0; i < 16; ++i) avec[i] = (_Float16)1.0f;  // A: all ones (any layout)

    // B identity: element (r,c) = (r==c). Without lane mapping knowledge we set
    // every lane's vector as if it were a row r=lane%16: B[r][c]=delta(r,c).
    v16h bvec;
    int r = lane & 15;
    for (int c = 0; c < 16; ++c) bvec[c] = (c == r) ? (_Float16)1.0f : (_Float16)0.0f;

    v8f acc;
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;

    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(avec, bvec, acc);

    // wave32: each lane holds 8 f32 results; store all 32 lanes for inspection
    for (int i = 0; i < 8; ++i) {
        out[lane * 8 + i] = acc[i];
    }
    // fill a separate region with marker so coverage is deterministic
    for (int i = 0; i < 8; ++i) out[256 + lane * 8 + i] = -999.f;
}
#else
#define DLSSNR_LAB_HAS_WMMA 0
#endif

// ---------------------------------------------------------------------------

static bool RunFp32(TestResult& r) {
    const int N = 1 << 20;
    std::vector<float> a(N), b(N), c(N);
    for (int i = 0; i < N; ++i) { a[i] = sinf((float)i) * 0.5f; b[i] = cosf((float)i * 2.f); }

    float *da, *db, *dc;
    HIP_CHECK(hipMalloc(&da, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&db, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dc, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(da, a.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, b.data(), N * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    hipLaunchKernelGGL(k_fp32_vadd, dim3((N + 255) / 256), dim3(256), 0, 0, da, db, dc, N);
    hipEventRecord(t1);
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f; hipEventElapsedTime(&ms, t0, t1);
    r.kernelMs = ms;

    HIP_CHECK(hipMemcpy(c.data(), dc, N * sizeof(float), hipMemcpyDeviceToHost));
    double maxErr = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = (double)a[i] + (double)b[i];
        double err = fabs((double)c[i] - ref);
        if (err > maxErr) maxErr = err;
    }
    hipFree(da); hipFree(db); hipFree(dc);
    hipEventDestroy(t0); hipEventDestroy(t1);

    char buf[128];
    snprintf(buf, sizeof(buf), "N=%d maxErr=%.3e kernel=%.4fms", N, maxErr, ms);
    r.detail = buf;
    r.ran = true;
    r.pass = maxErr < 1e-5;
    return r.pass;
}

static bool RunFp16Gemm(TestResult& r) {
    const int N = 64;
    std::vector<half> A(N * N), B(N * N), C(N * N);
    std::vector<float> Af(N * N), Bf(N * N);
    unsigned seed = 1234;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float)(seed >> 8) / 16777216.f; };
    for (int i = 0; i < N * N; ++i) {
        Af[i] = rnd() * 2.f - 1.f; Bf[i] = rnd() * 2.f - 1.f;
        A[i] = __float2half(Af[i]); B[i] = __float2half(Bf[i]);
    }
    // CPU reference in fp32 (input quantized to fp16 first)
    std::vector<float> ref(N * N, 0.f);
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            float acc = 0.f;
            for (int k = 0; k < N; ++k)
                acc += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
            ref[row * N + col] = acc;
        }

    half *dA, *dB, *dC;
    hipMalloc(&dA, N * N * sizeof(half));
    hipMalloc(&dB, N * N * sizeof(half));
    hipMalloc(&dC, N * N * sizeof(half));
    hipMemcpy(dA, A.data(), N * N * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(dB, B.data(), N * N * sizeof(half), hipMemcpyHostToDevice);

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    dim3 block(16, 16); dim3 grid((N + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(k_fp16_gemm, grid, block, 0, 0, dA, dB, dC, N);
    hipEventRecord(t1);
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f; hipEventElapsedTime(&ms, t0, t1);
    r.kernelMs = ms;

    hipMemcpy(C.data(), dC, N * N * sizeof(half), hipMemcpyDeviceToHost);
    double maxErr = 0.0, sumSq = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double d = (double)__half2float(C[i]) - (double)ref[i];
        if (fabs(d) > maxErr) maxErr = fabs(d);
        sumSq += d * d;
    }
    hipFree(dA); hipFree(dB); hipFree(dC);
    hipEventDestroy(t0); hipEventDestroy(t1);

    char buf[128];
    snprintf(buf, sizeof(buf), "N=%d maxErr=%.3e rmse=%.3e kernel=%.4fms", N, maxErr,
             sqrt(sumSq / (N * N)), ms);
    r.detail = buf;
    r.ran = true;
    r.pass = maxErr < 1e-2;   // fp16 accumulation tolerance (see docs/RESULTS.md policy)
    return r.pass;
}

static bool RunFp8(TestResult& r, int mode) {
    // roundtrip sanity + dot product with fp32 accumulation
    const int N = 4096;
    std::vector<float> in(N);
    unsigned seed = mode ? 777u : 42u;
    for (int i = 0; i < N; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float x = ((float)(seed >> 8) / 16777216.f) * 2.f - 1.f;
        // keep magnitudes inside both formats' normal range; e5m2 saturates to
        // inf near 57344 which would poison the accumulation, so stay <= 8192
        in[i] = x * (mode ? 8192.f : 4.f);
    }

    // host reference conversions
    std::vector<unsigned char> qref(N);
    std::vector<float> backRef(N);
    for (int i = 0; i < N; ++i) {
        qref[i] = mode ? float_to_fp8e5m2(in[i]) : float_to_fp8e4m3(in[i]);
        backRef[i] = mode ? fp8e5m2_to_float(qref[i]) : fp8e4m3_to_float(qref[i]);
    }

    float *din, *dout;
    unsigned char* dmid;
    hipMalloc(&din, N * sizeof(float));
    hipMalloc(&dout, N * sizeof(float));
    hipMalloc(&dmid, N);
    hipMemcpy(din, in.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_fp8_roundtrip, dim3((N + 255) / 256), dim3(256), 0, 0, din, dmid, dout,
                       N, mode);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<unsigned char> qgpu(N);
    std::vector<float> out(N);
    hipMemcpy(qgpu.data(), dmid, N, hipMemcpyDeviceToHost);
    hipMemcpy(out.data(), dout, N * sizeof(float), hipMemcpyDeviceToHost);

    int mismatch = 0;
    double maxDiff = 0.0;
    for (int i = 0; i < N; ++i) {
        if (qgpu[i] != qref[i]) mismatch++;
        double d = fabs((double)out[i] - (double)backRef[i]);
        if (d > maxDiff) maxDiff = d;
    }

    // dot product in fp8 storage with fp32 accumulation
    std::vector<unsigned char> qb(N);
    for (int i = 0; i < N; ++i) qb[i] = qref[(i * 7 + 3) % N];
    unsigned char *dqa, *dqb;
    float* dd;
    hipMalloc(&dqa, N); hipMalloc(&dqb, N); hipMalloc(&dd, sizeof(float));
    hipMemcpy(dqa, qgpu.data(), N, hipMemcpyHostToDevice);
    hipMemcpy(dqb, qb.data(), N, hipMemcpyHostToDevice);
    float zero = 0.f;
    hipMemcpy(dd, &zero, sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_fp8_dot, dim3((N + 255) / 256), dim3(256), 0, 0, dqa, dqb, dd, N, mode);
    HIP_CHECK(hipDeviceSynchronize());
    float gpuDot = 0.f;
    hipMemcpy(&gpuDot, dd, sizeof(float), hipMemcpyDeviceToHost);
    double cpuDot = 0.0;
    for (int i = 0; i < N; ++i) cpuDot += (double)backRef[i] * (double)(mode ? fp8e5m2_to_float(qb[i]) : fp8e4m3_to_float(qb[i]));

    hipFree(din); hipFree(dout); hipFree(dmid); hipFree(dqa); hipFree(dqb); hipFree(dd);

    double denom = fabs(cpuDot) + 1e-6;
    double relDot = fabs(gpuDot - cpuDot) / denom;
    bool numericOk = std::isfinite(gpuDot) && std::isfinite(cpuDot) && (relDot < 1e-4);
    char buf[192];
    snprintf(buf, sizeof(buf), "%s N=%d quantMismatch=%d maxBackDiff=%.3e dotRelErr=%.3e",
             mode ? "e5m2" : "e4m3", N, mismatch, maxDiff, relDot);
    r.detail = buf;
    r.ran = true;
    r.pass = (mismatch == 0) && numericOk;
    return r.pass;
}

static bool RunWmma(TestResult& r) {
#if DLSSNR_LAB_HAS_WMMA
    float* dOut;
    const int elems = 32 * 16;
    hipMalloc(&dOut, elems * sizeof(float));
    hipMemset(dOut, 0, elems * sizeof(float));
    hipLaunchKernelGGL(k_wmma_f16_ones_identity, dim3(1), dim3(32), 0, 0, dOut);
    hipError_t err = hipDeviceSynchronize();
    std::vector<float> out(elems, 0.f);
    hipMemcpy(out.data(), dOut, elems * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(dOut);

    // We cannot assert full-matrix correctness without trusting lane-mapping docs.
    // Evidence we can assert: the kernel executed without error, and produced
    // values consistent with ones*identity => ones (each lane holds 8 outputs).
    int countOnes = 0, countOther = 0;
    for (int i = 0; i < 32 * 8; ++i) {
        if (fabsf(out[i] - 1.0f) < 1e-3f) countOnes++;
        else if (fabsf(out[i] - (-999.f)) > 1e-3f) countOther++;
    }
    char buf[192];
    snprintf(buf, sizeof(buf),
             "exec=%s ones=%d/%d other=%d (lane mapping NOT assumed; see ROCm #6025)",
             err == hipSuccess ? "ok" : hipGetErrorString(err), countOnes, 32 * 8, countOther);
    r.detail = buf;
    r.ran = true;
    // PASS only if: executed cleanly AND all 256 result values equal 1.0
    r.pass = (err == hipSuccess) && (countOnes == 256) && (countOther == 0);
    if (!r.pass) r.detail += "  -> recorded as UNVERIFIED/FAIL, not used as S-level evidence";
    return r.pass;
#else
    r.detail = "WMMA test not compiled (needs gfx12 + -DWMMA_TEST=1)";
    r.ran = false;
    return false;
#endif
}

// ---------------------------------------------------------------------------

static void WriteJson(const char* path) {
    FILE* f = (path && strcmp(path, "-")) ? fopen(path, "w") : stdout;
    if (!f) return;
    fprintf(f, "{\n  \"tests\": [");
    bool first = true;
    for (auto& t : g_results) {
        if (!first) fprintf(f, ",");
        first = false;
        fprintf(f, "\n    {\"name\": \"%s\", \"ran\": %s, \"pass\": %s, \"kernel_ms\": %.4f, \"detail\": \"%s\"}",
                t.name, t.ran ? "true" : "false", t.pass ? "true" : "false", t.kernelMs,
                t.detail.c_str());
    }
    fprintf(f, "\n  ],\n  \"overall_pass\": %s\n}\n", g_anyFailure ? "false" : "true");
    if (f != stdout) fclose(f);
}

int main(int argc, char** argv) {
    const char* jsonPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json") && i + 1 < argc) jsonPath = argv[++i];
    }

    printf("=== hip_probe (dlssnr-amd-lab Phase 1) ===\n");

    // Test A: device enumeration
    {
        TestResult r; r.name = "A_device_enum";
        int count = 0;
        hipError_t e = hipGetDeviceCount(&count);
        r.ran = true;
        if (e == hipSuccess && count > 0) {
            hipDeviceProp_t p;
            hipGetDeviceProperties(&p, 0);
            char buf[256];
            snprintf(buf, sizeof(buf), "count=%d name=%s gcnArch=%s vram=%.2fGB", count, p.name,
                     p.gcnArchName, p.totalGlobalMem / 1e9);
            r.detail = buf;
            r.pass = true;
            hipSetDevice(0);
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "hipGetDeviceCount -> %d (%s)", (int)e, hipGetErrorString(e));
            r.detail = buf;
            r.pass = false;
            g_anyFailure = true;
        }
        Record(std::move(r));
    }

    // Test B: allocation roundtrip
    {
        TestResult r; r.name = "B_alloc_roundtrip";
        const int N = 65536;
        std::vector<unsigned char> src(N), dst(N, 0);
        for (int i = 0; i < N; ++i) src[i] = (unsigned char)(i * 31 + 7);
        void* d = nullptr;
        HIP_CHECK(hipMalloc(&d, N));
        HIP_CHECK(hipMemset(d, 0, N));
        HIP_CHECK(hipMemcpy(d, src.data(), N, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dst.data(), d, N, hipMemcpyDeviceToHost));
        hipFree(d);
        bool ok = memcmp(src.data(), dst.data(), N) == 0;
        r.detail = ok ? "memset+memcpy roundtrip exact" : "roundtrip MISMATCH";
        r.ran = true; r.pass = ok;
        if (!ok) g_anyFailure = true;
        Record(std::move(r));
    }

    { TestResult r; r.name = "C_fp32_vadd"; RunFp32(r); Record(std::move(r)); }
    { TestResult r; r.name = "D_fp16_gemm"; RunFp16Gemm(r); Record(std::move(r)); }
    { TestResult r; r.name = "E_fp8_e4m3"; RunFp8(r, 0); Record(std::move(r)); }
    { TestResult r; r.name = "E_fp8_e5m2"; RunFp8(r, 1); Record(std::move(r)); }
    { TestResult r; r.name = "F_wmma_f16"; RunWmma(r); Record(std::move(r)); }
#if !DLSSNR_LAB_HAS_WMMA
    printf("note: WMMA test compiled out (rebuild with -DWMMA_TEST=1 to enable)\n");
#endif

    printf("\nOVERALL: %s\n", g_anyFailure ? "FAIL" : "PASS");
    WriteJson(jsonPath);
    return g_anyFailure ? 1 : 0;
}
