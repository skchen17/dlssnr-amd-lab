// ngx_abi_probe — Round 2 Phase B: NGX ABI regression probe.
//
// Compile-time layer (always runs, no DLL needed):
//   - static_assert on every NVSDK_NGX_Result enumerator (incl. 0xBAD00001
//     == FAIL_FeatureNotSupported) and NVSDK_NGX_Feature values
//   - sizeof() checks on FeatureCommonInfo / FeatureDiscoveryInfo / ...
//   - std::is_same_v checks that our runtime function-pointer typedefs are
//     byte-identical to the official prototypes in nvsdk_ngx.h
//
// Runtime layer (only when a DLL is available, env DLSS_DLL_PATH or --dll):
//   - enumerates the PE export table and records per-expected-export
//     present/absent (snippet-branch DLLs legitimately lack the parameter
//     lifecycle exports — see docs/NGX_ABI_AUDIT.md E7)
//
// Output: JSON report (stdout and --json <path>). NO hard-coded PASS when the
// DLL is absent: the runtime section is then BLOCKED_MISSING_PREREQUISITE and
// only the compile-time layer counts.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "nvsdk_ngx.h"

// ----------------------------------------------------------------------------
// 1. Enum value regression
// ----------------------------------------------------------------------------
static_assert(NVSDK_NGX_Result_Success == 0x1, "ABI regression: Success");
static_assert(NVSDK_NGX_Result_Fail == 0xBAD00000, "ABI regression: Fail base");
static_assert(NVSDK_NGX_Result_FAIL_FeatureNotSupported == 0xBAD00001, "ABI regression: 0xBAD00001");
static_assert(NVSDK_NGX_Result_FAIL_PlatformError == 0xBAD00002, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_FeatureAlreadyExists == 0xBAD00003, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_FeatureNotFound == 0xBAD00004, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_InvalidParameter == 0xBAD00005, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall == 0xBAD00006, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_NotInitialized == 0xBAD00007, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_UnsupportedInputFormat == 0xBAD00008, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_RWFlagMissing == 0xBAD00009, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_MissingInput == 0xBAD0000A, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature == 0xBAD0000B, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_OutOfDate == 0xBAD0000C, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_OutOfGPUMemory == 0xBAD0000D, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_UnsupportedFormat == 0xBAD0000E, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath == 0xBAD0000F, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_UnsupportedParameter == 0xBAD00010, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_Denied == 0xBAD00011, "ABI regression");
static_assert(NVSDK_NGX_Result_FAIL_NotImplemented == 0xBAD00012, "ABI regression");

static_assert(NVSDK_NGX_VERSION_API_MACRO == 0x0000015, "ABI regression: API version macro");
static_assert(NVSDK_NGX_Version_API == NVSDK_NGX_VERSION_API_MACRO, "ABI regression: Version enum");

static_assert(NVSDK_NGX_Feature_Reserved0 == 0, "ABI regression: feature 0");
static_assert(NVSDK_NGX_Feature_SuperSampling == 1, "ABI regression: DLSS feature id");
static_assert(NVSDK_NGX_Feature_InPainting == 2, "ABI regression");
static_assert(NVSDK_NGX_Feature_ImageSuperResolution == 3, "ABI regression");
static_assert(NVSDK_NGX_Feature_FrameGeneration == 11, "ABI regression");
static_assert(NVSDK_NGX_Feature_RayReconstruction == 13, "ABI regression");
static_assert(NVSDK_NGX_Feature_Reserved18 == 18, "ABI regression: feature 18 stays reserved in the public enum");

static_assert(NVSDK_NGX_PerfQuality_Value_MaxPerf == 0, "ABI regression");
static_assert(NVSDK_NGX_PerfQuality_Value_Balanced == 1, "ABI regression");
static_assert(NVSDK_NGX_PerfQuality_Value_MaxQuality == 2, "ABI regression");
static_assert(NVSDK_NGX_PerfQuality_Value_UltraPerformance == 3, "ABI regression");
static_assert(NVSDK_NGX_PerfQuality_Value_UltraQuality == 4, "ABI regression");
static_assert(NVSDK_NGX_PerfQuality_Value_DLAA == 5, "ABI regression: DLAA mode value");

static_assert(NVSDK_NGX_DLSS_Feature_Flags_None == 0, "ABI regression");
static_assert(NVSDK_NGX_DLSS_Feature_Flags_IsHDR == (1 << 0), "ABI regression");
static_assert(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes == (1 << 1), "ABI regression");
static_assert(NVSDK_NGX_DLSS_Feature_Flags_MVJittered == (1 << 2), "ABI regression");
static_assert(NVSDK_NGX_DLSS_Feature_Flags_DepthInverted == (1 << 3), "ABI regression");
static_assert(NVSDK_NGX_DLSS_Feature_Flags_AutoExposure == (1 << 6), "ABI regression");
static_assert(NVSDK_NGX_DLSS_Feature_Flags_AlphaUpscaling == (1 << 7), "ABI regression");

// ----------------------------------------------------------------------------
// 2. Struct size / layout regression (x64)
// ----------------------------------------------------------------------------
static_assert(sizeof(NVSDK_NGX_Coordinates) == 8, "ABI regression");
static_assert(sizeof(NVSDK_NGX_Dimensions) == 8, "ABI regression");
static_assert(sizeof(NVSDK_NGX_PrecisionInfo) == 12, "ABI regression");
static_assert(sizeof(NVSDK_NGX_PathListInfo) == 16, "ABI regression");
static_assert(sizeof(NVSDK_NGX_LoggingInfo) == 16, "ABI regression");
static_assert(sizeof(NVSDK_NGX_FeatureCommonInfo) == 40, "ABI regression");
static_assert(sizeof(NVSDK_NGX_ProjectIdDescription) == 24, "ABI regression");
static_assert(sizeof(NVSDK_NGX_Application_Identifier) == 32, "ABI regression");
static_assert(sizeof(NVSDK_NGX_FeatureDiscoveryInfo) == 56, "ABI regression");
static_assert(sizeof(NVSDK_NGX_FeatureRequirement) == 264, "ABI regression");
static_assert(sizeof(NVSDK_NGX_Feature_Create_Params) == 20, "ABI regression");
static_assert(sizeof(NVSDK_NGX_DLSS_Create_Params) == 28, "ABI regression");

// ----------------------------------------------------------------------------
// 3. Function-pointer prototype regression.
//    Take the address of each official declaration and assert its decayed type
//    is identical to the typedef we use for GetProcAddress in nr_host.
// ----------------------------------------------------------------------------
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_Init_Loader)(unsigned long long, const wchar_t*, ID3D12Device*, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_Init_Snippet)(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_Init_with_ProjectID)(const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*, ID3D12Device*, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_AllocateParameters)(NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_GetCapabilityParameters)(NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_DestroyParameters)(NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_Shutdown1)(ID3D12Device*);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_GetScratchBufferSize)(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t*);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_CreateFeature)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_ReleaseFeature)(NVSDK_NGX_Handle*);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_EvaluateFeature_C)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback_C);
typedef NVSDK_NGX_Result (NVSDK_CONV* PFN_GetFeatureRequirements)(IDXGIAdapter*, const NVSDK_NGX_FeatureDiscoveryInfo*, NVSDK_NGX_FeatureRequirement*);
typedef const wchar_t*   (NVSDK_CONV* PFN_ResultAsString)(NVSDK_NGX_Result);

static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_Init), PFN_Init_Loader>,
              "official Init (loader branch) prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_Init_with_ProjectID), PFN_Init_with_ProjectID>,
              "Init_with_ProjectID prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_AllocateParameters), PFN_AllocateParameters>,
              "AllocateParameters prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_GetCapabilityParameters), PFN_GetCapabilityParameters>,
              "GetCapabilityParameters prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_DestroyParameters), PFN_DestroyParameters>,
              "DestroyParameters prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_Shutdown1), PFN_Shutdown1>,
              "Shutdown1 prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_GetScratchBufferSize), PFN_GetScratchBufferSize>,
              "GetScratchBufferSize prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_CreateFeature), PFN_CreateFeature>,
              "CreateFeature prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_ReleaseFeature), PFN_ReleaseFeature>,
              "ReleaseFeature prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_EvaluateFeature_C), PFN_EvaluateFeature_C>,
              "EvaluateFeature_C prototype drifted");
static_assert(std::is_same_v<decltype(&NVSDK_NGX_D3D12_GetFeatureRequirements), PFN_GetFeatureRequirements>,
              "GetFeatureRequirements prototype drifted");
static_assert(std::is_same_v<decltype(&GetNGXResultAsString), PFN_ResultAsString>,
              "GetNGXResultAsString prototype drifted");

// ----------------------------------------------------------------------------
// 4. Runtime export-table check
// ----------------------------------------------------------------------------
struct ExportProbe {
    const char* name;
    bool required;    // required for the minimal create/evaluate pipeline
    bool present;
};

static std::vector<std::string> EnumerateExports(const char* dllPath) {
    std::vector<std::string> names;
    HANDLE f = CreateFileA(dllPath, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return names;
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(f);
    if (!m) return names;
    const uint8_t* base = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (!base) return names;
    do {
        auto rd32 = [&](size_t off) { return *(const uint32_t*)(base + off); };
        auto rd16 = [&](size_t off) { return *(const uint16_t*)(base + off); };
        if (rd16(0) != 0x5A4D) break;                       // MZ
        size_t peOff = rd32(0x3C);
        if (memcmp(base + peOff, "PE\0\0", 4) != 0) break;
        if (rd16(peOff + 24) != 0x20B) break;               // PE32+
        uint16_t nsec = rd16(peOff + 6);
        uint16_t sizeOpt = rd16(peOff + 20);
        size_t secOff = peOff + 24 + sizeOpt;
        uint32_t expRva = rd32(peOff + 24 + 112);
        if (!expRva) break;
        auto rvaToOff = [&](uint32_t rva) -> size_t {
            for (uint16_t i = 0; i < nsec; ++i) {
                size_t o = secOff + (size_t)i * 40;
                uint32_t vs = rd32(o + 8), va = rd32(o + 12), ro = rd32(o + 20);
                if (rva >= va && rva < va + vs) return ro + (rva - va);
            }
            return (size_t)-1;
        };
        size_t eo = rvaToOff(expRva);
        if (eo == (size_t)-1) break;
        uint32_t nNames = rd32(eo + 24);
        size_t no = rvaToOff(rd32(eo + 32));
        if (no == (size_t)-1) break;
        for (uint32_t i = 0; i < nNames; ++i) {
            size_t so = rvaToOff(rd32(no + i * 4));
            if (so == (size_t)-1) continue;
            names.push_back((const char*)(base + so));
        }
    } while (false);
    UnmapViewOfFile(base);
    return names;
}

static void JsonEscape(std::string& s) {
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i] == '"' || s[i] == '\\') { s.insert(i, "\\"); ++i; }
}

int main(int argc, char** argv) {
    const char* dll = nullptr;
    const char* jsonPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dll") && i + 1 < argc) dll = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) jsonPath = argv[++i];
    }
    if (!dll) dll = getenv("DLSS_DLL_PATH");

    printf("=== ngx_abi_probe (dlssnr-amd-lab Round 2 Phase B) ===\n");
    // Compile-time layer: reaching this point means every static_assert passed.
    printf("compile_time: PASS (enums=%d checked, structs=%zu sizes, prototypes=12, all static_assert held)\n",
           19, sizeof(void*) == 8 ? (size_t)12 : (size_t)0);

    std::string report;
    report += "{\n  \"tool\": \"ngx_abi_probe\",\n  \"compile_time\": \"PASS\",\n";
    report += "  \"api_version_macro\": 21,\n";

    int exitCode = 0;
    if (!dll || !*dll || GetFileAttributesA(dll) == INVALID_FILE_ATTRIBUTES) {
        printf("runtime: BLOCKED_MISSING_PREREQUISITE (no DLL; set DLSS_DLL_PATH or --dll)\n");
        report += "  \"runtime\": \"BLOCKED_MISSING_PREREQUISITE\",\n  \"dll\": null\n}\n";
        exitCode = 3;
    } else {
        auto names = EnumerateExports(dll);
        if (names.empty()) {
            printf("runtime: FAIL (cannot parse PE export table of %s)\n", dll);
            report += "  \"runtime\": \"FAIL_EXPORT_PARSE\",\n";
            std::string d = dll; JsonEscape(d);
            report += "  \"dll\": \"" + d + "\"\n}\n";
            exitCode = 4;
        } else {
            ExportProbe probes[] = {
                {"NVSDK_NGX_D3D12_Init", true, false},
                {"NVSDK_NGX_D3D12_Init_Ext", false, false},
                {"NVSDK_NGX_D3D12_Init_with_ProjectID", false, false},
                {"NVSDK_NGX_D3D12_Shutdown1", false, false},
                {"NVSDK_NGX_D3D12_AllocateParameters", false, false},
                {"NVSDK_NGX_D3D12_GetCapabilityParameters", false, false},
                {"NVSDK_NGX_D3D12_DestroyParameters", false, false},
                {"NVSDK_NGX_D3D12_GetScratchBufferSize", false, false},
                {"NVSDK_NGX_D3D12_CreateFeature", true, false},
                {"NVSDK_NGX_D3D12_EvaluateFeature", true, false},
                {"NVSDK_NGX_D3D12_EvaluateFeature_C", false, false},
                {"NVSDK_NGX_D3D12_ReleaseFeature", true, false},
                {"NVSDK_NGX_D3D12_GetFeatureRequirements", false, false},
                {"NVSDK_NGX_UpdateFeature", false, false},
                {"GetNGXResultAsString", false, false},
                {"NGX_SNIPPETS_GetRequiredDriverSupport", false, false},
            };
            for (auto& p : probes)
                for (auto& n : names)
                    if (n == p.name) { p.present = true; break; }
            bool requiredOk = true;
            for (auto& p : probes)
                if (p.required && !p.present) requiredOk = false;
            bool isSnippet = false;
            for (auto& p : probes)
                if (!strcmp(p.name, "NGX_SNIPPETS_GetRequiredDriverSupport")) isSnippet = p.present;
            printf("runtime: exports=%zu required_ok=%d abi_branch=%s\n", names.size(), requiredOk ? 1 : 0,
                   isSnippet ? "snippet(NGX_SNIPPET_BUILD)" : "loader");
            for (auto& p : probes)
                printf("  [%s] %s%s\n", p.present ? "x" : " ", p.name, p.required ? " (required)" : "");
            std::string d = dll; JsonEscape(d);
            report += "  \"runtime\": \"";
            report += requiredOk ? "PASS" : "FAIL_MISSING_REQUIRED_EXPORT";
            report += "\",\n";
            report += "  \"dll\": \"" + d + "\",\n";
            report += std::string("  \"abi_branch\": \"") + (isSnippet ? "snippet" : "loader") + "\",\n";
            report += "  \"export_count\": " + std::to_string(names.size()) + ",\n";
            report += "  \"exports\": {\n";
            for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
                report += "    \"" + std::string(probes[i].name) + "\": " + (probes[i].present ? "true" : "false");
                report += i + 1 < sizeof(probes) / sizeof(probes[0]) ? ",\n" : "\n";
            }
            report += "  }\n}\n";
            exitCode = requiredOk ? 0 : 4;
        }
    }

    printf("\n%s\n", report.c_str());
    if (jsonPath) {
        FILE* f = fopen(jsonPath, "w");
        if (f) { fputs(report.c_str(), f); fclose(f); printf("json written: %s\n", jsonPath); }
    }
    return exitCode;
}
