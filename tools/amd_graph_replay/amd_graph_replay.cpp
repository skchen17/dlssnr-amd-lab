// amd_graph_replay — first AMD-side experiment after RTX graph capture.
//
// Replays the captured 156-slot launch geometry on the local gfx1201 GPU using
// a lab-originated HIP marker kernel. It validates:
//   * 9-module / 96-function handle lifecycle reconstructed from CSV evidence;
//   * function ownership and every frame slot's captured launch metadata;
//   * ordered HIP submission with the original grid/block dimensions;
//   * parameter-block transport via a deterministic hash.
//
// This deliberately DOES NOT execute NVIDIA SASS or neural math and therefore
// does not count as S6. It is a transport/scheduler prerequisite only.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

static bool g_failed = false;

#define HIP_REQUIRE(cmd) do {                                                  \
    hipError_t e_ = (cmd);                                                     \
    if (e_ != hipSuccess) {                                                    \
        printf("HIP ERROR %s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #cmd,  \
               (int)e_, hipGetErrorString(e_));                               \
        g_failed = true;                                                       \
    }                                                                          \
} while (0)

static uint64_t Fnv1a(const void* ptr, size_t n) {
    const auto* p = static_cast<const uint8_t*>(ptr);
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t Fnv1a(const std::string& s) { return Fnv1a(s.data(), s.size()); }

static std::string JsonEscape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { o += '\\'; o += (char)c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c >= 0x20) o += (char)c;
    }
    return o;
}

static std::vector<std::string> ParseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                cur += '"'; ++i;
            } else if (c == '"') quoted = false;
            else cur += c;
        } else if (c == '"') quoted = true;
        else if (c == ',') { fields.push_back(cur); cur.clear(); }
        else cur += c;
    }
    fields.push_back(cur);
    return fields;
}

struct CsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    std::unordered_map<std::string, size_t> col;
};

static bool LoadCsv(const fs::path& path, CsvTable& t) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("ERROR: cannot open %s\n", path.string().c_str()); return false; }
    std::string line;
    if (!std::getline(f, line)) return false;
    if (line.size() >= 3 && (uint8_t)line[0] == 0xef && (uint8_t)line[1] == 0xbb &&
        (uint8_t)line[2] == 0xbf) line.erase(0, 3);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    t.header = ParseCsvLine(line);
    for (size_t i = 0; i < t.header.size(); ++i) t.col[t.header[i]] = i;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto r = ParseCsvLine(line);
        if (r.size() != t.header.size()) {
            printf("ERROR: malformed CSV row in %s\n", path.string().c_str());
            return false;
        }
        t.rows.push_back(std::move(r));
    }
    return true;
}

static const std::string& Cell(const CsvTable& t, const std::vector<std::string>& r,
                               const char* name) {
    static const std::string empty;
    auto it = t.col.find(name);
    return it == t.col.end() ? empty : r[it->second];
}

static uint32_t U32(const std::string& s) { return (uint32_t)std::stoul(s, nullptr, 0); }
static uint64_t U64Hex(const std::string& s) { return std::stoull(s, nullptr, 16); }

static std::vector<uint8_t> DecodeHex(const std::string& s) {
    std::vector<uint8_t> out;
    if (s.size() & 1) return out;
    out.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2)
        out.push_back((uint8_t)std::stoul(s.substr(i, 2), nullptr, 16));
    return out;
}

struct ModuleRec {
    uint64_t handle = 0;
    uint64_t dllOffset = 0;
    uint64_t capturedFnv = 0;
    uint32_t size = 0;
    uint32_t expectedFunctions = 0;
    uint32_t liveFunctions = 0;
    bool alive = true;
};
struct FunctionRec {
    uint64_t handle = 0;
    std::string name;
    uint64_t moduleOffset = 0;
    bool alive = true;
};
struct SlotRec {
    uint32_t slot = 0;
    std::string functionName;
    uint64_t moduleOffset = 0;
    dim3 grid{1,1,1};
    dim3 block{1,1,1};
    uint32_t dynamicShared = 0;
    uint32_t paramSize = 0;
    bool stable = false;
    uint32_t paramOffset = 0;
    std::vector<uint8_t> params;
    uint64_t functionHash = 0;
    uint64_t paramHash = 0;
};

struct Marker {
    uint32_t magic;
    uint32_t slot;
    uint32_t sequence;
    uint32_t reserved;
    uint64_t functionHash;
    uint64_t paramHash;
    uint32_t grid[3];
    uint32_t block[3];
};

__global__ static void MarkerKernel(Marker* markers, uint32_t* sequence,
                                    uint32_t slot, uint64_t functionHash,
                                    const uint8_t* parameterBytes,
                                    uint32_t paramOffset, uint32_t paramSize) {
    if (blockIdx.x || blockIdx.y || blockIdx.z ||
        threadIdx.x || threadIdx.y || threadIdx.z) return;
    Marker& m = markers[slot];
    m.magic = 0xA6D60001u;
    m.slot = slot;
    m.sequence = atomicAdd(sequence, 1u);
    m.functionHash = functionHash;
    uint64_t paramHash = 14695981039346656037ull;
    for (uint32_t i = 0; i < paramSize; ++i) {
        paramHash ^= parameterBytes[paramOffset + i];
        paramHash *= 1099511628211ull;
    }
    m.paramHash = paramHash;
    m.grid[0] = gridDim.x; m.grid[1] = gridDim.y; m.grid[2] = gridDim.z;
    m.block[0] = blockDim.x; m.block[1] = blockDim.y; m.block[2] = blockDim.z;
}

int main(int argc, char** argv) {
    fs::path graphDir;
    fs::path jsonPath;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--graph-dir") && i + 1 < argc) graphDir = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) jsonPath = argv[++i];
    }
    if (graphDir.empty()) {
        printf("usage: amd_graph_replay --graph-dir <R-21 result dir> --json <out.json>\n");
        return 2;
    }

    printf("=== AMD Feature-18 graph transport replay (NOT neural execution) ===\n");
    printf("graph dir: %s\n", graphDir.string().c_str());

    CsvTable mt, ft, gt;
    if (!LoadCsv(graphDir / "module_map.csv", mt) ||
        !LoadCsv(graphDir / "function_map.csv", ft) ||
        !LoadCsv(graphDir / "frame_001_sequence.csv", gt)) return 3;

    // Hardware identity: require a real AMD DXGI adapter and a gfx12 HIP target.
    uint32_t dxgiVendor = 0, dxgiDevice = 0;
    ComPtr<IDXGIFactory6> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> a;
            if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.VendorId == 0x1002) {
                dxgiVendor = d.VendorId; dxgiDevice = d.DeviceId; break;
            }
        }
    }

    int deviceCount = 0;
    HIP_REQUIRE(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 1) { printf("ERROR: no HIP device\n"); return 4; }
    hipDeviceProp_t prop{};
    HIP_REQUIRE(hipGetDeviceProperties(&prop, 0));
    HIP_REQUIRE(hipSetDevice(0));
    printf("HIP device: %s arch=%s; DXGI vendor=0x%04x device=0x%04x\n",
           prop.name, prop.gcnArchName, dxgiVendor, dxgiDevice);
    bool amdProof = dxgiVendor == 0x1002 && !strncmp(prop.gcnArchName, "gfx12", 5);

    // Reconstruct module handles from the dynamic/static join, without touching
    // proprietary blob bytes or attempting to load sm_120 SASS.
    std::unordered_map<uint64_t, ModuleRec> modules;
    uint64_t nextHandle = 0xA6D6000000000001ull;
    for (const auto& r : mt.rows) {
        ModuleRec m;
        m.handle = nextHandle++;
        m.dllOffset = U64Hex(Cell(mt, r, "dll_offset"));
        m.size = U32(Cell(mt, r, "size"));
        m.capturedFnv = U64Hex(Cell(mt, r, "fnv1a64"));
        m.expectedFunctions = U32(Cell(mt, r, "function_count"));
        modules.emplace(m.dllOffset, m);
    }

    std::unordered_map<std::string, FunctionRec> functions;
    for (const auto& r : ft.rows) {
        FunctionRec fn;
        fn.handle = nextHandle++;
        fn.name = Cell(ft, r, "function_name");
        fn.moduleOffset = U64Hex(Cell(ft, r, "module_dll_offset"));
        auto mi = modules.find(fn.moduleOffset);
        if (mi == modules.end() || functions.count(fn.name)) {
            printf("ERROR: invalid/duplicate function %s\n", fn.name.c_str());
            g_failed = true; continue;
        }
        mi->second.liveFunctions++;
        functions.emplace(fn.name, std::move(fn));
    }
    for (const auto& [off, m] : modules)
        if (m.liveFunctions != m.expectedFunctions) g_failed = true;

    std::vector<SlotRec> slots;
    std::vector<uint8_t> flatParameters;
    slots.reserve(gt.rows.size());
    for (const auto& r : gt.rows) {
        SlotRec s;
        s.slot = U32(Cell(gt, r, "slot"));
        s.functionName = Cell(gt, r, "function_name");
        s.moduleOffset = U64Hex(Cell(gt, r, "module_dll_offset"));
        s.grid = dim3(U32(Cell(gt,r,"grid_x")), U32(Cell(gt,r,"grid_y")), U32(Cell(gt,r,"grid_z")));
        s.block = dim3(U32(Cell(gt,r,"block_x")), U32(Cell(gt,r,"block_y")), U32(Cell(gt,r,"block_z")));
        s.dynamicShared = U32(Cell(gt, r, "dynamic_shared"));
        s.paramSize = U32(Cell(gt, r, "param_size"));
        s.stable = Cell(gt, r, "parameters_stable_across_5_frames") == "True";
        s.params = DecodeHex(Cell(gt, r, "frame1_param_hex"));
        s.paramOffset = (uint32_t)flatParameters.size();
        auto fi = functions.find(s.functionName);
        if (s.slot != slots.size() || fi == functions.end() ||
            fi->second.moduleOffset != s.moduleOffset || s.params.size() != s.paramSize ||
            s.block.x * s.block.y * s.block.z > 1024) {
            printf("ERROR: invalid graph slot %u (%s)\n", s.slot, s.functionName.c_str());
            g_failed = true;
        }
        s.functionHash = Fnv1a(s.functionName);
        s.paramHash = Fnv1a(s.params.data(), s.params.size());
        flatParameters.insert(flatParameters.end(), s.params.begin(), s.params.end());
        slots.push_back(std::move(s));
    }
    if (modules.size() != 9 || functions.size() != 96 || slots.size() != 156) {
        printf("ERROR: expected 9 modules / 96 functions / 156 slots\n");
        g_failed = true;
    }

    Marker* dMarkers = nullptr;
    uint32_t* dSequence = nullptr;
    uint8_t* dParameters = nullptr;
    HIP_REQUIRE(hipMalloc(&dMarkers, slots.size() * sizeof(Marker)));
    HIP_REQUIRE(hipMalloc(&dSequence, sizeof(uint32_t)));
    HIP_REQUIRE(hipMalloc(&dParameters, flatParameters.size()));
    HIP_REQUIRE(hipMemset(dMarkers, 0, slots.size() * sizeof(Marker)));
    HIP_REQUIRE(hipMemset(dSequence, 0, sizeof(uint32_t)));
    HIP_REQUIRE(hipMemcpy(dParameters, flatParameters.data(), flatParameters.size(),
                          hipMemcpyHostToDevice));

    hipEvent_t begin{}, end{};
    HIP_REQUIRE(hipEventCreate(&begin)); HIP_REQUIRE(hipEventCreate(&end));
    HIP_REQUIRE(hipEventRecord(begin));
    for (const auto& s : slots) {
        hipLaunchKernelGGL(MarkerKernel, s.grid, s.block, s.dynamicShared, 0,
                           dMarkers, dSequence, s.slot, s.functionHash,
                           dParameters, s.paramOffset, s.paramSize);
        HIP_REQUIRE(hipGetLastError());
    }
    HIP_REQUIRE(hipEventRecord(end));
    HIP_REQUIRE(hipEventSynchronize(end));
    float elapsedMs = -1.0f;
    HIP_REQUIRE(hipEventElapsedTime(&elapsedMs, begin, end));

    std::vector<Marker> got(slots.size());
    HIP_REQUIRE(hipMemcpy(got.data(), dMarkers, got.size() * sizeof(Marker), hipMemcpyDeviceToHost));
    uint32_t sequenceCount = 0;
    HIP_REQUIRE(hipMemcpy(&sequenceCount, dSequence, sizeof(sequenceCount), hipMemcpyDeviceToHost));

    uint32_t mismatches = 0;
    for (const auto& s : slots) {
        const Marker& m = got[s.slot];
        bool ok = m.magic == 0xA6D60001u && m.slot == s.slot && m.sequence == s.slot &&
                  m.functionHash == s.functionHash && m.paramHash == s.paramHash &&
                  m.grid[0] == s.grid.x && m.grid[1] == s.grid.y && m.grid[2] == s.grid.z &&
                  m.block[0] == s.block.x && m.block[1] == s.block.y && m.block[2] == s.block.z;
        if (!ok) ++mismatches;
    }
    if (sequenceCount != slots.size()) mismatches++;

    HIP_REQUIRE(hipEventDestroy(begin)); HIP_REQUIRE(hipEventDestroy(end));
    HIP_REQUIRE(hipFree(dMarkers)); HIP_REQUIRE(hipFree(dSequence));
    HIP_REQUIRE(hipFree(dParameters));

    // Destruction contract: functions first, then modules; prove no leaked handles.
    uint32_t destroyedFunctions = 0, destroyedModules = 0;
    for (auto& [name, fn] : functions) {
        if (fn.alive) { fn.alive = false; modules[fn.moduleOffset].liveFunctions--; destroyedFunctions++; }
    }
    for (auto& [off, m] : modules) {
        if (m.alive && m.liveFunctions == 0) { m.alive = false; destroyedModules++; }
    }
    bool lifecyclePass = destroyedFunctions == 96 && destroyedModules == 9;
    bool pass = !g_failed && amdProof && mismatches == 0 && lifecyclePass;
    printf("[%s] modules=9 functions=96 slots=156 gpu_markers=%u mismatches=%u time=%.3f ms\n",
           pass ? "PASS" : "FAIL", sequenceCount, mismatches, elapsedMs);
    printf("classification: LAB_TRANSPORT_ONLY; counts_as_S6=false; neural_math=false\n");

    if (!jsonPath.empty()) {
        FILE* jf = nullptr;
        _wfopen_s(&jf, jsonPath.wstring().c_str(), L"wb");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"schema\": 1,\n");
            fprintf(jf, "  \"experiment\": \"amd_graph_replay\",\n");
            fprintf(jf, "  \"status\": \"%s\",\n", pass ? "PASS" : "FAIL");
            fprintf(jf, "  \"classification\": \"LAB_TRANSPORT_ONLY\",\n");
            fprintf(jf, "  \"counts_as_s6\": false,\n  \"neural_math_executed\": false,\n");
            fprintf(jf, "  \"hip_device\": \"%s\",\n", JsonEscape(prop.name).c_str());
            fprintf(jf, "  \"hip_arch\": \"%s\",\n", JsonEscape(prop.gcnArchName).c_str());
            fprintf(jf, "  \"dxgi_vendor\": \"0x%04x\",\n  \"dxgi_device\": \"0x%04x\",\n", dxgiVendor, dxgiDevice);
            fprintf(jf, "  \"modules_created\": %zu,\n  \"functions_created\": %zu,\n", modules.size(), functions.size());
            fprintf(jf, "  \"graph_slots\": %zu,\n  \"gpu_markers\": %u,\n", slots.size(), sequenceCount);
            fprintf(jf, "  \"parameter_bytes_uploaded\": %zu,\n", flatParameters.size());
            fprintf(jf, "  \"parameter_hash_location\": \"GPU\",\n");
            fprintf(jf, "  \"validation_mismatches\": %u,\n", mismatches);
            fprintf(jf, "  \"functions_destroyed\": %u,\n  \"modules_destroyed\": %u,\n", destroyedFunctions, destroyedModules);
            fprintf(jf, "  \"kernel_time_ms\": %.6f,\n", elapsedMs);
            fprintf(jf, "  \"source_graph\": \"%s\"\n", JsonEscape(graphDir.string()).c_str());
            fprintf(jf, "}\n");
            fclose(jf);
        } else pass = false;
    }
    return pass ? 0 : 1;
}
