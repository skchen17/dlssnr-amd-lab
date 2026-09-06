#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "ffx_observer.h"
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <share.h>

static_assert(sizeof(void*) == 8 && sizeof(ffxApiHeader) == 16);
static_assert(sizeof(FfxApiResource) == 48 && sizeof(FfxApiResourceDescription) == 32);
static_assert(sizeof(ffxCreateContextDescUpscale) == 48);
static_assert(sizeof(ffxDispatchDescUpscale) == 432);
static_assert(offsetof(ffxDispatchDescUpscale, color) == 24);
static_assert(offsetof(ffxDispatchDescUpscale, output) == 312);
static_assert(offsetof(ffxDispatchDescUpscale, renderSize) == 376);
static_assert(offsetof(ffxDispatchDescUpscale, reset) == 408);

namespace {
std::mutex control, logging;
FILE* logFile = nullptr;
uint32_t profile = 0, limit = 0;
uint64_t emitted = 0, dropped = 0;
std::atomic<uint64_t> calls{0}, active{0};
bool attached = false;
HMODULE retainedBackend = nullptr;
struct Slot { void* volatile* address; void* original; void* hook; };
std::vector<Slot> slots;
std::array<void*, 5> original{};

bool Copy(const void* from, void* to, size_t size) {
    SIZE_T copied = 0;
    return from && ReadProcessMemory(GetCurrentProcess(), from, to, size, &copied) && copied == size;
}
template<class T> bool Copy(const void* from, T& to) { return Copy(from, &to, sizeof(to)); }
std::string Pointer(const void* value) {
    char text[32]; std::snprintf(text, sizeof(text), "\"0x%llx\"", (unsigned long long)(uintptr_t)value); return text;
}
std::string Float(float value) {
    if (!std::isfinite(value)) return "null";
    char text[48]; std::snprintf(text, sizeof(text), "%.9g", double(value)); return text;
}
void Emit(const std::string& text) {
    std::lock_guard<std::mutex> guard(logging);
    if (!logFile) return;
    if (emitted >= limit) { ++dropped; return; }
    std::fprintf(logFile, "%s\n", text.c_str()); std::fflush(logFile); ++emitted;
}
std::string Header(const ffxApiHeader* pointer) {
    std::ostringstream out;
    ffxApiHeader h{};
    if (!Copy(pointer, h)) return "\"header_readable\":false";
    out << "\"header_readable\":true,\"descriptor_type\":" << h.type << ",\"extension_types\":[";
    std::array<const void*, 9> visited{}; visited[0] = pointer;
    const ffxApiHeader* next = h.pNext;
    const char* state = "complete"; unsigned count = 0;
    while (next) {
        bool cycle = false;
        for (unsigned i = 0; i <= count; ++i) cycle |= visited[i] == next;
        if (cycle) { state = "cycle"; break; }
        if (count == 8) { state = "limit"; break; }
        if (!Copy(next, h)) { state = "unreadable"; break; }
        visited[++count] = next;
        if (count > 1) out << ',';
        out << h.type; next = h.pNext;
    }
    out << "],\"extension_chain\":\"" << state << "\"";
    return out.str();
}
void Resource(std::ostringstream& out, const char* name, const FfxApiResource& r) {
    out << '"' << name << "\":{\"pointer\":" << Pointer(r.resource)
        << ",\"ffx_state\":" << r.state << ",\"type\":" << r.description.type
        << ",\"format\":" << r.description.format << ",\"width\":" << r.description.width
        << ",\"height\":" << r.description.height << ",\"depth\":" << r.description.depth
        << ",\"mips\":" << r.description.mipCount << ",\"flags\":" << r.description.flags
        << ",\"usage\":" << r.description.usage << '}';
}
std::string Body(const ffxApiHeader* pointer, bool create) {
    if (profile != 1) return "\"body_decoded\":false,\"body_reason\":\"header_only_profile\"";
    ffxApiHeader header{};
    if (!Copy(pointer, header)) return "\"body_decoded\":false,\"body_reason\":\"unreadable_header\"";
    std::ostringstream out;
    if (create && header.type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
        ffxCreateContextDescUpscale d{};
        if (!Copy(pointer, d)) return "\"body_decoded\":false,\"body_reason\":\"short_create\"";
        out << "\"body_decoded\":true,\"create_flags\":" << d.flags << ",\"max_render\":["
            << d.maxRenderSize.width << ',' << d.maxRenderSize.height << "],\"max_upscale\":["
            << d.maxUpscaleSize.width << ',' << d.maxUpscaleSize.height << ']';
    } else if (!create && header.type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
        ffxDispatchDescUpscale d{};
        if (!Copy(pointer, d)) return "\"body_decoded\":false,\"body_reason\":\"short_dispatch\"";
        // Read bool representations as bytes to avoid interpreting invalid bools.
        const auto reset = reinterpret_cast<const unsigned char*>(&d.reset)[0];
        const auto sharpen = reinterpret_cast<const unsigned char*>(&d.enableSharpening)[0];
        out << "\"body_decoded\":true,\"command_list\":" << Pointer(d.commandList) << ",\"resources\":{";
        Resource(out,"color",d.color); out << ','; Resource(out,"depth",d.depth); out << ',';
        Resource(out,"motion",d.motionVectors); out << ','; Resource(out,"exposure",d.exposure); out << ',';
        Resource(out,"reactive",d.reactive); out << ','; Resource(out,"transparency",d.transparencyAndComposition); out << ',';
        Resource(out,"output",d.output);
        out << "},\"render_size\":[" << d.renderSize.width << ',' << d.renderSize.height
            << "],\"upscale_size\":[" << d.upscaleSize.width << ',' << d.upscaleSize.height
            << "],\"jitter\":[" << Float(d.jitterOffset.x) << ',' << Float(d.jitterOffset.y)
            << "],\"motion_scale\":[" << Float(d.motionVectorScale.x) << ',' << Float(d.motionVectorScale.y)
            << "],\"reset\":" << unsigned(reset) << ",\"enable_sharpening\":" << unsigned(sharpen)
            << ",\"bool_encoding_valid\":" << (reset <= 1 && sharpen <= 1 ? "true" : "false")
            << ",\"sharpness\":" << Float(d.sharpness) << ",\"frame_time_ms\":" << Float(d.frameTimeDelta)
            << ",\"pre_exposure\":" << Float(d.preExposure) << ",\"camera_near\":" << Float(d.cameraNear)
            << ",\"camera_far\":" << Float(d.cameraFar) << ",\"camera_fov_y\":" << Float(d.cameraFovAngleVertical)
            << ",\"view_to_meters\":" << Float(d.viewSpaceToMetersFactor) << ",\"dispatch_flags\":" << d.flags;
    } else return "\"body_decoded\":false,\"body_reason\":\"unknown_descriptor_type\"";
    return out.str();
}
struct ActiveCall { ActiveCall() { ++active; ++calls; } ~ActiveCall() { --active; } };
template<class Invoke> ffxReturnCode_t Observe(const char* name, ffxContext* context,
        const ffxApiHeader* desc, bool hasBody, bool create, Invoke invoke) {
    const DWORD incomingError = GetLastError();
    ActiveCall count;
    bool skipObservation = false;
    { std::lock_guard<std::mutex> guard(logging);
      if (emitted >= limit) { ++dropped; skipObservation = true; } }
    if (skipObservation) { SetLastError(incomingError); return invoke(); }
    ffxContext before = nullptr; Copy(context, before);
    // Observation exceptions must never prevent or duplicate the real call.
    std::string metadata;
    try { metadata = Header(desc); if (hasBody) metadata += ',' + Body(desc, create); }
    catch (...) { metadata = "\"observation_error\":true"; }
    SetLastError(incomingError);
    const auto status = invoke();
    const DWORD outgoingError = GetLastError();
    try {
        ffxContext after = nullptr; Copy(context, after);
        std::ostringstream out;
        out << "{\"event\":\"" << name << "\",\"thread_id\":" << GetCurrentThreadId()
            << ",\"profile\":" << profile << ",\"context_before\":" << Pointer(before)
            << ",\"context_after\":" << Pointer(after) << ",\"return_code\":" << status << ',' << metadata << '}';
        Emit(out.str());
    } catch (...) { /* Rendering must not depend on diagnostic allocation/IO. */ }
    SetLastError(outgoingError);
    return status;
}
ffxReturnCode_t Create(ffxContext* c, ffxCreateContextDescHeader* d, const ffxAllocationCallbacks* a) {
    return Observe("ffxCreateContext", c, d, true, true, [&] { return reinterpret_cast<PfnFfxCreateContext>(original[0])(c,d,a); });
}
ffxReturnCode_t Destroy(ffxContext* c, const ffxAllocationCallbacks* a) {
    return Observe("ffxDestroyContext", c, nullptr, false, false, [&] { return reinterpret_cast<PfnFfxDestroyContext>(original[1])(c,a); });
}
ffxReturnCode_t Configure(ffxContext* c, const ffxConfigureDescHeader* d) {
    return Observe("ffxConfigure", c, d, false, false, [&] { return reinterpret_cast<PfnFfxConfigure>(original[2])(c,d); });
}
ffxReturnCode_t Query(ffxContext* c, ffxQueryDescHeader* d) {
    return Observe("ffxQuery", c, d, false, false, [&] { return reinterpret_cast<PfnFfxQuery>(original[3])(c,d); });
}
ffxReturnCode_t Dispatch(ffxContext* c, const ffxDispatchDescHeader* d) {
    return Observe("ffxDispatch", c, d, true, false, [&] { return reinterpret_cast<PfnFfxDispatch>(original[4])(c,d); });
}
const char* names[] = {"ffxCreateContext","ffxDestroyContext","ffxConfigure","ffxQuery","ffxDispatch"};
void* hooks[] = {reinterpret_cast<void*>(Create), reinterpret_cast<void*>(Destroy), reinterpret_cast<void*>(Configure),
                 reinterpret_cast<void*>(Query), reinterpret_cast<void*>(Dispatch)};
bool Exchange(const Slot& slot, bool restore) {
    DWORD protection = 0;
    if (!VirtualProtect(const_cast<void**>(slot.address), sizeof(void*), PAGE_READWRITE, &protection)) return false;
    void* expected = restore ? slot.hook : slot.original;
    void* value = restore ? slot.original : slot.hook;
    auto previous = InterlockedCompareExchangePointer(slot.address, value, expected);
    DWORD ignored;
    const bool protectedAgain = VirtualProtect(const_cast<void**>(slot.address), sizeof(void*), protection, &ignored) != 0;
    return previous == expected && protectedAgain;
}
} // namespace

extern "C" __declspec(dllexport) int __cdecl FfxObserver_Attach(const FfxObserverConfigV1* config) {
    std::lock_guard<std::mutex> guard(control);
    if (attached || !config || config->size != sizeof(*config) || config->profile > 1 ||
        !config->importing_module || !config->log_path || !config->event_limit || config->event_limit > 10000) return 0;
    try {
        if (!std::filesystem::path(config->log_path).is_absolute()) return 0;
        auto base = reinterpret_cast<BYTE*>(config->importing_module);
        IMAGE_DOS_HEADER dos{}; if (!Copy(base, dos) || dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < 0) return 0;
        IMAGE_NT_HEADERS64 nt{}; if (!Copy(base + dos.e_lfanew, nt) || nt.Signature != IMAGE_NT_SIGNATURE || nt.OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return 0;
        auto directory = nt.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        if (!directory.VirtualAddress || directory.VirtualAddress >= nt.OptionalHeader.SizeOfImage ||
            directory.Size > nt.OptionalHeader.SizeOfImage - directory.VirtualAddress) return 0;
        std::array<Slot,5> found{}; unsigned foundCount = 0;
        for (size_t i = 0; i < directory.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR); ++i) {
            IMAGE_IMPORT_DESCRIPTOR imp{};
            if (!Copy(base + directory.VirtualAddress + i*sizeof(imp), imp)) return 0;
            if (!imp.Name) break;
            if (!imp.OriginalFirstThunk) continue; // Never treat resolved addresses as name RVAs.
            for (size_t j = 0; j < 8192; ++j) {
                const uint64_t nameThunk = uint64_t(imp.OriginalFirstThunk) + j*sizeof(IMAGE_THUNK_DATA64);
                const uint64_t valueThunk = uint64_t(imp.FirstThunk) + j*sizeof(IMAGE_THUNK_DATA64);
                if (nameThunk + sizeof(IMAGE_THUNK_DATA64) > nt.OptionalHeader.SizeOfImage ||
                    valueThunk + sizeof(IMAGE_THUNK_DATA64) > nt.OptionalHeader.SizeOfImage) return 0;
                IMAGE_THUNK_DATA64 thunk{};
                if (!Copy(base + imp.OriginalFirstThunk + j*sizeof(thunk), thunk)) return 0;
                if (!thunk.u1.AddressOfData) break;
                if (thunk.u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;
                if (thunk.u1.AddressOfData > nt.OptionalHeader.SizeOfImage - 64) return 0;
                char name[64]{};
                if (!Copy(base + thunk.u1.AddressOfData + 2, name, sizeof(name)-1)) return 0;
                for (unsigned k = 0; k < 5; ++k) if (!strcmp(name,names[k])) {
                    if (found[k].address) return 0;
                    auto addr = reinterpret_cast<void* volatile*>(base + imp.FirstThunk + j*sizeof(thunk));
                    void* value = nullptr; if (!Copy(const_cast<void**>(addr), value) || !value) return 0;
                    found[k] = {addr,value,hooks[k]}; ++foundCount;
                }
            }
        }
        if (foundCount != 5) return 0;
        HMODULE backend = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(found[0].original), &backend)) return 0;
        // Every target must be the direct export of one backend, not a prior proxy hook.
        bool valid = true;
        for (unsigned k=0;k<5;++k) valid &= reinterpret_cast<void*>(GetProcAddress(backend,names[k])) == found[k].original;
        if (!valid) { FreeLibrary(backend); return 0; }
        // Create-new prevents accidentally overwriting a previous observation log.
        HANDLE file = CreateFileW(config->log_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) { FreeLibrary(backend); return 0; }
        CloseHandle(file);
        logFile=_wfsopen(config->log_path,L"ab",_SH_DENYWR);
        if (!logFile) { FreeLibrary(backend); return 0; }
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                               reinterpret_cast<LPCWSTR>(&FfxObserver_Attach), &self)) { std::fclose(logFile); logFile=nullptr; FreeLibrary(backend); return 0; }
        profile=config->profile; limit=config->event_limit; emitted=dropped=0; calls=0;
        retainedBackend=backend; slots.assign(found.begin(),found.end());
        for (unsigned k=0;k<5;++k) original[k]=found[k].original;
        unsigned patched=0;
        for (;patched<5;++patched) if (!Exchange(slots[patched],false)) break;
        if (patched != 5) {
            // Includes a potentially exchanged slot whose protection restore failed.
            for (unsigned k=0;k<=patched && k<5;++k) Exchange(slots[k],true);
            slots.clear(); std::fclose(logFile); logFile=nullptr; FreeLibrary(backend); retainedBackend=nullptr; return 0;
        }
        attached=true; return 1;
    } catch (...) {
        // Allocations can fail before publication. No hook is active until all
        // preparation allocations complete; the patch loop itself does not throw.
        if (!attached) {
            if (logFile) { std::fclose(logFile); logFile=nullptr; }
            if (retainedBackend) { FreeLibrary(retainedBackend); retainedBackend=nullptr; }
            slots.clear();
        }
        return 0;
    }
}
extern "C" __declspec(dllexport) int __cdecl FfxObserver_Detach() {
    std::lock_guard<std::mutex> guard(control);
    if (!attached || active.load()) return 0;
    bool success=true;
    for (const auto& slot:slots) success &= Exchange(slot,true);
    if (!success) return 0; // Do not overwrite foreign hooks or unload observer.
    slots.clear(); attached=false;
    { std::lock_guard<std::mutex> logGuard(logging); std::fclose(logFile); logFile=nullptr; }
    FreeLibrary(retainedBackend); retainedBackend=nullptr; return 1;
}
extern "C" __declspec(dllexport) int __cdecl FfxObserver_GetStats(FfxObserverStatsV1* stats) {
    if (!stats || stats->size != sizeof(*stats)) return 0;
    std::lock_guard<std::mutex> guard(control); std::lock_guard<std::mutex> logGuard(logging);
    *stats={sizeof(*stats),uint32_t(slots.size()),calls.load(),emitted,dropped,active.load()}; return 1;
}

// Called explicitly after LoadLibrary completes, never from DllMain. No GPU
// capture, thread creation, automatic loading or game-output modification.
extern "C" __declspec(dllexport) DWORD WINAPI FfxObserver_Bootstrap(void* pointer) {
    FfxObserverBootstrapV2 b{};
    if(!Copy(pointer,b)||b.size!=sizeof(b)||b.version!=2||b.profile>1||b.log_path[1023]!=0)return 0;
    FfxObserverConfigV1 c{sizeof(c),b.profile,GetModuleHandleW(nullptr),b.log_path,b.event_limit};
    return FfxObserver_Attach(&c)?1:0;
}
