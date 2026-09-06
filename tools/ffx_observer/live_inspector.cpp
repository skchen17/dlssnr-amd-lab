#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "live_inspector.h"
#include <d3d12.h>
#include <wrl/client.h>
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <io.h>
#include <fcntl.h>
#include <stdexcept>

using Microsoft::WRL::ComPtr;
namespace {
std::mutex setup,output,tracking,queueSetup;
std::atomic<PfnFfxDispatch> originalDispatch{nullptr};
using Execute_t=void (STDMETHODCALLTYPE*)(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*);
std::atomic<Execute_t> originalExecute{nullptr};
FILE* file=nullptr;uint32_t sampleLimit=0;std::atomic<uint64_t> sequence{0},unmatchedQueues{0};
bool initialized=false,queueAttempted=false;
struct Pending {uint64_t sequence;ComPtr<IUnknown> identity;};
std::vector<Pending> pending;
template<class T> bool Copy(const void* p,T& t){SIZE_T n=0;return p&&ReadProcessMemory(GetCurrentProcess(),p,&t,sizeof(t),&n)&&n==sizeof(t);}
void Emit(const std::string& s){std::lock_guard<std::mutex> l(output);if(file){std::fprintf(file,"%s\n",s.c_str());std::fflush(file);}}
std::string Ptr(const void* p){char text[32];std::snprintf(text,sizeof(text),"\"0x%llx\"",(unsigned long long)(uintptr_t)p);return text;}
bool Patch(void* volatile* slot,void* expected,void* target) {
    DWORD old=0,ignored=0;if(!VirtualProtect(const_cast<void**>(slot),sizeof(void*),PAGE_READWRITE,&old))return false;
    void* previous=InterlockedCompareExchangePointer(slot,target,expected);
    const bool restored=VirtualProtect(const_cast<void**>(slot),sizeof(void*),old,&ignored)!=0;
    // If protection restoration fails after publication, retain the hook and
    // originals for process lifetime. Do not tear out code used by live calls.
    if(previous==expected&&!restored)Emit("{\"event\":\"protection_restore_failed\"}");
    return previous==expected;
}
void STDMETHODCALLTYPE Queue(ID3D12CommandQueue* q,UINT count,ID3D12CommandList* const* lists) {
    const DWORD incoming=GetLastError();
    try {
        std::vector<uint64_t> matched;std::ostringstream ptrs;bool first=true;
        if(lists&&count<=64) {
            std::lock_guard<std::mutex> lock(tracking);
            for(UINT i=0;i<count;++i) {
                ComPtr<IUnknown> id;if(!lists[i]||FAILED(lists[i]->QueryInterface(IID_PPV_ARGS(&id))))continue;
                if(!first)ptrs<<',';first=false;ptrs<<Ptr(lists[i]);
                for(auto it=pending.begin();it!=pending.end();) {
                    if(it->identity.Get()==id.Get()){matched.push_back(it->sequence);it=pending.erase(it);}else ++it;
                }
            }
        }
        if(!matched.empty()||unmatchedQueues.fetch_add(1)<32) {
            std::ostringstream s;s<<"{\"event\":\"execute\",\"thread\":"<<GetCurrentThreadId()<<",\"queue\":"<<Ptr(q)<<",\"queue_type\":"<<q->GetDesc().Type
              <<",\"list_count\":"<<count<<",\"lists\":["<<ptrs.str()<<"],\"sample_ids\":[";
            for(size_t i=0;i<matched.size();++i){if(i)s<<',';s<<matched[i];}s<<"],\"original_forwarded_once\":true}";Emit(s.str());
        }
    }catch(...){/* Diagnostics must not suppress or duplicate submission. */}
    SetLastError(incoming);originalExecute.load(std::memory_order_acquire)(q,count,lists);
}
void QueueHook(ID3D12Device* d) {
    std::lock_guard<std::mutex> lock(queueSetup);if(queueAttempted)return;queueAttempted=true;
    ComPtr<ID3D12CommandQueue> probe;D3D12_COMMAND_QUEUE_DESC desc{};
    if(FAILED(d->CreateCommandQueue(&desc,IID_PPV_ARGS(&probe)))){Emit("{\"event\":\"queue_hook\",\"installed\":false}");return;}
    auto table=*reinterpret_cast<void***>(probe.Get());void* expected=table[10];
    originalExecute.store(reinterpret_cast<Execute_t>(expected),std::memory_order_release);
    bool ok=Patch(reinterpret_cast<void* volatile*>(&table[10]),expected,reinterpret_cast<void*>(Queue));
    Emit(std::string("{\"event\":\"queue_hook\",\"installed\":")+(ok?"true":"false")+",\"scope\":\"one observed direct-queue vtable; no completeness claim\"}");
}
std::string Inspect(uint64_t id,const ffxDispatchDescUpscale& a) {
    auto* list=static_cast<ID3D12GraphicsCommandList*>(a.commandList);
    if(!list)throw std::runtime_error("null list");
    ComPtr<ID3D12Device> device;ComPtr<IUnknown> identity,deviceIdentity;
    if(FAILED(list->GetDevice(IID_PPV_ARGS(&device)))||FAILED(list->QueryInterface(IID_PPV_ARGS(&identity)))||FAILED(device.As(&deviceIdentity)))throw std::runtime_error("list/device query");
    QueueHook(device.Get());
    std::ostringstream s;s<<"{\"event\":\"resources\",\"sample_id\":"<<id<<",\"thread\":"<<GetCurrentThreadId()<<",\"list\":"<<Ptr(list)
       <<",\"list_identity\":"<<Ptr(identity.Get())<<",\"list_type\":"<<list->GetType()<<",\"device\":"<<Ptr(device.Get())
       <<",\"adapter_luid_low\":"<<device->GetAdapterLuid().LowPart<<",\"adapter_luid_high\":"<<device->GetAdapterLuid().HighPart
       <<",\"render_size\":["<<a.renderSize.width<<','<<a.renderSize.height<<"],\"upscale_size\":["<<a.upscaleSize.width<<','<<a.upscaleSize.height
       <<"],\"motion_scale\":["<<a.motionVectorScale.x<<','<<a.motionVectorScale.y<<"],\"jitter\":["<<a.jitterOffset.x<<','<<a.jitterOffset.y
       <<"],\"reset\":"<<(a.reset?"true":"false")<<",\"camera_near\":"<<a.cameraNear<<",\"camera_far\":"<<a.cameraFar<<",\"resources\":[";
    const char* roles[]={"color","depth","motion","exposure","reactive","transparency","output"};
    const FfxApiResource* entries[]={&a.color,&a.depth,&a.motionVectors,&a.exposure,&a.reactive,&a.transparencyAndComposition,&a.output};
    uint64_t total=0;bool all=true;
    for(unsigned i=0;i<7;++i) {
        if(i)s<<',';const auto& r=*entries[i];s<<"{\"role\":\""<<roles[i]<<"\",\"pointer\":"<<Ptr(r.resource);
        if(!r.resource){s<<",\"present\":false}";continue;}
        ComPtr<ID3D12Resource> resource;
        if(FAILED(static_cast<IUnknown*>(r.resource)->QueryInterface(IID_PPV_ARGS(&resource))))throw std::runtime_error("resource interface query");
        auto desc=resource->GetDesc();ComPtr<ID3D12Device> owner;ComPtr<IUnknown> ownerIdentity;
        if(FAILED(resource->GetDevice(IID_PPV_ARGS(&owner)))||FAILED(owner.As(&ownerIdentity)))throw std::runtime_error("resource device query");
        const bool same=ownerIdentity.Get()==deviceIdentity.Get();auto expected=ffxApiGetResourceDX12(resource.Get(),r.state);
        bool match=!std::memcmp(&expected.description,&r.description,sizeof(r.description));all&=same&&match;
        UINT rows=0;UINT64 rowBytes=0,bytes=0;D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        device->GetCopyableFootprints(&desc,0,1,0,&fp,&rows,&rowBytes,&bytes);total+=bytes;
        s<<",\"present\":true,\"width\":"<<desc.Width<<",\"height\":"<<desc.Height<<",\"dxgi_format\":"<<desc.Format
          <<",\"dimension\":"<<desc.Dimension<<",\"array_size\":"<<desc.DepthOrArraySize<<",\"mips\":"<<desc.MipLevels<<",\"samples\":"<<desc.SampleDesc.Count
          <<",\"d3d12_flags\":"<<desc.Flags<<",\"ffx_format\":"<<r.description.format<<",\"ffx_usage\":"<<r.description.usage
          <<",\"declared_ffx_state\":"<<r.state<<",\"same_device\":"<<(same?"true":"false")<<",\"descriptor_matches\":"<<(match?"true":"false")
          <<",\"row_pitch\":"<<fp.Footprint.RowPitch<<",\"row_bytes\":"<<rowBytes<<",\"footprint_bytes\":"<<bytes<<'}';
    }
    s<<"],\"total_footprint_bytes\":"<<total<<",\"all_descriptors_match\":"<<(all?"true":"false")<<",\"resource_states_verified\":false,\"gpu_copies\":0}";
    {std::lock_guard<std::mutex> lock(tracking);pending.push_back({id,identity});}
    return s.str();
}
ffxReturnCode_t Dispatch(ffxContext* c,const ffxDispatchDescHeader* descriptor) {
    DWORD incoming=GetLastError();const uint64_t id=sequence.fetch_add(1)+1;const bool sample=id<=sampleLimit;
    if(sample)try {
        ffxDispatchDescUpscale a{};
        if(!Copy(descriptor,a)||a.header.type!=FFX_API_DISPATCH_DESC_TYPE_UPSCALE||a.header.pNext)throw std::runtime_error("unverified dispatch body");
        Emit(Inspect(id,a));
    }catch(...){try{Emit("{\"event\":\"inspection_failed\"}");}catch(...){}}
    SetLastError(incoming);const auto status=originalDispatch.load(std::memory_order_acquire)(c,descriptor);DWORD outgoing=GetLastError();
    if(sample)try{Emit("{\"event\":\"dispatch_return\",\"sample_id\":"+std::to_string(id)+",\"status\":"+std::to_string(status)+"}");}catch(...){}
    SetLastError(outgoing);return status;
}
bool Initialize(const FfxLiveConfigV1& c,PfnFfxDispatch original) {
    if(initialized||c.size!=sizeof(c)||c.version!=1||c.sample_limit<1||c.sample_limit>32||c.log_path[1023]||!original||!std::filesystem::path(c.log_path).is_absolute())return false;
    HANDLE h=CreateFileW(c.log_path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)return false;
    int fd=_open_osfhandle(reinterpret_cast<intptr_t>(h),_O_WRONLY|_O_BINARY);
    if(fd<0){CloseHandle(h);return false;}file=_fdopen(fd,"wb");if(!file){_close(fd);return false;}
    HMODULE self=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,reinterpret_cast<LPCWSTR>(Dispatch),&self)){std::fclose(file);file=nullptr;return false;}
    sampleLimit=c.sample_limit;pending.reserve(sampleLimit);originalDispatch.store(original,std::memory_order_release);initialized=true;return true;
}
void* volatile* DispatchSlot() {
    auto base=reinterpret_cast<BYTE*>(GetModuleHandleW(nullptr));IMAGE_DOS_HEADER dos{};IMAGE_NT_HEADERS64 nt{};
    if(!Copy(base,dos)||dos.e_magic!=IMAGE_DOS_SIGNATURE||dos.e_lfanew<=0||!Copy(base+dos.e_lfanew,nt)||nt.Signature!=IMAGE_NT_SIGNATURE||nt.OptionalHeader.Magic!=IMAGE_NT_OPTIONAL_HDR64_MAGIC)return nullptr;
    auto dir=nt.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];const auto size=nt.OptionalHeader.SizeOfImage;
    if(!dir.VirtualAddress||dir.VirtualAddress>=size||dir.Size>size-dir.VirtualAddress)return nullptr;
    void* volatile* result=nullptr;
    for(size_t i=0;i<dir.Size/sizeof(IMAGE_IMPORT_DESCRIPTOR);++i) {
        IMAGE_IMPORT_DESCRIPTOR imp{};if(!Copy(base+dir.VirtualAddress+i*sizeof(imp),imp))return nullptr;if(!imp.Name)break;if(!imp.OriginalFirstThunk)continue;
        for(size_t j=0;j<8192;++j) {
            uint64_t name=uint64_t(imp.OriginalFirstThunk)+j*8,value=uint64_t(imp.FirstThunk)+j*8;
            if(name+8>size||value+8>size)return nullptr;
            IMAGE_THUNK_DATA64 t{};if(!Copy(base+name,t))return nullptr;if(!t.u1.AddressOfData)break;if(t.u1.Ordinal&IMAGE_ORDINAL_FLAG64)continue;
            if(t.u1.AddressOfData>size-64)return nullptr;char text[62]{};
            SIZE_T n=0;if(!ReadProcessMemory(GetCurrentProcess(),base+t.u1.AddressOfData+2,text,sizeof(text)-1,&n)||n!=sizeof(text)-1)return nullptr;
            if(!std::strcmp(text,"ffxDispatch")){if(result)return nullptr;result=reinterpret_cast<void* volatile*>(base+value);}
        }
    }return result;
}
}
extern "C" __declspec(dllexport) DWORD WINAPI FfxLive_Attach(void* pointer) {
    std::lock_guard<std::mutex> lock(setup);
    try {
        FfxLiveConfigV1 c{};if(!Copy(pointer,c)||!c.expected_observer_base)return 0;
        auto slot=DispatchSlot();if(!slot)return 0;void* target=nullptr;if(!Copy(const_cast<void**>(slot),target))return 0;
        HMODULE owner=nullptr;if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(target),&owner)||
            reinterpret_cast<uintptr_t>(owner)!=c.expected_observer_base)return 0;
#ifdef FFX_CONTRACT_UPDATE
        if(!GetProcAddress(owner,"FfxLive_TestDispatch"))return 0;
#else
        if(!GetProcAddress(owner,"FfxObserver_GetStats"))return 0;
#endif
        if(!Initialize(c,reinterpret_cast<PfnFfxDispatch>(target)))return 0;
        bool ok=Patch(slot,target,reinterpret_cast<void*>(Dispatch));Emit(std::string("{\"event\":\"attach\",\"iat_chained_to_existing_observer\":")+(ok?"true":"false")+"}");return ok?1:0;
    }catch(...){return 0;}
}
// Isolated test only: original must be a direct ffxDispatch DLL export.
extern "C" __declspec(dllexport) DWORD WINAPI FfxLive_TestInitialize(void* pointer) {
    std::lock_guard<std::mutex> lock(setup);
    try {FfxLiveTestConfigV1 c{};if(!Copy(pointer,c)||!c.original_dispatch)return 0;
        HMODULE owner=nullptr;auto p=reinterpret_cast<PfnFfxDispatch>(c.original_dispatch);
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(p),&owner)||GetProcAddress(owner,"ffxDispatch")!=reinterpret_cast<FARPROC>(p))return 0;
        return Initialize(c.common,p)?1:0;
    }catch(...){return 0;}
}
extern "C" __declspec(dllexport) ffxReturnCode_t FfxLive_TestDispatch(ffxContext* c,const ffxDispatchDescHeader* d){return Dispatch(c,d);}
