#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "capture_session.h"
#include "texture_capture.h"
#include "output_boundary_batch.h"
#include "output_boundary_copy.h"
#include "output_boundary_roundtrip.h"
#include "output_boundary_patch.h"
#include "output_boundary_filter.h"
#include "output_boundary_network.h"
#include "output_static_preview.h"
#include "resident_surface.h"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <io.h>
#include <fcntl.h>
#include <fstream>

using Microsoft::WRL::ComPtr;
namespace {
using ResetFn=HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,ID3D12CommandAllocator*,ID3D12PipelineState*);
using CloseFn=HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);
using BarrierFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,UINT,const D3D12_RESOURCE_BARRIER*);
using ExecuteFn=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,UINT,ID3D12CommandList*const*);
using GpuDispatchFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,UINT,UINT,UINT);
using IndirectFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,ID3D12CommandSignature*,UINT,ID3D12Resource*,UINT64,ID3D12Resource*,UINT64);
using EnhancedFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList7*,UINT32,const D3D12_BARRIER_GROUP*);
using DrawFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,UINT,UINT,UINT,UINT);
using DrawIndexedFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,UINT,UINT,UINT,INT,UINT);
using CopyTextureFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,const D3D12_TEXTURE_COPY_LOCATION*,UINT,UINT,UINT,const D3D12_TEXTURE_COPY_LOCATION*,const D3D12_BOX*);
using CopyResourceFn=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,ID3D12Resource*,ID3D12Resource*);
std::atomic<DrawFn> drawOriginal{};std::atomic<DrawIndexedFn> drawIndexedOriginal{};
std::atomic<CopyTextureFn> copyTextureOriginal{};std::atomic<CopyResourceFn> copyResourceOriginal{};
std::atomic<PfnFfxCreateContext> createOriginal{};std::atomic<PfnFfxDestroyContext> destroyOriginal{};std::atomic<PfnFfxDispatch> dispatchOriginal{};
std::atomic<ResetFn> resetOriginal{};std::atomic<CloseFn> closeOriginal{};std::atomic<BarrierFn> barrierOriginal{};std::atomic<ExecuteFn> executeOriginal{};
std::atomic<GpuDispatchFn> gpuDispatchOriginal{};std::atomic<IndirectFn> indirectOriginal{};std::atomic<EnhancedFn> enhancedOriginal{};
// These counters describe only synchronous callbacks through installed tables.
// Zero does not mean no GPU work: driver-private/cached/worker-thread calls can bypass them.
struct PathCounts {uint64_t gpu=0,indirect=0,legacy=0,legacyBarriers=0,enhanced=0,enhancedGroups=0,otherList=0;};
struct PathScope {PathCounts counts;ID3D12GraphicsCommandList* list=nullptr;uint64_t window=0,generation=0;uint32_t sample=0;bool details=false;
    ID3D12Resource* roleResources[7]{};IUnknown* roleIdentities[7]{};uint32_t declaredStates[7]{};};
thread_local PathScope* currentPath=nullptr;
// Commands owned by this observer must bypass its global vtable hooks. Without
// this scope, constructing/closing a private list from a hooked callback can
// recursively acquire the session mutex and deadlock the host process.
thread_local uint32_t internalCommandDepth=0;
struct InternalCommandScope {InternalCommandScope(){++internalCommandDepth;}~InternalCommandScope(){--internalCommandDepth;}InternalCommandScope(const InternalCommandScope&)=delete;};
void PathCall(const void* list,uint64_t PathCounts::*field,uint64_t amount=1) {
    if(currentPath){currentPath->counts.*field+=amount;if(list!=currentPath->list)++currentPath->counts.otherList;}
}
std::mutex guard,installGuard;std::atomic<bool> active{false};std::atomic<ULONGLONG> deadline{0};
FILE* logFile=nullptr;std::filesystem::path root;bool initialized=false,testMode=false,hooksReady=false,captureWanted=false,failed=false,complete=false;
uint32_t samples=0,sampleLimit=64;uint64_t contextEpoch=0,eventCount=0;
uint32_t traceCount=0;constexpr uint32_t traceLimit=1024;
uint32_t resourcePathCount=0;constexpr uint32_t resourcePathLimit=512;
const char* resourceRoles[]={"color","depth","motion","exposure","reactive","transparency","output"};
std::atomic<uint64_t> windowEpoch{0},nextBatch{0};
constexpr uint64_t maxObservationWindows=8;
void** listTable=nullptr;void** queueTable=nullptr;
void** enhancedTable=nullptr;
struct Context {ffxCreateContextDescUpscale desc{};uint64_t epoch=0,provider=0;ComPtr<ID3D12Device> device;};
std::unordered_map<ffxContext,Context> contexts;
struct Resource {ComPtr<ID3D12Resource> object;ComPtr<IUnknown> identity;UINT planes=1;};
std::unordered_map<ID3D12Resource*,Resource> resources;
struct PlaneStates {int64_t value[2]={-1,-1};};
struct BoundaryPending {ID3D12Resource* output=nullptr;uint64_t window=0,generation=0;uint32_t sample=0;};
struct TailCounts {ID3D12Resource* target=nullptr;uint64_t window=0;uint32_t sample=0;uint64_t draw=0,drawIndexed=0,dispatch=0,indirect=0,copyTexture=0,copyResource=0,targetCopyReads=0,targetCopyWrites=0;};
struct List {ComPtr<ID3D12GraphicsCommandList> object;ComPtr<IUnknown> identity;uint64_t generation=0;bool reset=false,closed=false;uint32_t ffxCalls=0;BoundaryPending boundary;TailCounts tail;std::unordered_map<ID3D12Resource*,PlaneStates> states;};
std::unordered_map<ID3D12GraphicsCommandList*,List> lists;
std::unique_ptr<ffx_capture::Collector> collector;
ComPtr<ID3D12GraphicsCommandList> capturedList;uint64_t capturedGeneration=0,capturedTicket=0,capturedEpoch=0;
bool outputCopied=false,submitted=false;
// Version 3 is isolated-test-only. Live Attach rejects it. Uncertain jobs are
// deliberately retained rather than destroyed while GPU work may reference them.
bool boundaryTest=false,boundaryLiveArmed=false,boundaryWriteBack=false,boundaryPatchMode=false,boundaryFilterMode=false,boundaryNetworkMode=false,boundaryPreparing=false,boundaryRecorded=false,boundarySubmitted=false,boundaryDone=false;
ffx_boundary::OutputCopy* boundaryCopy=nullptr;
ffx_boundary::OutputRoundTrip* boundaryRoundTrip=nullptr;
ffx_boundary::OutputPatch* boundaryPatch=nullptr;
ffx_boundary::OutputFilter* boundaryFilter=nullptr;
ffx_boundary::OutputNetwork* boundaryNetwork=nullptr;
std::vector<uint8_t> boundaryFilterShader;
std::vector<uint8_t> boundaryNetworkShader;
std::array<float,256> boundaryNetworkWeights{};
// Retained until process exit; stop cannot release uploads referenced by GPU lists.
ffx_boundary::StaticPreview* staticPreview=nullptr;
std::vector<uint8_t> previewInput,previewOutput;
bool previewMode=false,previewPreparing=false,previewNetwork=true;
uint32_t previewFrames=0;
bool residentMode=false,residentPreparing=false,residentBusy=false;
std::atomic<bool> residentInstalled{false};std::recursive_mutex residentSubmitGuard;
ffx_resident::Exchange* residentExchange=nullptr;
ffx_resident::Surface* residentSurface=nullptr;
ComPtr<ID3D12GraphicsCommandList> residentList;ComPtr<IUnknown> residentIdentity;
ComPtr<ID3D12Resource> residentTarget;
uint64_t residentGeneration=0;uint32_t residentFrames=0;
constexpr uint32_t residentFrameLimit=12;
ComPtr<ID3D12GraphicsCommandList> boundaryCopyList;uint64_t boundaryCopyGeneration=0;
ComPtr<IUnknown> boundaryCopyIdentity;
UINT64 boundaryRawBytes=0;UINT boundaryWidth=0,boundaryHeight=0,boundaryFormat=0;
template<class T> bool Read(const void* p,T& t){SIZE_T n=0;return p&&ReadProcessMemory(GetCurrentProcess(),p,&t,sizeof(t),&n)&&n==sizeof(t);}
std::string Ptr(const void* p){char s[32];std::snprintf(s,sizeof(s),"\"0x%llx\"",(unsigned long long)(uintptr_t)p);return s;}
void Emit(const std::string& text) { // guard held; callbacks never throw through original APIs
    if(logFile&&eventCount++<4096){std::fprintf(logFile,"{\"window\":%llu,%s\n",(unsigned long long)windowEpoch.load(),text.c_str()+1);std::fflush(logFile);}
}
bool Tracking(){return active.load(std::memory_order_acquire)&&GetTickCount64()<deadline.load(std::memory_order_relaxed);}
bool SameWindow(uint64_t epoch){return epoch==windowEpoch.load(std::memory_order_acquire)&&Tracking();}
void Trace(const std::string& fields) { // Diagnostic only; never relaxes capture gates. Guard held.
    if(previewMode||residentMode)return; // continuous modes do not collect full-frame traces
    if(traceCount<traceLimit) {
        Emit("{\"event\":\"state_trace\",\"sequence\":"+std::to_string(++traceCount)+","+fields+"}");
        if(traceCount==traceLimit)Emit("{\"event\":\"state_trace_limit\",\"limit\":1024}");
    }
}
void Fail(const std::string& why){failed=true;captureWanted=false;active.store(false,std::memory_order_release);Emit(std::string("{\"event\":\"failure\",\"reason\":\"")+why+"\"}");}
List& GetList(ID3D12GraphicsCommandList* p) {
    auto it=lists.find(p);if(it!=lists.end())return it->second;
    if(lists.size()>=128)throw std::runtime_error("list cap");
    List a;a.object=p;ffx_capture::Check(p->QueryInterface(IID_PPV_ARGS(&a.identity)));return lists.emplace(p,std::move(a)).first->second;
}
void Watch(const ffxDispatchDescUpscale& a) {
    const FfxApiResource* entries[]={&a.color,&a.depth,&a.motionVectors,&a.exposure,&a.reactive,&a.transparencyAndComposition,&a.output};
    for(auto r:entries)if(r->resource) {
        auto* p=static_cast<ID3D12Resource*>(r->resource);if(resources.count(p))continue;
        if(resources.size()>=64)throw std::runtime_error("resource cap");
        Resource v;v.object=p;ffx_capture::Check(p->QueryInterface(IID_PPV_ARGS(&v.identity)));auto d=p->GetDesc();
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.MipLevels!=1||d.DepthOrArraySize!=1||d.SampleDesc.Count!=1)throw std::runtime_error("resource shape");
        v.planes=d.Format==DXGI_FORMAT_D32_FLOAT_S8X24_UINT?2:1;resources.emplace(p,std::move(v));
        Trace("\"kind\":\"watch\",\"resource\":"+Ptr(p)+",\"format\":"+std::to_string(d.Format));
    }
}
// Metadata only. COM identity is compared within this call, never by texture
// dimensions or by pointer reuse across lifetimes. No identities propagate states.
std::string ResourceInfo(ID3D12Resource* p,const PathScope& path) {
    std::ostringstream s;s<<"{\"resource\":"<<Ptr(p);
    if(p){ComPtr<IUnknown> identity;ffx_capture::Check(p->QueryInterface(IID_PPV_ARGS(&identity)));auto d=p->GetDesc();
        s<<",\"identity\":"<<Ptr(identity.Get())<<",\"dimension\":"<<d.Dimension<<",\"width\":"<<d.Width<<",\"height\":"<<d.Height<<",\"format\":"<<d.Format<<",\"mips\":"<<d.MipLevels<<",\"array_size\":"<<d.DepthOrArraySize<<",\"flags\":"<<d.Flags;
        for(bool canonical:{false,true}){s<<(canonical?",\"identity_matches\":[":",\"raw_matches\":[");bool first=true;
            for(unsigned i=0;i<7;++i)if(path.roleResources[i]&&(canonical?path.roleIdentities[i]==identity.Get():path.roleResources[i]==p)){if(!first)s<<',';first=false;s<<'"'<<resourceRoles[i]<<'"';}s<<']';}
    }s<<'}';return s.str();
}
void ResourcePath(const PathScope& path,const std::string& fields) { // guard held
    if(!path.details||!SameWindow(path.window)||resourcePathCount>=resourcePathLimit)return;
    Emit("{\"event\":\"resource_path\",\"sequence\":"+std::to_string(++resourcePathCount)+",\"sample\":"+std::to_string(path.sample)+",\"ffx_list\":"+Ptr(path.list)+",\"generation\":"+std::to_string(path.generation)+","+fields+"}");
    if(resourcePathCount==resourcePathLimit)Emit("{\"event\":\"resource_path_limit\",\"limit\":512}");
}
void ResourceSnapshot(PathScope& path,const ffxDispatchDescUpscale& a,bool before) { // guard held
    if(!path.details||resourcePathCount>=resourcePathLimit)return;
    const FfxApiResource* r[]={&a.color,&a.depth,&a.motionVectors,&a.exposure,&a.reactive,&a.transparencyAndComposition,&a.output};
    if(before)for(unsigned i=0;i<7;++i){auto* p=static_cast<ID3D12Resource*>(r[i]->resource);path.roleResources[i]=p;path.roleIdentities[i]=p?resources.at(p).identity.Get():nullptr;path.declaredStates[i]=r[i]->state;}
    std::ostringstream s;s<<"\"kind\":\""<<(before?"resources_before":"resources_after")<<"\",\"roles\":[";bool unchanged=a.commandList==path.list;
    for(unsigned i=0;i<7;++i){if(i)s<<',';auto* p=static_cast<ID3D12Resource*>(r[i]->resource);
        if(!before)unchanged&=path.roleResources[i]==p&&path.declaredStates[i]==r[i]->state;
        s<<"{\"role\":\""<<resourceRoles[i]<<"\",\"declared_ffx_state\":"<<r[i]->state<<",\"info\":"<<ResourceInfo(p,path)<<'}';}
    s<<"],\"command_list\":"<<Ptr(a.commandList)<<",\"resource_fields_unchanged\":"<<(unchanged?"true":"false");ResourcePath(path,s.str());
}
bool Patch(void** slot,void* expected,void* replacement) {
    DWORD old=0,ignored=0;if(!VirtualProtect(slot,sizeof(void*),PAGE_READWRITE,&old))return false;
    void* previous=InterlockedCompareExchangePointer(slot,replacement,expected);
    if(!VirtualProtect(slot,sizeof(void*),old,&ignored)){std::lock_guard<std::mutex> lock(guard);Fail("page protection restore");}
    return previous==expected;
}
HRESULT STDMETHODCALLTYPE Reset(ID3D12GraphicsCommandList* p,ID3D12CommandAllocator* a,ID3D12PipelineState* s) {
    if(internalCommandDepth)return resetOriginal.load(std::memory_order_acquire)(p,a,s);
    auto epoch=windowEpoch.load(std::memory_order_acquire);
    auto status=resetOriginal.load(std::memory_order_acquire)(p,a,s);DWORD error=GetLastError();
    if(SameWindow(epoch))try{std::lock_guard<std::mutex> lock(guard);if(!SameWindow(epoch)){SetLastError(error);return status;}auto& l=GetList(p);
        if(capturedList.Get()==p&&capturedTicket&&!submitted){Fail("captured list reset before observed submission; references retained");}
        if(residentList.Get()==p&&(residentSurface||residentPreparing)&&!residentBusy)Fail("resident producer reset before submission");
        if(boundaryCopyList.Get()==p&&(boundaryCopy||boundaryRoundTrip||boundaryPatch||boundaryFilter||boundaryNetwork||boundaryPreparing)&&!boundarySubmitted)Fail("boundary list reset before submission; references retained");
        if(!l.states.empty())Trace("\"kind\":\"reset\",\"list\":"+Ptr(p)+",\"generation\":"+std::to_string(l.generation)+",\"succeeded\":"+(SUCCEEDED(status)?"true":"false"));
        ++l.generation;l.states.clear();l.closed=false;l.reset=SUCCEEDED(status);l.ffxCalls=0;l.boundary={};l.tail={};
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("reset tracking cap/query");}
    SetLastError(error);return status;
}
HRESULT STDMETHODCALLTYPE Close(ID3D12GraphicsCommandList* p) {
    if(internalCommandDepth)return closeOriginal.load(std::memory_order_acquire)(p);
    auto epoch=windowEpoch.load(std::memory_order_acquire);
    auto status=closeOriginal.load(std::memory_order_acquire)(p);DWORD error=GetLastError();
    if(SameWindow(epoch))try{std::lock_guard<std::mutex> lock(guard);if(!SameWindow(epoch)){SetLastError(error);return status;}GetList(p).closed=SUCCEEDED(status);
        auto& l=GetList(p);if(!l.states.empty())Trace("\"kind\":\"close\",\"list\":"+Ptr(p)+",\"generation\":"+std::to_string(l.generation)+",\"succeeded\":"+(SUCCEEDED(status)?"true":"false"));
        if(l.tail.target&&l.tail.window==epoch){const auto& t=l.tail;std::ostringstream text;
            text<<"{\"event\":\"output_boundary_tail\",\"list\":"<<Ptr(p)<<",\"generation\":"<<l.generation<<",\"sample\":"<<t.sample
                <<",\"target\":"<<Ptr(t.target)<<",\"draw\":"<<t.draw<<",\"draw_indexed\":"<<t.drawIndexed<<",\"dispatch\":"<<t.dispatch
                <<",\"indirect\":"<<t.indirect<<",\"copy_texture\":"<<t.copyTexture<<",\"copy_resource\":"<<t.copyResource
                <<",\"target_copy_reads\":"<<t.targetCopyReads<<",\"target_copy_writes\":"<<t.targetCopyWrites
                <<",\"close_succeeded\":"<<(SUCCEEDED(status)?"true":"false")<<",\"metadata_only\":true,\"shader_resource_reads_resolved\":false}";Emit(text.str());l.tail={};}
        l.boundary={};
        if(FAILED(status)&&capturedList.Get()==p)Fail("captured list close failed");
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("close tracking cap/query");}
    SetLastError(error);return status;
}
void STDMETHODCALLTYPE Barrier(ID3D12GraphicsCommandList* p,UINT count,const D3D12_RESOURCE_BARRIER* barriers) {
    if(internalCommandDepth){barrierOriginal.load(std::memory_order_acquire)(p,count,barriers);return;}
    DWORD incoming=GetLastError();auto epoch=windowEpoch.load(std::memory_order_acquire);
    BoundaryPending boundary;ffx_boundary::Batch batch;bool boundarySelected=false;
    ffx_boundary::OutputCopy* recordBoundary=nullptr;ffx_boundary::OutputRoundTrip* recordRoundTrip=nullptr;ffx_boundary::OutputPatch* recordPatch=nullptr;
    ComPtr<ID3D12Device> filterDevice;ComPtr<ID3D12Resource> filterOutput;bool prepareFilter=false,prepareNetwork=false;
    ComPtr<ID3D12Device> previewDevice;ffx_boundary::StaticPreview* recordPreview=nullptr;
    bool preparePreview=false,selectedNetwork=false;
    bool prepareResident=false;ComPtr<ID3D12Device> residentDevice;ComPtr<ID3D12Resource> residentOutput;
    PathCall(p,&PathCounts::legacy);if(currentPath)currentPath->counts.legacyBarriers+=count;
    if(SameWindow(epoch))try {
        std::lock_guard<std::mutex> lock(guard);
        if(SameWindow(epoch)) {
        if(count>4096)throw std::runtime_error("barrier cap");
        auto pending=lists.find(p);
        if(residentMode&&residentList.Get()==p&&(residentSurface||residentPreparing)&&!currentPath&&count){
            // A later target transition/alias invalidates the end-of-producer state.
            for(UINT i=0;i<count;++i)if(barriers[i].Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING||
                (barriers[i].Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION&&barriers[i].Transition.pResource==residentTarget.Get())){
                Fail("resident producer has later transition; no replacement submitted");break;}
        }
        if(pending!=lists.end()&&pending->second.boundary.output&&!currentPath){
            auto& l=pending->second;
            batch=ffx_boundary::InspectBatch(l.boundary.output,count,barriers);
            if(batch.relevant){boundary=l.boundary;l.boundary={}; // consume BEFORE forwarding: reentry cannot select twice
                boundarySelected=boundary.window==epoch&&boundary.generation==l.generation;}
        }
        for(UINT i=0;i<count;++i) {
            const auto& b=barriers[i];
            if(currentPath&&currentPath->details&&resourcePathCount<resourcePathLimit){
                std::ostringstream s;s<<"\"kind\":\"barrier\",\"list\":"<<Ptr(p)<<",\"type\":"<<b.Type<<",\"flags\":"<<b.Flags;
                if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)s<<",\"before\":"<<b.Transition.StateBefore<<",\"after\":"<<b.Transition.StateAfter<<",\"subresource\":"<<b.Transition.Subresource<<",\"info\":"<<ResourceInfo(b.Transition.pResource,*currentPath);
                else if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_UAV)s<<",\"info\":"<<ResourceInfo(b.UAV.pResource,*currentPath);
                else if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING)s<<",\"before_info\":"<<ResourceInfo(b.Aliasing.pResourceBefore,*currentPath)<<",\"after_info\":"<<ResourceInfo(b.Aliasing.pResourceAfter,*currentPath);
                ResourcePath(*currentPath,s.str());
            }
            if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
                auto r=resources.find(b.Transition.pResource);if(r==resources.end())continue;
                auto& l=GetList(p);auto& states=l.states[b.Transition.pResource];
                Trace("\"kind\":\"transition\",\"list\":"+Ptr(p)+",\"generation\":"+std::to_string(l.generation)+",\"resource\":"+Ptr(b.Transition.pResource)+",\"subresource\":"+std::to_string(b.Transition.Subresource)+",\"flags\":"+std::to_string(b.Flags)+",\"before\":"+std::to_string(b.Transition.StateBefore)+",\"after\":"+std::to_string(b.Transition.StateAfter));
                for(UINT plane=0;plane<r->second.planes;++plane)
                    if(b.Transition.Subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES||b.Transition.Subresource==plane)
                        states.value[plane]=b.Flags==D3D12_RESOURCE_BARRIER_FLAG_NONE?int64_t(b.Transition.StateAfter):-1;
            } else if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
                if(!resources.empty())Trace("\"kind\":\"alias\",\"list\":"+Ptr(p));
                // Alias contents/ownership need a separate proof. Conservatively
                // invalidate recorded states rather than infer safety.
                auto it=lists.find(p);if(it!=lists.end())it->second.states.clear();
            }
        }
        }
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("barrier tracking invalid/cap");}
    SetLastError(incoming);barrierOriginal.load(std::memory_order_acquire)(p,count,barriers);DWORD outgoing=GetLastError();
    if(boundarySelected&&SameWindow(epoch))try{std::lock_guard<std::mutex> lock(guard);if(SameWindow(epoch)){
        auto it=lists.find(p);bool lifecycle=it!=lists.end()&&it->second.generation==boundary.generation&&it->second.reset&&!it->second.closed&&it->second.ffxCalls==1;
        const auto& t=batch.target.Transition;
        if(batch.accepted&&lifecycle){auto& tail=it->second.tail;tail={};tail.target=boundary.output;tail.window=epoch;tail.sample=boundary.sample;}
        std::ostringstream s;s<<"{\"event\":\"output_boundary_return\",\"sample\":"<<boundary.sample<<",\"list\":"<<Ptr(p)<<",\"generation\":"<<boundary.generation
          <<",\"resource\":"<<Ptr(boundary.output)<<",\"barrier_count\":"<<count<<",\"target_transitions\":"<<batch.transitions<<",\"alias_barriers\":"<<batch.aliases<<",\"unknown_barriers\":"<<batch.unknown
          <<",\"last_target_before\":"<<t.StateBefore<<",\"last_target_after\":"<<t.StateAfter<<",\"last_target_subresource\":"<<t.Subresource<<",\"last_target_flags\":"<<batch.target.Flags
          <<",\"batch_supported\":"<<(batch.accepted?"true":"false")<<",\"lifecycle_match\":"<<(lifecycle?"true":"false")<<",\"original_callback_returned\":true,\"original_forward_calls\":1,\"capture_authorized\":false}";Emit(s.str());
        if(previewMode&&batch.accepted&&lifecycle&&!failed&&!previewPreparing&&previewFrames<18000){
            selectedNetwork=previewNetwork;
            if(staticPreview)recordPreview=staticPreview;
            else{ffx_capture::Check(p->GetDevice(IID_PPV_ARGS(&previewDevice)));previewPreparing=true;preparePreview=true;}
        }
        if(residentMode&&batch.accepted&&lifecycle&&!failed&&!residentPreparing&&!residentBusy&&!residentSurface&&residentFrames<residentFrameLimit){
            ffx_capture::Check(p->GetDevice(IID_PPV_ARGS(&residentDevice)));residentOutput=boundary.output;
            residentPreparing=true;prepareResident=true;residentList=p;residentIdentity=it->second.identity;residentGeneration=boundary.generation;residentTarget=boundary.output;
        }
        if(!previewMode&&(boundaryTest||boundaryLiveArmed)&&batch.accepted&&lifecycle&&!boundaryCopy&&!boundaryRoundTrip&&!boundaryPatch&&!boundaryFilter&&!boundaryNetwork&&!boundaryPreparing&&!boundaryDone&&!failed){
            ComPtr<ID3D12Device> device;ffx_capture::Check(p->GetDevice(IID_PPV_ARGS(&device)));
            auto desc=boundary.output->GetDesc();boundaryWidth=UINT(desc.Width);boundaryHeight=desc.Height;boundaryFormat=desc.Format;
            if(boundaryNetworkMode){filterDevice=device;filterOutput=boundary.output;boundaryPreparing=true;prepareNetwork=true;}
            else if(boundaryFilterMode){filterDevice=device;filterOutput=boundary.output;boundaryPreparing=true;prepareFilter=true;}
            else if(boundaryPatchMode){boundaryPatch=new ffx_boundary::OutputPatch(device.Get(),boundary.output);recordPatch=boundaryPatch;}
            else if(boundaryWriteBack){boundaryRoundTrip=new ffx_boundary::OutputRoundTrip(device.Get(),boundary.output);recordRoundTrip=boundaryRoundTrip;}
            else{boundaryCopy=new ffx_boundary::OutputCopy(device.Get(),boundary.output);recordBoundary=boundaryCopy;}
            boundaryCopyList=p;boundaryCopyGeneration=boundary.generation;
            boundaryCopyIdentity=GetList(p).identity;
            boundaryLiveArmed=false;
        }
    }}catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);Fail(std::string("output boundary return diagnostic: ")+e.what());}
      catch(...){std::lock_guard<std::mutex> lock(guard);Fail("output boundary return diagnostic");}
    if(prepareResident)try{
        InternalCommandScope internal;auto* surface=new ffx_resident::Surface(residentDevice.Get(),residentOutput.Get());
        std::lock_guard<std::mutex> lock(guard);residentSurface=surface;residentPreparing=false;
    }catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);residentPreparing=false;Fail(std::string("resident preparation: ")+e.what());}
    if(preparePreview||recordPreview)try{
        InternalCommandScope internal;
        if(preparePreview){auto* candidate=new ffx_boundary::StaticPreview(previewDevice.Get(),previewInput,previewOutput);
            std::lock_guard<std::mutex> lock(guard);staticPreview=candidate;previewPreparing=false;recordPreview=candidate;}
        bool enabled=false;{std::lock_guard<std::mutex> lock(guard);enabled=SameWindow(epoch)&&previewMode&&!failed;}
        if(enabled){recordPreview->Record(p,batch.target,selectedNetwork);
            std::lock_guard<std::mutex> lock(guard);++previewFrames;
            if(previewFrames==1||previewFrames%300==0)Emit("{\"event\":\"static_preview_recorded\",\"frames\":"+std::to_string(previewFrames)+",\"network_output\":"+(selectedNetwork?"true":"false")+",\"live_inference\":false,\"resources_retained\":true,\"internal_hook_bypass\":true}");}
    }catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);previewPreparing=false;Fail(std::string("static preview: ")+e.what());}
    // Copy callbacks reenter Barrier; no session lock and no pending target.
    if(recordBoundary)try{recordBoundary->Record(p,batch.target);std::lock_guard<std::mutex> lock(guard);boundaryRecorded=true;
        Emit(std::string("{\"event\":\"boundary_output_recorded\",\"isolated_host_only\":")+(testMode?"true":"false")+",\"write_back_performed\":false,\"game_frame_capture_verified\":false}");
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("boundary recording uncertain; references retained");}
    if(recordRoundTrip)try{recordRoundTrip->Record(p,batch.target);std::lock_guard<std::mutex> lock(guard);boundaryRecorded=true;
        Emit(std::string("{\"event\":\"boundary_output_recorded\",\"isolated_host_only\":")+(testMode?"true":"false")+",\"write_back_performed\":true,\"game_frame_capture_verified\":false}");
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("boundary roundtrip recording uncertain; references retained");}
    if(recordPatch)try{recordPatch->Record(p,batch.target);std::lock_guard<std::mutex> lock(guard);boundaryRecorded=true;
        Emit(std::string("{\"event\":\"boundary_output_recorded\",\"isolated_host_only\":")+(testMode?"true":"false")+",\"write_back_performed\":true,\"replacement_pixels_supplied\":true,\"game_frame_capture_verified\":false}");
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("boundary patch recording uncertain; references retained");}
    if(prepareFilter){ffx_boundary::OutputFilter* candidate=nullptr;try{{InternalCommandScope internal;candidate=new ffx_boundary::OutputFilter(filterDevice.Get(),filterOutput.Get(),boundaryFilterShader);candidate->Record(batch.target);}
        std::lock_guard<std::mutex> lock(guard);boundaryFilter=candidate;boundaryPreparing=false;boundaryRecorded=true;
        Emit(std::string("{\"event\":\"boundary_output_recorded\",\"isolated_host_only\":")+(testMode?"true":"false")+",\"write_back_performed\":true,\"replacement_pixels_supplied\":true,\"gpu_residual_filter_recorded\":true,\"internal_hook_bypass\":true,\"game_frame_capture_verified\":false}");
    }catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);boundaryPreparing=false;Fail(std::string("boundary filter recording uncertain: ")+e.what());}
      catch(...){std::lock_guard<std::mutex> lock(guard);boundaryPreparing=false;Fail("boundary filter recording uncertain; references retained");}}
    if(prepareNetwork){ffx_boundary::OutputNetwork* candidate=nullptr;try{{InternalCommandScope internal;candidate=new ffx_boundary::OutputNetwork(filterDevice.Get(),filterOutput.Get(),boundaryNetworkShader,boundaryNetworkWeights);candidate->Record(batch.target);}
        std::lock_guard<std::mutex> lock(guard);boundaryNetwork=candidate;boundaryPreparing=false;boundaryRecorded=true;
        Emit(std::string("{\"event\":\"boundary_output_recorded\",\"isolated_host_only\":")+(testMode?"true":"false")+",\"write_back_performed\":true,\"replacement_pixels_supplied\":true,\"gpu_residual_network_recorded\":true,\"external_weights\":true,\"internal_hook_bypass\":true,\"game_frame_capture_verified\":false}");
    }catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);boundaryPreparing=false;Fail(std::string("boundary network recording uncertain: ")+e.what());}
      catch(...){std::lock_guard<std::mutex> lock(guard);boundaryPreparing=false;Fail("boundary network recording uncertain; references retained");}}
    SetLastError(outgoing);
}
void TailCall(ID3D12GraphicsCommandList* p,uint64_t TailCounts::*field,ID3D12Resource* destination=nullptr,ID3D12Resource* source=nullptr){
    if(internalCommandDepth||currentPath)return;DWORD incoming=GetLastError();auto epoch=windowEpoch.load(std::memory_order_acquire);
    if(SameWindow(epoch))try{std::lock_guard<std::mutex> lock(guard);auto it=lists.find(p);
        if(SameWindow(epoch)&&it!=lists.end()&&!it->second.closed&&it->second.tail.target&&it->second.tail.window==epoch){auto& t=it->second.tail;
            if(t.*field<100000)++(t.*field);t.targetCopyReads+=source==t.target;t.targetCopyWrites+=destination==t.target;
        }}catch(...){/* metadata failure never changes the original draw/copy */}
    SetLastError(incoming);
}
void STDMETHODCALLTYPE Draw(ID3D12GraphicsCommandList* p,UINT a,UINT b,UINT c,UINT d){TailCall(p,&TailCounts::draw);drawOriginal.load()(p,a,b,c,d);}
void STDMETHODCALLTYPE DrawIndexed(ID3D12GraphicsCommandList* p,UINT a,UINT b,UINT c,INT d,UINT e){TailCall(p,&TailCounts::drawIndexed);drawIndexedOriginal.load()(p,a,b,c,d,e);}
void STDMETHODCALLTYPE CopyTexture(ID3D12GraphicsCommandList* p,const D3D12_TEXTURE_COPY_LOCATION* d,UINT x,UINT y,UINT z,const D3D12_TEXTURE_COPY_LOCATION* s,const D3D12_BOX* box){
    DWORD incoming=GetLastError();D3D12_TEXTURE_COPY_LOCATION dst{},src{};
    if(!internalCommandDepth){Read(d,dst);Read(s,src);TailCall(p,&TailCounts::copyTexture,dst.pResource,src.pResource);}
    SetLastError(incoming);copyTextureOriginal.load()(p,d,x,y,z,s,box);
}
void STDMETHODCALLTYPE CopyResource(ID3D12GraphicsCommandList* p,ID3D12Resource* d,ID3D12Resource* s){TailCall(p,&TailCounts::copyResource,d,s);copyResourceOriginal.load()(p,d,s);}
void STDMETHODCALLTYPE GpuDispatch(ID3D12GraphicsCommandList* p,UINT x,UINT y,UINT z) {
    if(internalCommandDepth){gpuDispatchOriginal.load(std::memory_order_acquire)(p,x,y,z);return;}
    TailCall(p,&TailCounts::dispatch);PathCall(p,&PathCounts::gpu);gpuDispatchOriginal.load(std::memory_order_acquire)(p,x,y,z);
}
void STDMETHODCALLTYPE Indirect(ID3D12GraphicsCommandList* p,ID3D12CommandSignature* signature,UINT maximum,
                               ID3D12Resource* arguments,UINT64 argumentOffset,ID3D12Resource* count,UINT64 countOffset) {
    if(internalCommandDepth){indirectOriginal.load(std::memory_order_acquire)(p,signature,maximum,arguments,argumentOffset,count,countOffset);return;}
    TailCall(p,&TailCounts::indirect);PathCall(p,&PathCounts::indirect);indirectOriginal.load(std::memory_order_acquire)(p,signature,maximum,arguments,argumentOffset,count,countOffset);
}
void STDMETHODCALLTYPE Enhanced(ID3D12GraphicsCommandList7* p,UINT32 count,const D3D12_BARRIER_GROUP* groups) {
    if(internalCommandDepth){enhancedOriginal.load(std::memory_order_acquire)(p,count,groups);return;}
    DWORD incoming=GetLastError();auto epoch=windowEpoch.load(std::memory_order_acquire);
    PathCall(p,&PathCounts::enhanced);if(currentPath)currentPath->counts.enhancedGroups+=count;
    // Enhanced states/layouts are not translated into legacy states. Withhold
    // existing local-state knowledge when this tracked interface uses them.
    if(SameWindow(epoch))try {std::lock_guard<std::mutex> lock(guard);if(SameWindow(epoch)){
        ComPtr<ID3D12GraphicsCommandList> base;ffx_capture::Check(p->QueryInterface(IID_PPV_ARGS(&base)));
        auto it=lists.find(base.Get());if(it!=lists.end()){
            it->second.states.clear();it->second.boundary={};Trace("\"kind\":\"enhanced_state_unknown\",\"list\":"+Ptr(base.Get())+",\"generation\":"+std::to_string(it->second.generation)+",\"group_count\":"+std::to_string(count));
        }
    }}catch(...){std::lock_guard<std::mutex> lock(guard);Fail("enhanced state invalidation");}
    SetLastError(incoming);enhancedOriginal.load(std::memory_order_acquire)(p,count,groups);
}
void STDMETHODCALLTYPE Execute(ID3D12CommandQueue* q,UINT count,ID3D12CommandList*const* submittedLists) {
    if(internalCommandDepth){executeOriginal.load(std::memory_order_acquire)(q,count,submittedLists);return;}
    std::unique_lock<std::recursive_mutex> residentSerial(residentSubmitGuard,std::defer_lock);
    if(residentInstalled.load(std::memory_order_acquire))residentSerial.lock();
    auto epoch=windowEpoch.load(std::memory_order_acquire);auto batch=++nextBatch;DWORD incoming=GetLastError();
    if(SameWindow(epoch))try {
        std::lock_guard<std::mutex> lock(guard);
        if(SameWindow(epoch)&&count<=64&&traceCount<traceLimit) {
            std::ostringstream s;s<<"\"kind\":\"submit_begin\",\"queue\":"<<Ptr(q)<<",\"batch_id\":"<<batch<<",\"batch_count\":"<<count<<",\"lists\":[";
            for(UINT i=0;i<count;++i){if(i)s<<',';ComPtr<ID3D12GraphicsCommandList> l;
                if(submittedLists[i])submittedLists[i]->QueryInterface(IID_PPV_ARGS(&l));auto it=lists.find(l.Get());
                s<<"{\"index\":"<<i<<",\"list\":"<<Ptr(l.Get())<<",\"tracked\":"<<(it!=lists.end()?"true":"false")<<",\"generation\":"<<(it==lists.end()?0:it->second.generation)<<'}';}
            s<<']';Trace(s.str());
        }
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("submission begin trace");}
    bool splitForFilter=false;UINT filterIndex=0;ID3D12CommandList* insertedList=nullptr;
    bool splitResident=false;UINT residentIndex=0;ffx_resident::Surface* selectedSurface=nullptr;
    if(SameWindow(epoch))try{std::lock_guard<std::mutex> lock(guard);
        if(SameWindow(epoch)&&residentMode&&residentSurface&&!residentBusy&&!failed){
            if(!count||count>64)throw std::runtime_error("resident batch cap");
            unsigned matches=0;for(UINT i=0;i<count;++i){ComPtr<IUnknown> identity;
                if(submittedLists[i]&&SUCCEEDED(submittedLists[i]->QueryInterface(IID_PPV_ARGS(&identity)))&&identity.Get()==residentIdentity.Get()){++matches;residentIndex=i;}}
            if(matches>1)throw std::runtime_error("resident duplicate submission");
            if(matches==1){auto it=lists.find(residentList.Get());
                if(it==lists.end()||!it->second.reset||it->second.generation!=residentGeneration)throw std::runtime_error("resident producer generation");
                unsigned closed=0;for(const auto& entry:lists)if(entry.second.identity.Get()==residentIdentity.Get()&&entry.second.closed&&entry.second.generation==residentGeneration)++closed;
                if(!closed)throw std::runtime_error("resident canonical close");
                selectedSurface=residentSurface;residentBusy=true;splitResident=true;
            }
        }
    }catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);Fail(std::string("resident split: ")+e.what());}
    if(SameWindow(epoch))try{
        std::lock_guard<std::mutex> lock(guard);
        if(SameWindow(epoch)&&(boundaryFilterMode||boundaryNetworkMode)&&boundaryRecorded&&!boundarySubmitted&&!failed){
            if(!count||count>64)throw std::runtime_error("filter submission count cap");
            unsigned matches=0;
            for(UINT i=0;i<count;++i){ComPtr<IUnknown> identity;if(submittedLists[i]&&SUCCEEDED(submittedLists[i]->QueryInterface(IID_PPV_ARGS(&identity)))&&identity.Get()==boundaryCopyIdentity.Get()){++matches;filterIndex=i;}}
            if(matches>1)throw std::runtime_error("filter duplicate submission");
            if(matches==1){auto recorded=lists.find(boundaryCopyList.Get());if(recorded==lists.end()||!recorded->second.reset||recorded->second.generation!=boundaryCopyGeneration)throw std::runtime_error("filter recorded generation");
                unsigned closedAliases=0;for(const auto& entry:lists)if(entry.second.identity.Get()==boundaryCopyIdentity.Get()&&entry.second.closed&&entry.second.generation==boundaryCopyGeneration)++closedAliases;
                if(!closedAliases)throw std::runtime_error("filter canonical close");splitForFilter=true;insertedList=boundaryNetworkMode?boundaryNetwork->CommandList():boundaryFilter->CommandList();}
        }
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("filter split precondition; original batch preserved");}
    // Do not hold the session lock across the original call or older observers.
    SetLastError(incoming);
    auto originalExecute=executeOriginal.load(std::memory_order_acquire);
    if(splitResident){
        originalExecute(q,residentIndex+1,submittedLists);
        try{
            InternalCommandScope internal;auto result=selectedSurface->Run(q,originalExecute,*residentExchange);
            // Audit only after both queues' work completed. No GPU pointers escape.
            const auto seq=residentExchange->Sequence();
            if(seq<=2){auto folder=root/L"resident_frames";std::filesystem::create_directories(folder);
                auto save=[&](const wchar_t* prefix,const std::vector<uint8_t>& raw){std::ofstream f(folder/(std::wstring(prefix)+std::to_wstring(seq)+L".raw"),std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),raw.size());if(!f)throw std::runtime_error("resident audit output");};
                save(L"input",result.input);save(L"output",result.output);}
            std::lock_guard<std::mutex> lock(guard);++residentFrames;
            Emit("{\"event\":\"resident_frame_complete\",\"request_id\":"+std::to_string(seq)+",\"width\":"+std::to_string(result.width)+",\"height\":"+std::to_string(result.height)+",\"worker_output_readback_exact\":true,\"same_submission_frame\":true,\"batch_index\":"+std::to_string(residentIndex)+",\"suffix_lists\":"+std::to_string(count-residentIndex-1)+",\"independent_tiles\":true,\"dlss5_quality_verified\":false}");
            if(residentFrames>=residentFrameLimit){residentMode=false;active.store(false,std::memory_order_release);}
        }catch(const std::exception& e){std::lock_guard<std::mutex> lock(guard);Fail(std::string("resident frame: ")+e.what());residentMode=false;}
          catch(...){std::lock_guard<std::mutex> lock(guard);Fail("resident frame unknown error");residentMode=false;}
        // Always forward the untouched downstream suffix, even on worker failure.
        if(residentIndex+1<count)originalExecute(q,count-residentIndex-1,submittedLists+residentIndex+1);
        const bool retired=selectedSurface->SafeToDestroy();
        if(retired){InternalCommandScope internal;delete selectedSurface;}
        {std::lock_guard<std::mutex> lock(guard);if(retired)residentSurface=nullptr;residentBusy=false;residentList.Reset();residentIdentity.Reset();residentTarget.Reset();}
    }else if(splitForFilter){
        originalExecute(q,filterIndex+1,submittedLists);
        ID3D12CommandList* inserted[]={insertedList};originalExecute(q,1,inserted);
        if(filterIndex+1<count)originalExecute(q,count-filterIndex-1,submittedLists+filterIndex+1);
    }else originalExecute(q,count,submittedLists);
    DWORD error=GetLastError();
    if(SameWindow(epoch))try {
        std::lock_guard<std::mutex> lock(guard);
        if(!SameWindow(epoch)){SetLastError(error);return;}
        if(count<=64&&traceCount<traceLimit)for(UINT i=0;i<count;++i) {
            ComPtr<ID3D12GraphicsCommandList> list;
            if(submittedLists[i]&&SUCCEEDED(submittedLists[i]->QueryInterface(IID_PPV_ARGS(&list)))) {
                auto it=lists.find(list.Get());if(it!=lists.end()&&!it->second.states.empty())
                    Trace("\"kind\":\"submit_return\",\"queue\":"+Ptr(q)+",\"batch_id\":"+std::to_string(batch)+",\"list\":"+Ptr(list.Get())+",\"generation\":"+std::to_string(it->second.generation)+",\"batch_index\":"+std::to_string(i)+",\"batch_count\":"+std::to_string(count));
            }
        }
        if(count<=64)Trace("\"kind\":\"submit_end\",\"queue\":"+Ptr(q)+",\"batch_id\":"+std::to_string(batch));
        if(boundaryRecorded&&!boundarySubmitted&&!failed){
            if(count>64)throw std::runtime_error("boundary submission count cap");
            unsigned matches=0,submittedBatchIndex=0;bool rawPointerMatch=false;
            for(UINT i=0;i<count;++i){ComPtr<IUnknown> identity;if(submittedLists[i]&&SUCCEEDED(submittedLists[i]->QueryInterface(IID_PPV_ARGS(&identity)))&&identity.Get()==boundaryCopyIdentity.Get()){
                ++matches;submittedBatchIndex=i;ComPtr<ID3D12GraphicsCommandList> submitted;submittedLists[i]->QueryInterface(IID_PPV_ARGS(&submitted));rawPointerMatch|=submitted.Get()==boundaryCopyList.Get();}}
            if(matches>1)throw std::runtime_error("boundary duplicate submission");
            if(matches==1){
                auto recorded=lists.find(boundaryCopyList.Get());
                if(recorded==lists.end()||!recorded->second.reset||recorded->second.generation!=boundaryCopyGeneration)throw std::runtime_error("boundary recorded generation");
                unsigned identityAliases=0,closedAliases=0;
                for(const auto& entry:lists)if(entry.second.identity.Get()==boundaryCopyIdentity.Get()){
                    ++identityAliases;if(entry.second.closed&&entry.second.generation==boundaryCopyGeneration)++closedAliases;}
                if(!closedAliases)throw std::runtime_error("boundary canonical close");
                if(boundaryFilterMode||boundaryNetworkMode){if(!splitForFilter||filterIndex!=submittedBatchIndex)throw std::runtime_error("filter split association");if(boundaryNetworkMode)boundaryNetwork->Submitted(q);else boundaryFilter->Submitted(q);}
                else if(boundaryPatchMode)boundaryPatch->Submitted(q,boundaryCopyList.Get());else if(boundaryWriteBack)boundaryRoundTrip->Submitted(q,boundaryCopyList.Get());else boundaryCopy->Submitted(q,boundaryCopyList.Get());boundarySubmitted=true;active.store(false,std::memory_order_release);
                Emit(std::string("{\"event\":\"boundary_output_submitted\",\"isolated_host_only\":")+(testMode?"true":"false")+
                     ",\"write_back_performed\":"+(boundaryWriteBack?"true":"false")+",\"replacement_pixels_supplied\":"+((boundaryPatchMode||boundaryFilterMode||boundaryNetworkMode)?"true":"false")+",\"game_frame_capture_verified\":false,\"submitted_identity_matches\":1,\"identity_aliases\":"+std::to_string(identityAliases)+
                     ",\"closed_identity_aliases\":"+std::to_string(closedAliases)+",\"raw_pointer_match\":"+(rawPointerMatch?"true":"false")+",\"submission_batch_count\":"+std::to_string(count)+",\"submission_batch_index\":"+std::to_string(submittedBatchIndex)+",\"submission_suffix_lists\":"+std::to_string(count-submittedBatchIndex-1)+",\"submission_split_performed\":"+((boundaryFilterMode||boundaryNetworkMode)?"true":"false")+"}");}
        }
        if(collector&&outputCopied&&!submitted&&!failed&&count<=64) {
            auto it=lists.find(capturedList.Get());
            if(it==lists.end()||!it->second.closed||it->second.generation!=capturedGeneration)throw std::runtime_error("capture generation/close");
            unsigned matches=0;
            for(UINT i=0;i<count;++i){ComPtr<IUnknown> identity;if(submittedLists[i]&&SUCCEEDED(submittedLists[i]->QueryInterface(IID_PPV_ARGS(&identity)))&&identity.Get()==it->second.identity.Get())++matches;}
            if(matches>1)throw std::runtime_error("duplicate captured list submission");
            if(matches==1) {
                collector->Submitted(capturedTicket,q,capturedList.Get());submitted=true;
                Emit("{\"event\":\"capture_submitted\",\"ticket\":"+std::to_string(capturedTicket)+",\"generation\":"+std::to_string(capturedGeneration)+",\"queue\":"+Ptr(q)+"}");
                // No further instrumentation work is needed; Poll retires safely.
                active.store(false,std::memory_order_release);
            }
        }
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("submission association/signal uncertain; references retained");}
    SetLastError(error);
}
bool Install(ID3D12Device* d) {
    std::lock_guard<std::mutex> serial(installGuard);if(hooksReady)return true;
    ComPtr<ID3D12CommandAllocator> a;ComPtr<ID3D12GraphicsCommandList> l;ComPtr<ID3D12CommandQueue> q;
    if(FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&a)))||FAILED(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,a.Get(),nullptr,IID_PPV_ARGS(&l))))return false;
    if(FAILED(l->Close()))return false;D3D12_COMMAND_QUEUE_DESC desc{};if(FAILED(d->CreateCommandQueue(&desc,IID_PPV_ARGS(&q))))return false;
    listTable=*reinterpret_cast<void***>(l.Get());queueTable=*reinterpret_cast<void***>(q.Get());
    closeOriginal.store(reinterpret_cast<CloseFn>(listTable[9]));resetOriginal.store(reinterpret_cast<ResetFn>(listTable[10]));
    barrierOriginal.store(reinterpret_cast<BarrierFn>(listTable[26]));executeOriginal.store(reinterpret_cast<ExecuteFn>(queueTable[10]));
    gpuDispatchOriginal.store(reinterpret_cast<GpuDispatchFn>(listTable[14]));indirectOriginal.store(reinterpret_cast<IndirectFn>(listTable[59]));
    drawOriginal.store(reinterpret_cast<DrawFn>(listTable[12]));drawIndexedOriginal.store(reinterpret_cast<DrawIndexedFn>(listTable[13]));
    copyTextureOriginal.store(reinterpret_cast<CopyTextureFn>(listTable[16]));copyResourceOriginal.store(reinterpret_cast<CopyResourceFn>(listTable[17]));
    ComPtr<ID3D12GraphicsCommandList7> extended;
    if(SUCCEEDED(l.As(&extended))){enhancedTable=*reinterpret_cast<void***>(extended.Get());enhancedOriginal.store(reinterpret_cast<EnhancedFn>(enhancedTable[80]));}
    bool ok=Patch(&listTable[9],reinterpret_cast<void*>(closeOriginal.load()),reinterpret_cast<void*>(Close));
    ok=Patch(&listTable[10],reinterpret_cast<void*>(resetOriginal.load()),reinterpret_cast<void*>(Reset))&&ok;
    ok=Patch(&listTable[26],reinterpret_cast<void*>(barrierOriginal.load()),reinterpret_cast<void*>(Barrier))&&ok;
    ok=Patch(&queueTable[10],reinterpret_cast<void*>(executeOriginal.load()),reinterpret_cast<void*>(Execute))&&ok;
    ok=Patch(&listTable[14],reinterpret_cast<void*>(gpuDispatchOriginal.load()),reinterpret_cast<void*>(GpuDispatch))&&ok;
    ok=Patch(&listTable[59],reinterpret_cast<void*>(indirectOriginal.load()),reinterpret_cast<void*>(Indirect))&&ok;
    ok=Patch(&listTable[12],reinterpret_cast<void*>(drawOriginal.load()),reinterpret_cast<void*>(Draw))&&ok;
    ok=Patch(&listTable[13],reinterpret_cast<void*>(drawIndexedOriginal.load()),reinterpret_cast<void*>(DrawIndexed))&&ok;
    ok=Patch(&listTable[16],reinterpret_cast<void*>(copyTextureOriginal.load()),reinterpret_cast<void*>(CopyTexture))&&ok;
    ok=Patch(&listTable[17],reinterpret_cast<void*>(copyResourceOriginal.load()),reinterpret_cast<void*>(CopyResource))&&ok;
    if(enhancedTable)ok=Patch(&enhancedTable[80],reinterpret_cast<void*>(enhancedOriginal.load()),reinterpret_cast<void*>(Enhanced))&&ok;
    {std::lock_guard<std::mutex> lock(guard);Emit(std::string("{\"event\":\"lifecycle_hooks\",\"installed\":")+(ok?"true":"false")+"}");if(!ok)Fail("partial hooks; originals retained");}
    {std::lock_guard<std::mutex> lock(guard);Emit(std::string("{\"event\":\"command_path_hooks\",\"installed\":")+(ok?"true":"false")+",\"list_table\":"+Ptr(listTable)+",\"enhanced_table\":"+Ptr(enhancedTable)+",\"enhanced_supported\":"+(enhancedTable?"true":"false")+",\"scope\":\"synchronous same-thread calls through selected vtables only\"}");}
    hooksReady=ok;return ok;
}
bool DecodeCreate(const ffxCreateContextDescHeader* pointer,Context& c) {
    if(!Read(pointer,c.desc)||c.desc.header.type!=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE)return false;
    const ffxApiHeader* p=c.desc.header.pNext;unsigned seen=0;
    while(p&&seen++<8) {
        ffxApiHeader h{};if(!Read(p,h))return false;
        if(h.type==FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12){ffxCreateBackendDX12Desc b{};if(c.device||!Read(p,b)||!b.device)return false;c.device=static_cast<ID3D12Device*>(b.device);}
        else if(h.type==FFX_API_DESC_TYPE_OVERRIDE_VERSION){ffxOverrideVersion v{};if(c.provider||!Read(p,v))return false;c.provider=v.versionId;}
        else return false;p=h.pNext;
    }
    if(p||!c.device)return false;
    for(auto n:{c.desc.maxRenderSize.width,c.desc.maxRenderSize.height,c.desc.maxUpscaleSize.width,c.desc.maxUpscaleSize.height})if(!n||n>8192)return false;
    c.desc.header.pNext=nullptr;return true;
}
ffxReturnCode_t Create(ffxContext* context,ffxCreateContextDescHeader* desc,const ffxAllocationCallbacks* cb) {
    DWORD incoming=GetLastError();Context parsed;bool decoded=false;
    try{decoded=DecodeCreate(desc,parsed);if(decoded&&!Install(parsed.device.Get()))decoded=false;}catch(...) {decoded=false;}
    SetLastError(incoming);auto status=createOriginal.load()(context,desc,cb);DWORD outgoing=GetLastError();
    try{ffxContext key=nullptr;if(status==0&&Read(context,key)&&key){std::lock_guard<std::mutex> lock(guard);contexts.erase(key);
        if(decoded&&contexts.size()<8){parsed.epoch=++contextEpoch;contexts.emplace(key,parsed);samples=0;
            Emit("{\"event\":\"context_create\",\"context\":"+Ptr(key)+",\"epoch\":"+std::to_string(parsed.epoch)+",\"flags\":"+std::to_string(parsed.desc.flags)+",\"max_render\":["+std::to_string(parsed.desc.maxRenderSize.width)+","+std::to_string(parsed.desc.maxRenderSize.height)+"],\"max_upscale\":["+std::to_string(parsed.desc.maxUpscaleSize.width)+","+std::to_string(parsed.desc.maxUpscaleSize.height)+"],\"requested_provider_id\":"+std::to_string(parsed.provider)+"}");}
        else Emit("{\"event\":\"context_unverified\"}");
    }}catch(...){std::lock_guard<std::mutex> lock(guard);Fail("context registry");}
    SetLastError(outgoing);return status;
}
ffxReturnCode_t Destroy(ffxContext* context,const ffxAllocationCallbacks* cb) {
    DWORD incoming=GetLastError();ffxContext key=nullptr;Read(context,key);
    // Remove knowledge before forwarding; an in-flight/failed destroy never
    // leaves stale context metadata eligible for a later capture.
    try{std::lock_guard<std::mutex> lock(guard);contexts.erase(key);Emit("{\"event\":\"context_destroy_begin\",\"context\":"+Ptr(key)+"}");}catch(...){}
    SetLastError(incoming);return destroyOriginal.load()(context,cb);
}
int64_t Expected(uint32_t s){
    if(s==FFX_API_RESOURCE_STATE_COMPUTE_READ)return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if(s==FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ)return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if(s==FFX_API_RESOURCE_STATE_UNORDERED_ACCESS)return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    return -1;
}
ffxReturnCode_t Dispatch(ffxContext* context,const ffxDispatchDescHeader* pointer) {
    DWORD incoming=GetLastError();bool captured=false;auto epoch=windowEpoch.load(std::memory_order_acquire);
    PathScope path;bool pathSelected=false;uint32_t pathSample=0;uint64_t pathGeneration=0;
    if(SameWindow(epoch))try {
        ffxDispatchDescUpscale a{};ffxContext key=nullptr;
        if(!Read(pointer,a)||a.header.type!=FFX_API_DISPATCH_DESC_TYPE_UPSCALE||a.header.pNext||!Read(context,key))throw std::runtime_error("dispatch ABI");
        auto* list=static_cast<ID3D12GraphicsCommandList*>(a.commandList);if(!list)throw std::runtime_error("null command list");
        ComPtr<ID3D12Device> device;ffx_capture::Check(list->GetDevice(IID_PPV_ARGS(&device)));Install(device.Get());
        std::lock_guard<std::mutex> lock(guard);
        if(SameWindow(epoch)){auto& l=GetList(list);++l.ffxCalls;l.boundary={};}
        if(SameWindow(epoch)&&(previewMode||residentMode)&&samples<18000&&!failed){
            auto& l=GetList(list);auto c=contexts.find(key);
            if(c!=contexts.end()&&hooksReady&&*reinterpret_cast<void***>(list)==listTable&&l.reset&&!l.closed&&
               ffx_capture::Same(c->second.device.Get(),device.Get())){
                ++samples;pathSelected=true;path.list=list;path.window=epoch;pathSample=samples;pathGeneration=l.generation;
                path.roleResources[6]=static_cast<ID3D12Resource*>(a.output.resource);path.declaredStates[6]=a.output.state;
            }
        }
        if(SameWindow(epoch)&&!previewMode&&!residentMode&&samples<sampleLimit&&!collector&&!failed) {
            ++samples;Watch(a);auto& l=GetList(list);auto c=contexts.find(key);
            path.list=list;path.window=epoch;pathSelected=true;pathSample=samples;pathGeneration=l.generation;
            path.sample=samples;path.generation=l.generation;path.details=samples<=4;
            ResourceSnapshot(path,a,true);
            Trace("\"kind\":\"dispatch_begin\",\"list\":"+Ptr(list)+",\"generation\":"+std::to_string(l.generation)+",\"sample\":"+std::to_string(samples));
            bool ready=c!=contexts.end()&&hooksReady&&*reinterpret_cast<void***>(list)==listTable&&l.reset&&!l.closed;
            bool contextKnown=c!=contexts.end();if(contextKnown)ready&=ffx_capture::Same(c->second.device.Get(),device.Get());
            std::ostringstream s;s<<"{\"event\":\"candidate\",\"sample\":"<<samples<<",\"context\":"<<Ptr(key)<<",\"context_known\":"<<(contextKnown?"true":"false")
              <<",\"epoch\":"<<(contextKnown?c->second.epoch:0)<<",\"list\":"<<Ptr(list)<<",\"generation\":"<<l.generation<<",\"reset_observed\":"<<(l.reset?"true":"false")<<",\"resources\":[";
            const char* roles[]={"color","depth","motion","exposure","reactive","transparency","output"};const FfxApiResource* entries[]={&a.color,&a.depth,&a.motionVectors,&a.exposure,&a.reactive,&a.transparencyAndComposition,&a.output};
            bool first=true;
            for(unsigned i=0;i<7;++i)if(entries[i]->resource){auto* r=static_cast<ID3D12Resource*>(entries[i]->resource);auto it=l.states.find(r);int64_t state=it==l.states.end()?-1:it->second.value[0];auto expected=Expected(entries[i]->state);ready&=state>=0&&state==expected;
                if(!first)s<<',';first=false;s<<"{\"role\":\""<<roles[i]<<"\",\"resource\":"<<Ptr(r)<<",\"plane0_recorded_state\":"<<state<<",\"expected_state\":"<<expected<<",\"plane1_recorded_state\":"<<(it==l.states.end()?-1:it->second.value[1])<<'}';}
            s<<"],\"recording_gate_ready\":"<<(ready?"true":"false")<<",\"capture_armed\":"<<(captureWanted?"true":"false")<<'}';Emit(s.str());
            if(ready&&captureWanted) {
                collector=std::make_unique<ffx_capture::Collector>(device.Get(),root/"capture",c->second.desc,c->second.provider,1,192*1024*1024,ffx_capture::DepthPlanePolicy::D32S8DepthOnly,!testMode);
                // Copy callbacks re-enter Barrier. Unlocking is handled below:
                capturedList=list;capturedGeneration=l.generation;capturedEpoch=c->second.epoch;captured=true;
            }
        }
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("candidate validation/tracking");}
    if(captured)try{ffxDispatchDescUpscale a{};if(!Read(pointer,a))throw std::runtime_error("dispatch changed");capturedTicket=collector->Before(a,0);
        std::lock_guard<std::mutex> lock(guard);if(!capturedTicket)Fail("staging budget insufficient");else Emit("{\"event\":\"capture_recorded_inputs\",\"epoch\":"+std::to_string(capturedEpoch)+",\"ticket\":"+std::to_string(capturedTicket)+"}");
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("collector before; no output replacement");}
    // Scope only the original FFX call, excluding our Before/After collector work.
    struct RestorePath {PathScope* previous;~RestorePath(){currentPath=previous;}};
    ffxReturnCode_t status;DWORD outgoing;
    {RestorePath restore{currentPath};currentPath=pathSelected?&path:nullptr;
        SetLastError(incoming);status=dispatchOriginal.load()(context,pointer);outgoing=GetLastError();}
    if(pathSelected&&SameWindow(epoch))try {std::lock_guard<std::mutex> lock(guard);if(SameWindow(epoch)){
        auto& c=path.counts;
        if(path.details||previewMode||residentMode){ffxDispatchDescUpscale a{};if(!Read(pointer,a))throw std::runtime_error("post-call descriptor unreadable");ResourceSnapshot(path,a,false);
            auto& l=GetList(path.list);
            if(status==0&&!captureWanted&&a.commandList==path.list&&a.output.resource==path.roleResources[6]&&
               a.output.state==path.declaredStates[6]&&l.generation==pathGeneration&&l.ffxCalls==1)
                l.boundary={path.roleResources[6],epoch,pathGeneration,pathSample};}
        Emit("{\"event\":\"command_path\",\"sample\":"+std::to_string(pathSample)+",\"list\":"+Ptr(path.list)+",\"generation\":"+std::to_string(pathGeneration)+
             ",\"original_status\":"+std::to_string(status)+",\"gpu_dispatch_calls\":"+std::to_string(c.gpu)+",\"execute_indirect_calls\":"+std::to_string(c.indirect)+
             ",\"legacy_barrier_calls\":"+std::to_string(c.legacy)+",\"legacy_barriers\":"+std::to_string(c.legacyBarriers)+
             ",\"enhanced_barrier_calls\":"+std::to_string(c.enhanced)+",\"enhanced_groups\":"+std::to_string(c.enhancedGroups)+",\"other_interface_calls\":"+std::to_string(c.otherList)+
             ",\"list_table_matches\":"+(*reinterpret_cast<void***>(path.list)==listTable?"true":"false")+",\"capture_authorized\":false}");
    }}catch(...){std::lock_guard<std::mutex> lock(guard);Fail("command path summary");}
    if(SameWindow(epoch))try {ffxDispatchDescUpscale a{};if(Read(pointer,a)) {
        std::lock_guard<std::mutex> lock(guard);auto it=lists.find(static_cast<ID3D12GraphicsCommandList*>(a.commandList));
        if(SameWindow(epoch)&&it!=lists.end())Trace("\"kind\":\"dispatch_end\",\"list\":"+Ptr(a.commandList)+",\"generation\":"+std::to_string(it->second.generation)+",\"original_status\":"+std::to_string(status));
    }}catch(...){std::lock_guard<std::mutex> lock(guard);Fail("dispatch trace");}
    if(captured&&capturedTicket)try {
        bool valid=false;{std::lock_guard<std::mutex> lock(guard);auto& l=GetList(capturedList.Get());ffxDispatchDescUpscale a{};Read(pointer,a);
            auto it=l.states.find(static_cast<ID3D12Resource*>(a.output.resource));valid=status==0&&!failed&&it!=l.states.end()&&it->second.value[0]==8;}
        if(!valid)throw std::runtime_error("post-dispatch output state/status");
        collector->After(capturedTicket,capturedList.Get());std::lock_guard<std::mutex> lock(guard);outputCopied=true;
        Emit("{\"event\":\"capture_recorded_output\",\"original_status\":0}");
    }catch(...){std::lock_guard<std::mutex> lock(guard);Fail("post-dispatch state/status; staged references retained");}
    SetLastError(outgoing);return status;
}
void** Import(const char* name) {
    auto* base=reinterpret_cast<BYTE*>(GetModuleHandleW(nullptr));IMAGE_DOS_HEADER dos{};IMAGE_NT_HEADERS64 nt{};
    if(!Read(base,dos)||dos.e_magic!=IMAGE_DOS_SIGNATURE||dos.e_lfanew<=0||!Read(base+dos.e_lfanew,nt)||nt.Signature!=IMAGE_NT_SIGNATURE)return nullptr;
    auto size=nt.OptionalHeader.SizeOfImage;auto dir=nt.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];if(!dir.VirtualAddress||dir.VirtualAddress>=size||dir.Size>size-dir.VirtualAddress)return nullptr;
    void** found=nullptr;
    for(UINT i=0;i<dir.Size/sizeof(IMAGE_IMPORT_DESCRIPTOR);++i){IMAGE_IMPORT_DESCRIPTOR d{};if(!Read(base+dir.VirtualAddress+i*sizeof(d),d))return nullptr;if(!d.Name)break;if(!d.OriginalFirstThunk)continue;
        for(UINT j=0;j<8192;++j){uint64_t n=uint64_t(d.OriginalFirstThunk)+j*8,v=uint64_t(d.FirstThunk)+j*8;if(n+8>size||v+8>size)return nullptr;IMAGE_THUNK_DATA64 t{};if(!Read(base+n,t))return nullptr;if(!t.u1.AddressOfData)break;if(t.u1.Ordinal&IMAGE_ORDINAL_FLAG64)continue;
            if(t.u1.AddressOfData>size-64)return nullptr;char text[62]{};SIZE_T got=0;if(!ReadProcessMemory(GetCurrentProcess(),base+t.u1.AddressOfData+2,text,61,&got)||got!=61)return nullptr;
            if(!std::strcmp(text,name)){if(found)return nullptr;found=reinterpret_cast<void**>(base+v);}}
    }return found;
}
bool Owner(void* p,uint64_t expected,const char* exportName){HMODULE m=nullptr;return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(p),&m)&&reinterpret_cast<uintptr_t>(m)==expected&&GetProcAddress(m,exportName);}
bool Initialize(const FfxSessionConfigV1& c,PfnFfxCreateContext create,PfnFfxDestroyContext destroy,PfnFfxDispatch dispatch) {
    if(initialized||c.size!=sizeof(c)||(c.version!=1&&c.version!=2&&!(testMode&&c.version==3))||c.sample_limit<1||c.sample_limit>128||c.capture_enabled>1||(c.version>=2&&c.capture_enabled)||c.log_path[1023]||!std::filesystem::path(c.log_path).is_absolute())return false;
    root=std::filesystem::path(c.log_path).parent_path();if(std::filesystem::exists(root/"capture"))return false;
    HANDLE h=CreateFileW(c.log_path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(h==INVALID_HANDLE_VALUE)return false;
    int fd=_open_osfhandle(reinterpret_cast<intptr_t>(h),_O_WRONLY|_O_BINARY);if(fd<0){CloseHandle(h);return false;}logFile=_fdopen(fd,"wb");if(!logFile){_close(fd);return false;}
    HMODULE self=nullptr;if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,reinterpret_cast<LPCWSTR>(Dispatch),&self))return false;
    createOriginal.store(create);destroyOriginal.store(destroy);dispatchOriginal.store(dispatch);sampleLimit=c.sample_limit;captureWanted=c.capture_enabled!=0;
    boundaryTest=testMode&&c.version==3;
    if(boundaryTest)Emit("{\"event\":\"boundary_test_mode\",\"isolated_host_only\":true,\"live_attach_allowed\":false}");
    initialized=true;deadline.store(GetTickCount64()+600000);active.store(c.version!=2,std::memory_order_release);
    if(c.version==2)Emit("{\"event\":\"observation_idle\",\"capture_armed\":false}");return true;
}
}
extern "C" __declspec(dllexport) DWORD WINAPI FfxSession_Attach(void* pointer) {
    try {FfxSessionConfigV1 c{};if(!Read(pointer,c)||c.version==3||!c.expected_dispatch_module||!c.expected_context_module)return 0;
        auto cs=Import("ffxCreateContext"),ds=Import("ffxDestroyContext"),fs=Import("ffxDispatch");if(!cs||!ds||!fs)return 0;
        auto create=*cs,destroy=*ds,dispatch=*fs;
        if(!Owner(create,c.expected_context_module,"FfxObserver_GetStats")||!Owner(destroy,c.expected_context_module,"FfxObserver_GetStats")||!Owner(dispatch,c.expected_dispatch_module,"FfxLive_TestDispatch"))return 0;
        if(!Initialize(c,reinterpret_cast<PfnFfxCreateContext>(create),reinterpret_cast<PfnFfxDestroyContext>(destroy),reinterpret_cast<PfnFfxDispatch>(dispatch)))return 0;
        bool ok=Patch(cs,create,reinterpret_cast<void*>(Create));ok=Patch(ds,destroy,reinterpret_cast<void*>(Destroy))&&ok;ok=Patch(fs,dispatch,reinterpret_cast<void*>(Dispatch))&&ok;
        std::lock_guard<std::mutex> lock(guard);if(!ok)Fail("partial IAT attach");Emit(std::string("{\"event\":\"session_attach\",\"ok\":")+(ok?"true":"false")+"}");return ok?1:0;
    }catch(...){return 0;}
}
extern "C" __declspec(dllexport) DWORD WINAPI FfxSession_TestInitialize(void* pointer) {
    try{FfxSessionTestConfigV1 c{};if(!Read(pointer,c))return 0;HMODULE m=nullptr;auto p=reinterpret_cast<void*>(c.original_dispatch);
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(p),&m))return 0;
        if(GetProcAddress(m,"ffxDispatch")!=reinterpret_cast<FARPROC>(c.original_dispatch)||GetProcAddress(m,"ffxCreateContext")!=reinterpret_cast<FARPROC>(c.original_create)||GetProcAddress(m,"ffxDestroyContext")!=reinterpret_cast<FARPROC>(c.original_destroy))return 0;
        testMode=true;return Initialize(c.common,reinterpret_cast<PfnFfxCreateContext>(c.original_create),reinterpret_cast<PfnFfxDestroyContext>(c.original_destroy),reinterpret_cast<PfnFfxDispatch>(c.original_dispatch))?1:0;
    }catch(...){return 0;}
}
extern "C" __declspec(dllexport) ffxReturnCode_t FfxSession_TestCreate(ffxContext* c,ffxCreateContextDescHeader* d,const ffxAllocationCallbacks* a){return Create(c,d,a);}
extern "C" __declspec(dllexport) ffxReturnCode_t FfxSession_TestDestroy(ffxContext* c,const ffxAllocationCallbacks* a){return Destroy(c,a);}
extern "C" __declspec(dllexport) ffxReturnCode_t FfxSession_TestDispatch(ffxContext* c,const ffxDispatchDescHeader* d){return Dispatch(c,d);}
extern "C" __declspec(dllexport) DWORD WINAPI FfxSession_TestWatch(void* p){try{ffxDispatchDescUpscale a{};if(!testMode||!Read(p,a))return 0;std::lock_guard<std::mutex> lock(guard);if(Tracking())Watch(a);return 1;}catch(...){return 0;}}
extern "C" __declspec(dllexport) DWORD WINAPI FfxSession_Command(void* p) {
    try{FfxSessionCommandV1 c{};if(!Read(p,c)||c.size!=sizeof(c))return 0;std::lock_guard<std::mutex> lock(guard);if(!initialized)return 0;
        if(c.command>=19&&c.command<=21){
            if(c.command==20){residentMode=false;active.store(false,std::memory_order_release);Emit("{\"event\":\"resident_stop\",\"inflight_may_finish\":true}");return 1;}
            if(c.command==21){std::ofstream report(root/L"resident_status.json");report<<"{\"enabled\":"<<(residentMode&&Tracking()&&!failed?"true":"false")<<",\"completed_frames\":"<<residentFrames<<",\"busy\":"<<(residentBusy?"true":"false")<<",\"failed\":"<<(failed?"true":"false")<<",\"independent_tiles\":true,\"dlss5_quality_verified\":false}\n";return report?1:3;}
            if(boundaryTest||residentExchange||Tracking()||previewMode||staticPreview||captureWanted||collector||failed||complete||boundaryLiveArmed||boundaryRecorded||residentPreparing||residentSurface||windowEpoch.load()>=maxObservationWindows)return 0;
            HMODULE self=nullptr;wchar_t path[32768]{};
            if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(Dispatch),&self)||!GetModuleFileNameW(self,path,32768))return 0;
            residentExchange=new ffx_resident::Exchange(std::filesystem::path(path).parent_path()/L"worker_name.txt");
            ++windowEpoch;samples=0;eventCount=0;traceCount=0;resourcePathCount=0;
            for(auto& e:lists){e.second.states.clear();e.second.reset=false;e.second.closed=false;e.second.ffxCalls=0;e.second.boundary={};}
            residentMode=true;residentInstalled.store(true,std::memory_order_release);deadline.store(GetTickCount64()+180000);active.store(true,std::memory_order_release);
            Emit("{\"event\":\"resident_start\",\"frame_limit\":12,\"duration_ms\":180000,\"same_frame_cpu_handoff\":true,\"independent_tiles\":true}");return 1;
        }
        if(residentExchange||residentMode)return 0;
        if(c.command>=15&&c.command<=18){
            if(boundaryTest)return 0;
            if(c.command==16){if(!previewMode&&!staticPreview)return 0;previewMode=false;active.store(false,std::memory_order_release);
                Emit("{\"event\":\"static_preview_stop\",\"resources_retained\":true,\"live_inference\":false}");return 1;}
            if(c.command==17){std::ofstream report(root/L"static_preview_status.json");
                report<<"{\"enabled\":"<<(previewMode&&Tracking()&&!failed?"true":"false")<<",\"recorded_frames\":"<<previewFrames
                      <<",\"network_output\":"<<(previewNetwork?"true":"false")<<",\"failed\":"<<(failed?"true":"false")
                      <<",\"live_inference\":false,\"resources_retained_until_exit\":true}\n";return report?1:3;}
            if(captureWanted||collector||failed||complete||boundaryLiveArmed||boundaryCopy||boundaryRoundTrip||boundaryPatch||boundaryFilter||boundaryNetwork||boundaryPreparing||boundaryRecorded)return 0;
            if(previewMode&&Tracking()){previewNetwork=c.command==15;deadline.store(GetTickCount64()+60000);return 1;}
            if(Tracking()||windowEpoch.load()>=maxObservationWindows||previewPreparing)return 0;
            if(!staticPreview){HMODULE self=nullptr;wchar_t path[32768]{};
                if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(Dispatch),&self)||!GetModuleFileNameW(self,path,32768))return 0;
                auto folder=std::filesystem::path(path).parent_path();
                previewInput=ffx_boundary::StaticPreview::Load(folder/L"preview_input.rgba16f");
                previewOutput=ffx_boundary::StaticPreview::Load(folder/L"preview_output.rgba16f");}
            ++windowEpoch;samples=0;previewFrames=0;eventCount=0;traceCount=0;resourcePathCount=0;
            for(auto& e:lists){e.second.states.clear();e.second.reset=false;e.second.closed=false;e.second.ffxCalls=0;e.second.boundary={};}
            previewMode=true;previewNetwork=c.command==15;deadline.store(GetTickCount64()+60000);active.store(true,std::memory_order_release);
            Emit("{\"event\":\"static_preview_start\",\"timeout_ms\":60000,\"live_inference\":false,\"immutable_offline_pixels\":true}");return 1;
        }
        if(previewMode||staticPreview)return 0; // do not mix preview with one-shot captures
        if(boundaryTest&&c.command!=2)return 0;
        if((boundaryTest&&c.command==2)||(!boundaryTest&&(c.command==6||c.command==8||c.command==10||c.command==12||c.command==14))){
            if(!boundaryTest&&((c.command==10)!=boundaryPatchMode||(c.command==12)!=boundaryFilterMode||(c.command==14)!=boundaryNetworkMode||(c.command!=6)!=boundaryWriteBack))return 0;
            if(failed)return 3;if(boundaryDone)return 1;
            if((!boundaryCopy&&!boundaryRoundTrip&&!boundaryPatch&&!boundaryFilter&&!boundaryNetwork)||!boundarySubmitted)return 2;std::vector<uint8_t> raw,before;
            if(boundaryNetworkMode){if(!boundaryNetwork->Poll(before,raw))return 2;}
            else if(boundaryFilterMode){if(!boundaryFilter->Poll(before,raw))return 2;}
            else if(boundaryPatchMode){if(!boundaryPatch->Poll(before,raw))return 2;}
            else if(boundaryWriteBack){if(!boundaryRoundTrip->Poll(raw))return 2;}else if(!boundaryCopy->Poll(raw))return 2;
            // Job is retired; disk failure cannot invalidate the fence result.
            if(boundaryNetworkMode){delete boundaryNetwork;boundaryNetwork=nullptr;}else if(boundaryFilterMode){delete boundaryFilter;boundaryFilter=nullptr;}else if(boundaryPatchMode){delete boundaryPatch;boundaryPatch=nullptr;}else if(boundaryWriteBack){delete boundaryRoundTrip;boundaryRoundTrip=nullptr;}else{delete boundaryCopy;boundaryCopy=nullptr;}boundaryRawBytes=raw.size();
            auto outputRoot=boundaryTest?root:(root/(boundaryNetworkMode?L"output_network":boundaryFilterMode?L"output_filter":boundaryPatchMode?L"output_patch":boundaryWriteBack?L"output_roundtrip":L"output_boundary"));
            if(!boundaryTest&&!CreateDirectoryW(outputRoot.c_str(),nullptr)){Fail("boundary output directory");return 3;}
            if(boundaryPatchMode||boundaryFilterMode||boundaryNetworkMode){
                HANDLE original=CreateFileW((outputRoot/L"boundary_before.raw").c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
                if(original==INVALID_HANDLE_VALUE){Fail("boundary before file");return 3;}DWORD originalWritten=0;BOOL originalOk=WriteFile(original,before.data(),DWORD(before.size()),&originalWritten,nullptr);CloseHandle(original);
                if(!originalOk||originalWritten!=before.size()){Fail("boundary before write");return 3;}
            }
            HANDLE file=CreateFileW((outputRoot/L"boundary_output.raw").c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
            if(file==INVALID_HANDLE_VALUE){Fail("boundary output file");return 3;}
            DWORD written=0;BOOL ok=WriteFile(file,raw.data(),DWORD(raw.size()),&written,nullptr);CloseHandle(file);
            if(!ok||written!=raw.size()){Fail("boundary output write");return 3;}
            std::ostringstream metadata;metadata<<"{\"width\":"<<boundaryWidth<<",\"height\":"<<boundaryHeight<<",\"format\":"<<boundaryFormat<<",\"raw_bytes\":"<<raw.size()<<",\"fence_completed\":true,\"write_back_performed\":"<<(boundaryWriteBack?"true":"false")<<",\"replacement_pixels_supplied\":"<<((boundaryPatchMode||boundaryFilterMode||boundaryNetworkMode)?"true":"false")<<",\"patch_rect\":"<<(boundaryPatchMode?"[0,0,8,8]":"null")<<",\"filter_kind\":"<<(boundaryNetworkMode?"\"dynamic_residual_cnn_3x3_8\"":boundaryFilterMode?"\"dynamic_residual_5tap\"":"null")<<",\"gpu_residual_filter\":"<<((boundaryFilterMode||boundaryNetworkMode)?"true":"false")<<",\"external_weights\":"<<(boundaryNetworkMode?"true":"false")<<",\"trained_weights\":false,\"game_frame_capture_verified\":"<<(testMode?"false":"true")<<",\"nr_verified\":false}\n";
            auto text=metadata.str();file=CreateFileW((outputRoot/L"metadata.json").c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
            if(file==INVALID_HANDLE_VALUE){Fail("boundary metadata file");return 3;}
            written=0;ok=WriteFile(file,text.data(),DWORD(text.size()),&written,nullptr);CloseHandle(file);
            if(!ok||written!=text.size()){Fail("boundary metadata write");return 3;}
            boundaryDone=true;complete=true;Emit("{\"event\":\"boundary_output_complete\",\"raw_bytes\":"+std::to_string(raw.size())+",\"width\":"+std::to_string(boundaryWidth)+",\"height\":"+std::to_string(boundaryHeight)+",\"format\":"+std::to_string(boundaryFormat)+",\"fence_completed\":true,\"write_back_performed\":"+(boundaryWriteBack?"true":"false")+",\"replacement_pixels_supplied\":"+((boundaryPatchMode||boundaryFilterMode||boundaryNetworkMode)?"true":"false")+",\"patch_rect\":"+(boundaryPatchMode?"[0,0,8,8]":"null")+",\"filter_kind\":"+(boundaryNetworkMode?"\"dynamic_residual_cnn_3x3_8\"":boundaryFilterMode?"\"dynamic_residual_5tap\"":"null")+",\"gpu_residual_filter\":"+((boundaryFilterMode||boundaryNetworkMode)?"true":"false")+",\"external_weights\":"+(boundaryNetworkMode?"true":"false")+",\"trained_weights\":false,\"game_frame_capture_verified\":"+(testMode?"false":"true")+",\"nr_verified\":false}");return 1;
        }
        if(c.command==5||c.command==7||c.command==9||c.command==11||c.command==13){
            const bool network=c.command==13,filter=c.command==11,patch=c.command==9,writeBack=c.command!=5;auto outputRoot=root/(network?L"output_network":filter?L"output_filter":patch?L"output_patch":writeBack?L"output_roundtrip":L"output_boundary");
            if(boundaryTest||captureWanted||collector||failed||complete||Tracking()||boundaryCopy||boundaryRoundTrip||boundaryPatch||boundaryFilter||boundaryNetwork||boundaryPreparing||boundaryRecorded||windowEpoch.load()>=maxObservationWindows||std::filesystem::exists(outputRoot))return 0;
            if(filter||network){HMODULE self=nullptr;wchar_t path[32768]{};if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(Dispatch),&self)||!GetModuleFileNameW(self,path,32768))return 0;
                auto moduleRoot=std::filesystem::path(path).parent_path();auto shaderPath=moduleRoot/(network?L"output_residual_network.dxil":L"output_residual_filter.dxil");std::ifstream stream(shaderPath,std::ios::binary|std::ios::ate);auto shaderSize=stream.tellg();if(!stream||shaderSize<=0||shaderSize>1024*1024)return 0;auto& shader=network?boundaryNetworkShader:boundaryFilterShader;shader.resize(size_t(shaderSize));stream.seekg(0);if(!stream.read(reinterpret_cast<char*>(shader.data()),std::streamsize(shader.size())))return 0;
                if(network)boundaryNetworkWeights=ffx_boundary::LoadResidualModel(moduleRoot/L"ffx_dynamic_residual_v1.weights");}
            active.store(false,std::memory_order_release);++windowEpoch;samples=0;traceCount=0;resourcePathCount=0;eventCount=0;
            for(auto& entry:lists){entry.second.states.clear();entry.second.reset=false;entry.second.closed=false;entry.second.ffxCalls=0;entry.second.boundary={};}
            boundaryWriteBack=writeBack;boundaryPatchMode=patch;boundaryFilterMode=filter;boundaryNetworkMode=network;boundaryLiveArmed=true;deadline.store(GetTickCount64()+60000);active.store(true,std::memory_order_release);
            Emit("{\"event\":\"boundary_output_arm\",\"capture_scope\":\"one output texture\",\"duration_ms\":60000,\"write_back_performed\":"+(writeBack?std::string("true"):std::string("false"))+",\"replacement_pixels_supplied\":"+((patch||filter||network)?std::string("true"):std::string("false"))+",\"gpu_residual_filter\":"+((filter||network)?std::string("true"):std::string("false"))+",\"external_weights\":"+(network?std::string("true"):std::string("false"))+",\"nr_verified\":false}");return 1;
        }
        if(c.command==1){if(collector||failed||complete||boundaryLiveArmed||boundaryCopy||boundaryRoundTrip||boundaryPatch||boundaryFilter||boundaryNetwork||boundaryPreparing||boundaryRecorded||boundaryDone)return 0;samples=0;captureWanted=true;deadline.store(GetTickCount64()+600000);active.store(true,std::memory_order_release);Emit("{\"event\":\"capture_armed\"}");return 1;}
        if(c.command==2){if(failed)return 3;if(complete)return 1;if(!collector||!submitted)return 2;
            if(collector->Poll()==1){complete=true;Emit("{\"event\":\"capture_complete\",\"frames\":1,\"fence_completed\":true,\"nr_verified\":false}");return 1;}return 2;}
        if(c.command==3){
            // Observation cannot cancel an armed/in-flight capture, recover a
            // failed session, or carry stale list-state knowledge into a window.
            if(captureWanted||collector||failed||complete||boundaryLiveArmed||boundaryCopy||boundaryRoundTrip||boundaryPatch||boundaryFilter||boundaryNetwork||boundaryPreparing||boundaryRecorded||boundaryDone||Tracking()||windowEpoch.load()>=maxObservationWindows)return 0;
            active.store(false,std::memory_order_release);++windowEpoch;samples=0;traceCount=0;resourcePathCount=0;eventCount=0;
            for(auto& entry:lists){entry.second.states.clear();entry.second.reset=false;entry.second.closed=false;entry.second.ffxCalls=0;entry.second.boundary={};}
            deadline.store(GetTickCount64()+60000);active.store(true,std::memory_order_release);
            Emit("{\"event\":\"observation_begin\",\"capture_armed\":false,\"duration_ms\":60000}");return 1;
        }
        if(c.command==4){if(captureWanted||collector||failed||complete||boundaryLiveArmed||boundaryCopy||boundaryRoundTrip||boundaryPatch||boundaryFilter||boundaryNetwork||boundaryPreparing||boundaryRecorded)return 0;active.store(false,std::memory_order_release);Emit("{\"event\":\"observation_stop\",\"capture_armed\":false}");return 1;}
        return 0;
    }catch(...){return 3;}
}
