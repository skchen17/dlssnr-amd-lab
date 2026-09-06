#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../../tools/output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "../../tools/ffx_observer/ffx_observer.h"
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include <limits>

extern "C" __declspec(dllimport) void Fixture_SetCallback(void (*)(const ffxDispatchDescUpscale*));
extern "C" __declspec(dllimport) uint64_t Fixture_Count(unsigned);
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static const ffxDispatchDescUpscale* expectedDescriptor = nullptr;
void GpuCopy(const ffxDispatchDescUpscale* d) {
    Require(d==expectedDescriptor,"observer changed original descriptor pointer");
    auto* list=static_cast<ID3D12GraphicsCommandList*>(d->commandList);
    auto* input=static_cast<ID3D12Resource*>(d->color.resource);
    auto* output=static_cast<ID3D12Resource*>(d->output.resource);
    Recorder::Transition(list,input,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
    Recorder::Transition(list,output,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(output,input);
    Recorder::Transition(list,input,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
    Recorder::Transition(list,output,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
}
int main(int argc,char** argv) { try {
    if (argc!=3) throw std::invalid_argument("usage: ffx_observer_selftest <observer.dll> <result_directory>");
    fs::path directory=fs::absolute(argv[2]); fs::create_directories(directory);
    HMODULE observer=LoadLibraryW(fs::absolute(argv[1]).wstring().c_str()); Require(observer!=nullptr,"load observer");
    auto attach=reinterpret_cast<FfxObserverAttach_t>(GetProcAddress(observer,"FfxObserver_Attach"));
    auto detach=reinterpret_cast<FfxObserverDetach_t>(GetProcAddress(observer,"FfxObserver_Detach"));
    auto stats=reinterpret_cast<FfxObserverStats_t>(GetProcAddress(observer,"FfxObserver_GetStats"));
    Require(attach && detach && stats,"observer exports");
    auto logfile=(directory/"decoded.jsonl").wstring();
    FfxObserverConfigV1 cfg{sizeof(cfg),1,GetModuleHandleW(nullptr),logfile.c_str(),32};
    Require(attach(&cfg)==1,"attach five direct imports"); Require(attach(&cfg)==0,"duplicate attach rejected");
    ffxContext context=nullptr;
    ffxCreateContextDescUpscale create{}; create.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags=FFX_UPSCALE_ENABLE_DEPTH_INVERTED; create.maxRenderSize={640,360}; create.maxUpscaleSize={640,360};
    auto beforeCreate=create;
    Require(ffxCreateContext(&context,&create.header,nullptr)==0 && context==reinterpret_cast<void*>(0x12345678),"create forwarding");
    Require(!memcmp(&create,&beforeCreate,sizeof(create)),"create descriptor modified");
    ffxApiHeader query{0x5555,nullptr}; Require(ffxQuery(&context,&query)==31 && query.type==0x11223344,"query output and status preserved");
    ffxApiHeader cycle{0x8888,nullptr}; cycle.pNext=&cycle;
    SetLastError(0xCAFE);
    Require(ffxConfigure(&context,&cycle)==23 && cycle.pNext==&cycle,"configure cyclic extension forwarded");

    ComPtr<ID3D12Debug> debug; bool debugEnabled=SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if(debugEnabled) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory; HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 ad{}; bool found=false;
    for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
        HR_CHECK(adapter->GetDesc1(&ad)); if(!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&ad.VendorId==0x1002){found=true;break;}
    }
    Require(found,"AMD adapter required"); ComPtr<ID3D12Device> device;
    HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)));
    ComPtr<ID3D12InfoQueue> info; device.As(&info);
    auto input=Texture(device.Get()),output=Texture(device.Get());
    std::vector<uint8_t> pixels(static_cast<size_t>(kSurfaceBytes), 0);
    for(size_t i=0;i<pixels.size();++i) pixels[i]=uint8_t((i*13+i/77)%251);
    auto upload=Upload(device.Get(),pixels), readback=Readback(device.Get(),pixels.size());
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; HR_CHECK(device->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator; HR_CHECK(device->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list; HR_CHECK(device->CreateCommandList(0,qd.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
    Recorder::Transition(list.Get(),input.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
    auto src=Recorder::LinearLocation(upload.Get()), dst=Recorder::TextureLocation(input.Get());
    list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    Recorder::Transition(list.Get(),input.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
    ffxDispatchDescUpscale dispatch{}; dispatch.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.commandList=list.Get(); dispatch.color.resource=input.Get(); dispatch.output.resource=output.Get();
    dispatch.color.description={FFX_API_RESOURCE_TYPE_TEXTURE2D,FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,640,360,1,1,0,0};
    dispatch.output.description=dispatch.color.description;
    dispatch.color.state=dispatch.output.state=FFX_API_RESOURCE_STATE_COMMON;
    dispatch.renderSize={640,360}; dispatch.upscaleSize={640,360}; dispatch.jitterOffset={.25f,-.125f};
    dispatch.motionVectorScale={640,360}; dispatch.preExposure=1; dispatch.reset=true; dispatch.frameTimeDelta=16.5f;
    dispatch.cameraNear=.1f; dispatch.cameraFar=1000; dispatch.cameraFovAngleVertical=1.2f; dispatch.viewSpaceToMetersFactor=1;
    auto originalDispatch=dispatch; expectedDescriptor=&dispatch; Fixture_SetCallback(GpuCopy);
    Require(ffxDispatch(&context,&dispatch.header)==37,"dispatch status not preserved");
    Require(GetLastError()==0xFACE,"backend last-error state not preserved");
    Require(!memcmp(&dispatch,&originalDispatch,sizeof(dispatch)),"dispatch descriptor modified");
    Fixture_SetCallback(nullptr);
    Recorder::Transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
    src=Recorder::TextureLocation(output.Get()); dst=Recorder::LinearLocation(readback.Get());
    list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    HR_CHECK(list->Close()); ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
    ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(),1)); HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr); Require(event!=nullptr,"event");
    HR_CHECK(fence->SetEventOnCompletion(1,event)); auto wait=WaitForSingleObject(event,60000); CloseHandle(event);
    Require(wait==WAIT_OBJECT_0,"GPU timeout"); HR_CHECK(device->GetDeviceRemovedReason());
    auto actual=ReadGpu(readback.Get(),pixels.size()); Require(actual==pixels,"observer changed actual GPU output");
    Write(directory/"gpu_output.raw",actual.data(),actual.size());

    ffxApiHeader unknown{0xABCDEF,nullptr}; Require(ffxDispatch(&context,&unknown)==41,"unknown dispatch forwarding");
    Require(ffxDispatch(&context,nullptr)==47,"null forwarding");
    auto* pages=static_cast<uint8_t*>(VirtualAlloc(nullptr,8192,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE)); Require(pages!=nullptr,"guard allocation");
    auto* shortDesc=reinterpret_cast<ffxApiHeader*>(pages+4096-sizeof(ffxApiHeader)); *shortDesc={FFX_API_DISPATCH_DESC_TYPE_UPSCALE,nullptr};
    DWORD old; Require(VirtualProtect(pages+4096,4096,PAGE_NOACCESS,&old)!=0,"guard protection");
    Require(ffxDispatch(&context,shortDesc)==43,"short body forwarding"); VirtualFree(pages,0,MEM_RELEASE);
    dispatch.commandList=nullptr; dispatch.cameraFar=std::numeric_limits<float>::infinity();
    Require(ffxDispatch(&context,&dispatch.header)==37,"nonfinite metadata forwarding");
    Require(ffxDestroyContext(&context,nullptr)==29 && !context,"destroy forwarding");
    FfxObserverStatsV1 first{sizeof(first)}; Require(stats(&first) && first.forwarded_calls==9 && first.emitted_events==9 && first.patched_imports==5,"decoded counters");
    Require(detach()==1,"quiescent detach");
    Require(ffxDispatch(&context,&unknown)==41,"original call after detach");
    FfxObserverStatsV1 after{sizeof(after)}; Require(stats(&after)&&after.forwarded_calls==9&&after.patched_imports==0,"no observation after detach");

    auto headerLog=(directory/"header_only.jsonl").wstring(); cfg.profile=0; cfg.event_limit=2; cfg.log_path=headerLog.c_str();
    Require(attach(&cfg)==1,"header-only attach");
    for(unsigned i=0;i<10;++i) Require(ffxDispatch(&context,&dispatch.header)==37,"budget must not stop rendering");
    FfxObserverStatsV1 bounded{sizeof(bounded)}; Require(stats(&bounded)&&bounded.forwarded_calls==10&&bounded.emitted_events==2&&bounded.dropped_events==8,"bounded log");
    Require(detach()==1,"header-only detach");
    uint64_t debugErrors=0;
    if(info) for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t>b(n);auto*m=reinterpret_cast<D3D12_MESSAGE*>(b.data());HR_CHECK(info->GetMessage(i,m,&n));if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR){++debugErrors;std::fprintf(stderr,"%s\n",m->pDescription);}}
    Require(!debugErrors,"D3D12 debug errors");
    Require(Fixture_Count(0)==1 && Fixture_Count(1)==1 && Fixture_Count(2)==1 && Fixture_Count(3)==1 && Fixture_Count(4)==16,"exactly-once backend calls");
    std::ostringstream report; report << "{\"status\":\"PASS\",\"synthetic_backend\":true,\"direct_iat_imports\":5,\"decoded_events\":9,\"bounded_events\":2,\"dropped_events\":8,\"gpu_bytes_equal\":" << pixels.size()
        << ",\"debug_layer_enabled\":" << (debugEnabled?"true":"false") << ",\"debug_errors\":0,\"game_launched\":false,\"gpu_frame_capture_ready\":false,\"game_abi_verified\":false}\n";
    auto text=report.str(); Write(directory/"manifest.json",text.data(),text.size()); std::printf("%s",text.c_str()); return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;} }
