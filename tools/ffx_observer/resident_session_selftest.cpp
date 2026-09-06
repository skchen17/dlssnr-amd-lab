#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"
#include "capture_session.h"

static void Wait(ID3D12Device* d,ID3D12CommandQueue* q){
    ComPtr<ID3D12Fence> fence;HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));HR_CHECK(q->Signal(fence.Get(),1));HANDLE e=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if(!e||FAILED(fence->SetEventOnCompletion(1,e))||WaitForSingleObject(e,2000)!=WAIT_OBJECT_0)TerminateProcess(GetCurrentProcess(),3);CloseHandle(e);HR_CHECK(d->GetDeviceRemovedReason());
}
int wmain(int argc,wchar_t** argv){
    if(argc!=3&&argc!=4)return 2;const bool expectFailure=argc==4&&!wcscmp(argv[3],L"--expect-worker-failure");if(argc==4&&!expectFailure)return 2;bool amd=!wcscmp(argv[2],L"--amd");if(!amd&&wcscmp(argv[2],L"--warp"))return 2;
    auto root=fs::absolute(argv[1]);if(fs::exists(root))return 2;fs::create_directories(root);
    try{
        wchar_t exe[32768]{};GetModuleFileNameW(nullptr,exe,32768);auto bin=fs::path(exe).parent_path();
        auto fixture=LoadLibraryExW((bin/L"ffx_filter_session_fixture.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        auto session=LoadLibraryExW((bin/L"ffx_capture_session.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);if(!fixture||!session)throw std::runtime_error("modules");
        using Control=DWORD(WINAPI*)(void*);using Create=ffxReturnCode_t(*)(ffxContext*,ffxCreateContextDescHeader*,const ffxAllocationCallbacks*);using Dispatch=ffxReturnCode_t(*)(ffxContext*,const ffxDispatchDescHeader*);
        auto init=reinterpret_cast<Control>(GetProcAddress(session,"FfxSession_TestInitialize")),command=reinterpret_cast<Control>(GetProcAddress(session,"FfxSession_Command"));
        auto create=reinterpret_cast<Create>(GetProcAddress(session,"FfxSession_TestCreate"));auto dispatch=reinterpret_cast<Dispatch>(GetProcAddress(session,"FfxSession_TestDispatch"));
        if(!init||!command||!create||!dispatch)throw std::runtime_error("exports");
        FfxSessionTestConfigV1 cfg{};cfg.common.size=sizeof(cfg.common);cfg.common.version=2;cfg.common.sample_limit=4;wcscpy_s(cfg.common.log_path,(root/L"session.jsonl").c_str());
        cfg.original_create=reinterpret_cast<uintptr_t>(GetProcAddress(fixture,"ffxCreateContext"));cfg.original_destroy=reinterpret_cast<uintptr_t>(GetProcAddress(fixture,"ffxDestroyContext"));cfg.original_dispatch=reinterpret_cast<uintptr_t>(GetProcAddress(fixture,"ffxDispatch"));if(init(&cfg)!=1)throw std::runtime_error("init");
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};if(amd){bool found=false;for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()){adapter->GetDesc1(&ad);if(ad.VendorId==0x1002&&!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){found=true;break;}}if(!found)throw std::runtime_error("AMD missing");}else{HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));adapter->GetDesc1(&ad);}
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));
        ffxCreateBackendDX12Desc backend{};backend.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;backend.device=d.Get();ffxCreateContextDescUpscale cd{};cd.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;cd.header.pNext=&backend.header;cd.maxRenderSize={320,180};cd.maxUpscaleSize={640,360};ffxContext context=nullptr;if(create(&context,&cd.header,nullptr))throw std::runtime_error("create");
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> q;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)));
        D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;td.Width=640;td.Height=360;td.DepthOrArraySize=1;td.MipLevels=1;td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;ComPtr<ID3D12Resource> target;HR_CHECK(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&target)));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows;UINT64 rowBytes,total;d->GetCopyableFootprints(&td,0,1,0,&fp,&rows,&rowBytes,&total);
        std::vector<uint8_t> baseline(640*360*8),packed(size_t(total),0);for(size_t i=0;i<baseline.size();i+=2){uint16_t h=uint16_t(0x3000+(i/2)%1024);std::memcpy(baseline.data()+i,&h,2);}for(UINT y=0;y<360;++y)std::memcpy(packed.data()+fp.Offset+y*fp.Footprint.RowPitch,baseline.data()+y*rowBytes,size_t(rowBytes));
        auto upload=Upload(d.Get(),packed),down=Readback(d.Get(),total);
        ComPtr<ID3D12CommandAllocator> pa,ca;ComPtr<ID3D12GraphicsCommandList> producer,consumer;HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&pa)));HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&ca)));HR_CHECK(d->CreateCommandList(0,qd.Type,pa.Get(),nullptr,IID_PPV_ARGS(&producer)));HR_CHECK(d->CreateCommandList(0,qd.Type,ca.Get(),nullptr,IID_PPV_ARGS(&consumer)));HR_CHECK(producer->Close());HR_CHECK(consumer->Close());
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        bool exact=true,changed=false;
        for(UINT run=0;run<3;++run){if(run!=1){FfxSessionCommandV1 c{sizeof(c),run==0?19u:20u};if(command(&c)!=1)throw std::runtime_error("command");}
            HR_CHECK(pa->Reset());HR_CHECK(ca->Reset());HR_CHECK(producer->Reset(pa.Get(),nullptr));HR_CHECK(consumer->Reset(ca.Get(),nullptr));
            if(run)Recorder::Transition(producer.Get(),target.Get(),readable,D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION src{},dst=Recorder::TextureLocation(target.Get());src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;producer->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(producer.Get(),target.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ffxDispatchDescUpscale args{};args.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;args.commandList=producer.Get();args.output=ffxApiGetResourceDX12(target.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);args.renderSize={320,180};args.upscaleSize={640,360};if(dispatch(&context,&args.header))throw std::runtime_error("dispatch");Recorder::Transition(producer.Get(),target.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,readable);HR_CHECK(producer->Close());
            Recorder::Transition(consumer.Get(),target.Get(),readable,D3D12_RESOURCE_STATE_COPY_SOURCE);src=Recorder::TextureLocation(target.Get());dst.pResource=down.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;consumer->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(consumer.Get(),target.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,readable);HR_CHECK(consumer->Close());
            ID3D12CommandList* batch[]={producer.Get(),consumer.Get()};q->ExecuteCommandLists(2,batch);Wait(d.Get(),q.Get());
            auto data=ReadGpu(down.Get(),size_t(total));std::vector<uint8_t> actual(baseline.size());for(UINT y=0;y<360;++y)std::memcpy(actual.data()+y*rowBytes,data.data()+fp.Offset+y*fp.Footprint.RowPitch,size_t(rowBytes));
            if(run<2&&!expectFailure){auto returned=Read(root/L"resident_frames"/(L"output"+std::to_wstring(run+1)+L".raw"));auto input=Read(root/L"resident_frames"/(L"input"+std::to_wstring(run+1)+L".raw"));exact&=actual==returned&&input==baseline;changed|=actual!=baseline;}else exact&=actual==baseline;
            Write(root/(L"consumer"+std::to_wstring(run)+L".raw"),actual.data(),actual.size());
        }
        FfxSessionCommandV1 status{sizeof(status),21};if(command(&status)!=1)throw std::runtime_error("status");
        UINT errors=0,warnings=0;for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t> b(n);auto* m=reinterpret_cast<D3D12_MESSAGE*>(b.data());info->GetMessage(i,m,&n);errors+=m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR;warnings+=m->Severity==D3D12_MESSAGE_SEVERITY_WARNING;if(m->Severity<=D3D12_MESSAGE_SEVERITY_WARNING)std::fprintf(stderr,"%s\n",m->pDescription);}
        bool pass=exact&&(changed||expectFailure)&&!errors&&!warnings;std::ostringstream report;report<<"{\"status\":\""<<(pass?"PASS":"FAIL")<<"\",\"adapter_vendor\":"<<ad.VendorId<<",\"processed_frames\":"<<(expectFailure?0:2)<<",\"expected_worker_failure\":"<<(expectFailure?"true":"false")<<",\"stop_next_frame_exact\":"<<(exact?"true":"false")<<",\"same_frame_consumer_exact\":"<<(exact?"true":"false")<<",\"nonidentity\":"<<(changed?"true":"false")<<",\"d3d12_errors\":"<<errors<<",\"d3d12_warnings\":"<<warnings<<"}\n";auto text=report.str();Write(root/L"summary.json",text.data(),text.size());std::printf("%s",text.c_str());return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"resident session: %s\n",e.what());return 1;}
}
