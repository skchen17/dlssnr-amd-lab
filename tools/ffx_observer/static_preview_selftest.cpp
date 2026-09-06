#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"
#include "capture_session.h"
#include "output_static_preview.h"

static void Wait(ID3D12Device* device,ID3D12CommandQueue* queue){
    ComPtr<ID3D12Fence> fence;HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(),1));HANDLE done=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if(!done||FAILED(fence->SetEventOnCompletion(1,done))||WaitForSingleObject(done,10000)!=WAIT_OBJECT_0)
        TerminateProcess(GetCurrentProcess(),3); // isolated test only; never a game process
    CloseHandle(done);HR_CHECK(device->GetDeviceRemovedReason());
}

int wmain(int argc,wchar_t** argv){
    if(argc!=3)return 2;const bool amd=!wcscmp(argv[2],L"--amd");if(!amd&&wcscmp(argv[2],L"--warp"))return 2;
    auto root=fs::absolute(argv[1]);if(fs::exists(root))return 2;fs::create_directories(root);
    try{
        wchar_t exe[32768]{};GetModuleFileNameW(nullptr,exe,32768);auto bin=fs::path(exe).parent_path();
        auto fixture=LoadLibraryExW((bin/L"ffx_filter_session_fixture.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        auto session=LoadLibraryExW((bin/L"ffx_capture_session.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!fixture||!session)throw std::runtime_error("modules");
        using CreateFn=ffxReturnCode_t(*)(ffxContext*,ffxCreateContextDescHeader*,const ffxAllocationCallbacks*);
        using DispatchFn=ffxReturnCode_t(*)(ffxContext*,const ffxDispatchDescHeader*);
        using ControlFn=DWORD(WINAPI*)(void*);
        auto init=reinterpret_cast<ControlFn>(GetProcAddress(session,"FfxSession_TestInitialize"));
        auto command=reinterpret_cast<ControlFn>(GetProcAddress(session,"FfxSession_Command"));
        auto create=reinterpret_cast<CreateFn>(GetProcAddress(session,"FfxSession_TestCreate"));
        auto dispatch=reinterpret_cast<DispatchFn>(GetProcAddress(session,"FfxSession_TestDispatch"));
        if(!init||!command||!create||!dispatch)throw std::runtime_error("exports");
        FfxSessionTestConfigV1 cfg{};cfg.common.size=sizeof(cfg.common);cfg.common.version=2;cfg.common.sample_limit=4;
        wcscpy_s(cfg.common.log_path,(root/L"session.jsonl").c_str());
        cfg.original_create=reinterpret_cast<uintptr_t>(GetProcAddress(fixture,"ffxCreateContext"));
        cfg.original_destroy=reinterpret_cast<uintptr_t>(GetProcAddress(fixture,"ffxDestroyContext"));
        cfg.original_dispatch=reinterpret_cast<uintptr_t>(GetProcAddress(fixture,"ffxDispatch"));
        if(init(&cfg)!=1)throw std::runtime_error("init");
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};
        if(amd){bool found=false;for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()){
            adapter->GetDesc1(&ad);if(ad.VendorId==0x1002&&!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){found=true;break;}}
            if(!found)throw std::runtime_error("AMD absent");}
        else{HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));adapter->GetDesc1(&ad);}
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));
        ffxCreateBackendDX12Desc backend{};backend.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;backend.device=d.Get();
        ffxCreateContextDescUpscale cd{};cd.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;cd.header.pNext=&backend.header;
        cd.maxRenderSize={480,270};cd.maxUpscaleSize={960,540};ffxContext context=nullptr;
        if(create(&context,&cd.header,nullptr))throw std::runtime_error("create");
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> q;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)));
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC td{};
        td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;td.Width=960;td.Height=540;td.DepthOrArraySize=1;
        td.MipLevels=1;td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> target;HR_CHECK(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&target)));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows;UINT64 rowBytes,total;d->GetCopyableFootprints(&td,0,1,0,&fp,&rows,&rowBytes,&total);
        std::vector<uint8_t> baseline(960*540*8),packed(size_t(total),0);
        for(size_t i=0;i<baseline.size();i+=2){uint16_t h=uint16_t(0x3000+(i/2)%1024);std::memcpy(baseline.data()+i,&h,2);}
        for(UINT y=0;y<540;++y)std::memcpy(packed.data()+fp.Offset+y*fp.Footprint.RowPitch,baseline.data()+y*rowBytes,size_t(rowBytes));
        auto upload=Upload(d.Get(),packed),down=Readback(d.Get(),total);
        auto input=ffx_boundary::StaticPreview::Load(bin/L"preview_input.rgba16f");
        auto output=ffx_boundary::StaticPreview::Load(bin/L"preview_output.rgba16f");
        ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> list;
        HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&allocator)));HR_CHECK(d->CreateCommandList(0,qd.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));HR_CHECK(list->Close());
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        bool exact=true;const UINT commands[]={0,15,0,0,18,16,0,15,16};int mode=0;
        for(UINT run=0;run<9;++run){
            if(commands[run]){FfxSessionCommandV1 c{sizeof(c),commands[run]};if(command(&c)!=1)throw std::runtime_error("preview command");mode=commands[run]==16?0:commands[run]==15?2:1;}
            HR_CHECK(allocator->Reset());HR_CHECK(list->Reset(allocator.Get(),nullptr));
            if(run)Recorder::Transition(list.Get(),target.Get(),readable,D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION src{},dst=Recorder::TextureLocation(target.Get());src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;
            list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(list.Get(),target.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ffxDispatchDescUpscale args{};args.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;args.commandList=list.Get();
            args.output=ffxApiGetResourceDX12(target.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);args.renderSize={480,270};args.upscaleSize={960,540};
            if(dispatch(&context,&args.header))throw std::runtime_error("dispatch");
            Recorder::Transition(list.Get(),target.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,readable);
            Recorder::Transition(list.Get(),target.Get(),readable,D3D12_RESOURCE_STATE_COPY_SOURCE);
            src=Recorder::TextureLocation(target.Get());dst.pResource=down.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;
            list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(list.Get(),target.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,readable);
            HR_CHECK(list->Close());ID3D12CommandList* lists[]={list.Get()};q->ExecuteCommandLists(1,lists);Wait(d.Get(),q.Get());
            auto data=ReadGpu(down.Get(),size_t(total));std::vector<uint8_t> actual(baseline.size()),expected=baseline;
            for(UINT y=0;y<540;++y)std::memcpy(actual.data()+y*rowBytes,data.data()+fp.Offset+y*fp.Footprint.RowPitch,size_t(rowBytes));
            if(mode){auto& panel=mode==2?output:input;for(UINT y=0;y<384;++y)std::memcpy(expected.data()+((y+78)*960+158)*8,panel.data()+y*644*8,644*8);}
            exact&=actual==expected;if(actual!=expected)std::fprintf(stderr,"mismatch run %u mode %d\n",run,mode);
            Write(root/(std::to_string(run)+".raw"),actual.data(),actual.size());
        }
        FfxSessionCommandV1 status{sizeof(status),17};if(command(&status)!=1)throw std::runtime_error("status");
        UINT errors=0,warnings=0;for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t> bytes(n);auto* msg=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());info->GetMessage(i,msg,&n);
            errors+=msg->Severity<=D3D12_MESSAGE_SEVERITY_ERROR;warnings+=msg->Severity==D3D12_MESSAGE_SEVERITY_WARNING;
            if(msg->Severity<=D3D12_MESSAGE_SEVERITY_WARNING)std::fprintf(stderr,"%s\n",msg->pDescription);}
        const bool pass=exact&&!errors&&!warnings;std::ostringstream result;
        result<<"{\"status\":\""<<(pass?"STATIC_PREVIEW_SESSION_PASS":"FAIL")<<"\",\"adapter_vendor\":"<<ad.VendorId<<",\"cases\":9,\"all_pixels_exact\":"<<(exact?"true":"false")<<",\"stop_restores_next_frame\":"<<(exact?"true":"false")<<",\"d3d12_errors\":"<<errors<<",\"d3d12_warnings\":"<<warnings<<",\"live_inference\":false}\n";
        auto text=result.str();Write(root/"summary.json",text.data(),text.size());std::printf("%s",text.c_str());return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"preview selftest: %s\n",e.what());return 1;}
}
