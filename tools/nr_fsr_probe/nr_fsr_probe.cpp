// Standalone engineering-only input file -> real FSR 3.1.0 -> 4K.
// Reuse established DX12 upload/readback helpers; never modify the old harness.
#define wmain unused_legacy_ffx_probe
#include "../ffx_observer/dispatch_probe.cpp"
#undef wmain
#include <dxgi1_4.h>

int wmain(int argc,wchar_t** argv) {
    if(argc!=7){std::fprintf(stderr,"use run.ps1: DLL input width height new-result iterations\n");return 2;}
    try {
        RW=std::stoul(argv[3]);RH=std::stoul(argv[4]);OW=3840;OH=2160;
        const unsigned iterations=std::stoul(argv[6]);
        if(!((RW==1920&&RH==1080)||(RW==2560&&RH==1440)||(RW==3840&&RH==2160))||iterations<1||iterations>12)
            throw std::runtime_error("unreviewed geometry/count");
        const auto input=Read(fs::absolute(argv[2]));
        if(input.size()!=size_t(RW)*RH*8)throw std::runtime_error("input byte count mismatch");
        const auto* ih=reinterpret_cast<const uint16_t*>(input.data());
        for(size_t i=0;i<input.size()/2;++i)if(!HalfFinite(ih[i]))throw std::runtime_error("nonfinite input");
        const fs::path result=fs::absolute(argv[5]);
        if(!fs::create_directory(result))throw std::runtime_error("new result directory required");
        std::ofstream journal(result/"progress.jsonl");
        auto progress=[&](const char* text){journal<<"{\"stage\":\""<<text<<"\"}\n";journal.flush();};
        progress("cpu_input_validated");
        const auto library=fs::absolute(argv[1]);
        HMODULE backend=LoadLibraryExW(library.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!backend)throw std::runtime_error("provider load failed");
        auto create=reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(backend,"ffxCreateContext"));
        auto destroy=reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(backend,"ffxDestroyContext"));
        auto query=reinterpret_cast<PfnFfxQuery>(GetProcAddress(backend,"ffxQuery"));
        auto dispatch=reinterpret_cast<PfnFfxDispatch>(GetProcAddress(backend,"ffxDispatch"));
        if(!create||!destroy||!query||!dispatch)throw std::runtime_error("missing exports");
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
            HR_CHECK(adapter->GetDesc1(&ad));if(!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&ad.VendorId==0x1002&&std::wstring(ad.Description).find(L"RX 9070 XT")!=std::wstring::npos){found=true;break;}
        }if(!found)throw std::runtime_error("RX 9070 XT adapter absent");
        ComPtr<IDXGIAdapter3> a3;HR_CHECK(adapter.As(&a3));UINT64 peak=0;
        auto budget=[&](){DXGI_QUERY_VIDEO_MEMORY_INFO m{};HR_CHECK(a3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&m));
            peak=std::max(peak,m.CurrentUsage);if(m.CurrentUsage>6000000000ull||m.CurrentUsage>m.Budget)throw std::runtime_error("process local video memory budget exceeded");};
        budget();
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));
        uint64_t count=0;ffxQueryDescGetVersions v{};v.header.type=FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        v.createDescType=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;v.device=d.Get();v.outputCount=&count;
        Api(query(nullptr,&v.header),"version count");if(!count||count>32)throw std::runtime_error("version count");
        std::vector<uint64_t> ids(count);std::vector<const char*> names(count);v.versionIds=ids.data();v.versionNames=names.data();
        Api(query(nullptr,&v.header),"versions");if(count>ids.size())throw std::runtime_error("version capacity");
        uint64_t selected=0;unsigned matches=0;
        for(size_t i=0;i<count;++i)if(names[i]&&std::string(names[i])=="3.1.0"){selected=ids[i];++matches;}
        if(matches!=1)throw std::runtime_error("exact FSR3.1.0 provider absent/ambiguous");
        ffxOverrideVersion version{};version.header.type=FFX_API_DESC_TYPE_OVERRIDE_VERSION;version.versionId=selected;
        ffxCreateBackendDX12Desc dx{};dx.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;dx.header.pNext=&version.header;dx.device=d.Get();
        ffxCreateContextDescUpscale desc{};desc.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;desc.header.pNext=&dx.header;
        desc.flags=FFX_UPSCALE_ENABLE_DEBUG_CHECKING;desc.maxRenderSize={RW,RH};desc.maxUpscaleSize={OW,OH};desc.fpMessage=Message;
        ffxContext ctx=nullptr;Api(create(&ctx,&desc.header,nullptr),"create");if(!ctx)throw std::runtime_error("null context");budget();
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> queue;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12Fence> fence;HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event)throw std::runtime_error("event");UINT64 tick=0,frequency=0;
        HR_CHECK(queue->GetTimestampFrequency(&frequency));if(!frequency)throw std::runtime_error("timestamp frequency");
        auto wait=[&](){HR_CHECK(queue->Signal(fence.Get(),++tick));HR_CHECK(fence->SetEventOnCompletion(tick,event));
            if(WaitForSingleObject(event,10000)!=WAIT_OBJECT_0){
                progress("gpu_fence_timeout_resources_retained");
                std::fprintf(stderr,"GPU fence timeout; PID %lu retains resources awaiting manual disposition; no retry or automatic cleanup.\n",GetCurrentProcessId());std::fflush(stderr);
                // Even if the fence later signals, do not resume automatically.
                // Keep this stack/device/resources alive until explicit operator action.
                for(;;)Sleep(1000);
            }
            HR_CHECK(d->GetDeviceRemovedReason());};
        auto color=Tex(d.Get(),RW,RH,DXGI_FORMAT_R16G16B16A16_FLOAT),depth=Tex(d.Get(),RW,RH,DXGI_FORMAT_R32_FLOAT);
        auto motion=Tex(d.Get(),RW,RH,DXGI_FORMAT_R16G16_FLOAT),exposure=Tex(d.Get(),1,1,DXGI_FORMAT_R32_FLOAT);
        auto output=Tex(d.Get(),OW,OH,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
        ComPtr<ID3D12CommandAllocator> alloc;HR_CHECK(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)));
        ComPtr<ID3D12GraphicsCommandList> list;HR_CHECK(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc.Get(),nullptr,IID_PPV_ARGS(&list)));
        std::vector<uint8_t> z(size_t(RW)*RH*4),mv(z.size(),0),exp(4);float plane=.5f,one=1;
        for(size_t i=0;i<z.size();i+=4)std::memcpy(z.data()+i,&plane,4);std::memcpy(exp.data(),&one,4);
        auto uc=UploadTex(d.Get(),list.Get(),color.Get(),input),uz=UploadTex(d.Get(),list.Get(),depth.Get(),z);
        auto um=UploadTex(d.Get(),list.Get(),motion.Get(),mv),ue=UploadTex(d.Get(),list.Get(),exposure.Get(),exp);
        HR_CHECK(list->Close());ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);wait();
        uc.Reset();uz.Reset();um.Reset();ue.Reset();budget();progress("uploaded");
        D3D12_QUERY_HEAP_DESC qh{};qh.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;qh.Count=2;
        ComPtr<ID3D12QueryHeap> queries;HR_CHECK(d->CreateQueryHeap(&qh,IID_PPV_ARGS(&queries)));
        auto readback=[&](UINT64 bytes){D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_READBACK;auto bd=BufferDesc(bytes);ComPtr<ID3D12Resource> r;
            HR_CHECK(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&bd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&r)));return r;};
        auto times=readback(16);const auto footprint=Footprint(d.Get(),output.Get());auto pixels=readback(footprint.bytes);
        std::vector<double> ms;std::vector<uint8_t> previous;bool repeat=true;
        for(unsigned run=0;run<iterations+1;++run){
            budget();HR_CHECK(alloc->Reset());HR_CHECK(list->Reset(alloc.Get(),nullptr));
            // Poison output before every dispatch, outside the measured interval.
            std::vector<uint8_t> poison(size_t(OW)*OH*8);auto* ph=reinterpret_cast<uint16_t*>(poison.data());std::fill(ph,ph+poison.size()/2,uint16_t(0x7e00));
            Recorder::Transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
            auto up=UploadTex(d.Get(),list.Get(),output.Get(),poison);
            Recorder::Transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ffxDispatchDescUpscale a{};a.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;a.commandList=list.Get();
            a.color=ffxApiGetResourceDX12(color.Get());a.depth=ffxApiGetResourceDX12(depth.Get());a.motionVectors=ffxApiGetResourceDX12(motion.Get());
            a.exposure=ffxApiGetResourceDX12(exposure.Get());a.output=ffxApiGetResourceDX12(output.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            a.renderSize={RW,RH};a.upscaleSize={OW,OH};a.motionVectorScale={float(RW),float(RH)};a.reset=true;
            a.preExposure=1;a.frameTimeDelta=16.666667f;a.cameraNear=.1f;a.cameraFar=1000;a.cameraFovAngleVertical=1.04719755f;a.viewSpaceToMetersFactor=1;
            list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);Api(dispatch(&ctx,&a.header),"dispatch");
            list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);list->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,times.Get(),0);
            Recorder::Transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            auto src=Recorder::TextureLocation(output.Get()),dst=Linear(pixels.Get(),footprint);list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
            Recorder::Transition(list.Get(),output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            HR_CHECK(list->Close());queue->ExecuteCommandLists(1,lists);wait();budget();
            auto timing=ReadGpu(times.Get(),16);UINT64 stamps[2]{};std::memcpy(stamps,timing.data(),16);
            if(stamps[1]<=stamps[0])throw std::runtime_error("invalid timestamps");ms.push_back(double(stamps[1]-stamps[0])*1000/double(frequency));
            auto out=Unpack(pixels.Get(),footprint);const auto* h=reinterpret_cast<const uint16_t*>(out.data());
            for(size_t i=0;i<out.size()/2;++i)if(!HalfFinite(h[i]))throw std::runtime_error("nonfinite or unwritten output");
            if(!previous.empty()&&out!=previous)repeat=false;previous=std::move(out);
            journal<<"{\"run\":"<<run<<",\"warmup\":"<<(run==0?"true":"false")<<",\"fsr_gpu_ms\":"<<ms.back()<<",\"local_usage_peak\":"<<peak<<"}\n";journal.flush();
        }
        uint64_t errors=0;for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t> b(n);auto* m=reinterpret_cast<D3D12_MESSAGE*>(b.data());HR_CHECK(info->GetMessage(i,m,&n));if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)++errors;}
        Api(destroy(&ctx,nullptr),"destroy");CloseHandle(event);Write(result/"output.rgba16f",previous);
        std::ofstream report(result/"report.json");report<<std::setprecision(12)<<"{\"scope\":\"SYNTHETIC_STATIC_ENGINEERING_NOT_GAME_QUALITY\",\"provider\":\"3.1.0\",\"provider_id\":"<<selected
          <<",\"adapter\":\"RX 9070 XT\",\"adapter_device_id\":"<<ad.DeviceId<<",\"input_size\":["<<RW<<','<<RH<<"],\"output_size\":[3840,2160],\"output_format\":\"RGBA16F_LE_TIGHT\",\"reset_every_frame\":true,\"persistent_context\":true,\"finite\":true,\"repeat_exact\":"<<(repeat?"true":"false")
          <<",\"d3d12_errors\":"<<errors<<",\"process_local_usage_peak_sampled\":"<<peak<<",\"gpu_ms_including_warmup\":[";
        for(size_t i=0;i<ms.size();++i){if(i)report<<',';report<<ms[i];}report<<"],\"pass\":"<<(!errors&&repeat?"true":"false")<<"}\n";
        return !errors&&repeat?0:3;
    }catch(const std::exception& e){std::fprintf(stderr,"nr_fsr_probe: %s\n",e.what());return 1;}
}
