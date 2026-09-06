// Reuse the isolated texture producers, not a provider/game entry point.
#define wmain unused_dispatch_probe_main
#include "dispatch_probe.cpp"
#undef wmain

int wmain(int argc,wchar_t** argv) {
    if(argc!=2)return 2;auto root=fs::absolute(argv[1]);
    try {
        if(fs::exists(root)||!fs::create_directories(root))throw std::runtime_error("new output required");
        RW=OW=128;RH=OH=72;
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
            HR_CHECK(adapter->GetDesc1(&ad));if(ad.VendorId==0x1002&&!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){found=true;break;}
        }if(!found)throw std::runtime_error("AMD adapter required");
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> queue;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12CommandAllocator> allocator;HR_CHECK(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> list;HR_CHECK(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
        auto color=Tex(d.Get(),RW,RH,DXGI_FORMAT_R16G16B16A16_FLOAT),mv=Tex(d.Get(),RW,RH,DXGI_FORMAT_R16G16_FLOAT),out=Tex(d.Get(),OW,OH,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
        auto input=Color(0);auto uc=UploadTex(d.Get(),list.Get(),color.Get(),input),um=UploadTex(d.Get(),list.Get(),mv.Get(),std::vector<uint8_t>(RW*RH*4,0));
        Recorder::Transition(list.Get(),out.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
        auto uo=UploadTex(d.Get(),list.Get(),out.Get(),input);Recorder::Transition(list.Get(),out.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RESOURCE_DESC zd{};zd.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;zd.Width=RW;zd.Height=RH;zd.DepthOrArraySize=1;zd.MipLevels=1;zd.Format=DXGI_FORMAT_D32_FLOAT_S8X24_UINT;zd.SampleDesc.Count=1;zd.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_CLEAR_VALUE clear{};clear.Format=zd.Format;clear.DepthStencil={.5f,0xa7};
        ComPtr<ID3D12Resource> depth;HR_CHECK(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&zd,D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(&depth)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV;hd.NumDescriptors=1;ComPtr<ID3D12DescriptorHeap> heap;HR_CHECK(d->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));
        auto handle=heap->GetCPUDescriptorHandleForHeapStart();d->CreateDepthStencilView(depth.Get(),nullptr,handle);
        list->ClearDepthStencilView(handle,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,.5f,0xa7,0,nullptr);
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={depth.Get(),0,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};list->ResourceBarrier(1,&b);
        // Stencil deliberately remains DEPTH_WRITE while depth is shader-read.
        // The former ALL_SUBRESOURCES copier fails this debug-layer state test.
        ffxCreateContextDescUpscale context{};context.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;context.maxRenderSize={RW,RH};context.maxUpscaleSize={OW,OH};
        ffx_capture::Collector capture(d.Get(),root/"capture",context,1,1,16*1024*1024,ffx_capture::DepthPlanePolicy::D32S8DepthOnly);
        ffxDispatchDescUpscale a{};a.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;a.commandList=list.Get();a.renderSize={RW,RH};a.upscaleSize={0,0};
        a.color=ffxApiGetResourceDX12(color.Get());a.motionVectors=ffxApiGetResourceDX12(mv.Get());a.depth=ffxApiGetResourceDX12(depth.Get(),FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        a.depth.description.usage=FFX_API_RESOURCE_USAGE_DEPTHTARGET;a.output=ffxApiGetResourceDX12(out.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        auto ticket=capture.Before(a,0);capture.After(ticket,list.Get());
        auto stencilFp=Footprint(d.Get(),depth.Get(),1);auto stencilRb=Readback(d.Get(),stencilFp.bytes);
        b.Transition={depth.Get(),1,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_SOURCE};list->ResourceBarrier(1,&b);
        auto src=Recorder::TextureLocation(depth.Get());src.SubresourceIndex=1;auto dst=Linear(stencilRb.Get(),stencilFp);list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        std::swap(b.Transition.StateBefore,b.Transition.StateAfter);list->ResourceBarrier(1,&b);
        HR_CHECK(list->Close());ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);capture.Submitted(ticket,queue.Get(),list.Get());
        ComPtr<ID3D12Fence> fence;HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!event||FAILED(queue->Signal(fence.Get(),1))||FAILED(fence->SetEventOnCompletion(1,event))||WaitForSingleObject(event,60000)!=WAIT_OBJECT_0)TerminateProcess(GetCurrentProcess(),3);
        CloseHandle(event);HR_CHECK(d->GetDeviceRemovedReason());if(capture.Poll()!=1||capture.Pending())throw std::runtime_error("collector did not retire");
        auto stencil=Unpack(stencilRb.Get(),stencilFp);Write(root/"stencil.raw",stencil);bool equal=stencil==std::vector<uint8_t>(RW*RH,0xa7);
        std::vector<uint8_t> expectedDepth(RW*RH*4);float half=.5f;for(size_t i=0;i<expectedDepth.size();i+=4)std::memcpy(expectedDepth.data()+i,&half,4);
        equal&=Read(root/"capture/0/depth.raw")==expectedDepth&&Read(root/"capture/0/output.raw")==input;
        unsigned errors=0;std::ofstream messages(root/"d3d12_messages.log");
        for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;HR_CHECK(info->GetMessage(i,nullptr,&n));std::vector<uint8_t> bytes(n);auto* m=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());HR_CHECK(info->GetMessage(i,m,&n));messages<<m->Severity<<": "<<m->pDescription<<'\n';if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)++errors;}
        bool pass=equal&&!errors;std::ofstream(root/"manifest.json")<<"{\"status\":\""<<(pass?"PLANE_STATE_PASS":"PLANE_STATE_FAIL")<<"\",\"debug_errors\":"<<errors<<",\"captured_depth_output_exact\":"<<(equal?"true":"false")<<",\"stencil_state_during_capture\":16,\"depth_state_during_capture\":192,\"provider_invoked\":false,\"game_pixels_captured\":false}\n";
        std::printf("%s debug_errors=%u\n",pass?"PLANE_STATE_PASS":"PLANE_STATE_FAIL",errors);return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
