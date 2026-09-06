#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "resident_surface.h"

static void Submit(ID3D12CommandQueue* q,UINT n,ID3D12CommandList*const* lists){q->ExecuteCommandLists(n,lists);}
static void Wait(ID3D12Device* d,ID3D12CommandQueue* q){
    ComPtr<ID3D12Fence> fence;HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));HR_CHECK(q->Signal(fence.Get(),1));
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event||FAILED(fence->SetEventOnCompletion(1,event))||WaitForSingleObject(event,2000)!=WAIT_OBJECT_0)TerminateProcess(GetCurrentProcess(),3);
    CloseHandle(event);HR_CHECK(d->GetDeviceRemovedReason());
}
int wmain(int argc,wchar_t** argv){
    if(argc!=7)return 2;const bool amd=!wcscmp(argv[2],L"--amd");if(!amd&&wcscmp(argv[2],L"--warp"))return 2;
    auto root=fs::absolute(argv[1]);if(fs::exists(root))return 2;fs::create_directories(root);
    try{
        UINT width=std::stoul(argv[5]),height=std::stoul(argv[6]);auto source=Read(fs::path(argv[4]));
        if(source.size()!=size_t(width)*height*8)throw std::invalid_argument("source shape");
        ffx_resident::Exchange exchange(argv[3]);
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};
        if(amd){bool found=false;for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()){
            adapter->GetDesc1(&ad);if(ad.VendorId==0x1002&&!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){found=true;break;}}if(!found)throw std::runtime_error("AMD missing");}
        else{HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));adapter->GetDesc1(&ad);}
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> q;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)));
        D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;td.Width=width;td.Height=height;td.DepthOrArraySize=1;td.MipLevels=1;td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;ComPtr<ID3D12Resource> target;
        HR_CHECK(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&target)));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows;UINT64 rowBytes,total;d->GetCopyableFootprints(&td,0,1,0,&fp,&rows,&rowBytes,&total);
        std::vector<uint8_t> packed(size_t(total),0);for(UINT y=0;y<height;++y)std::memcpy(packed.data()+fp.Offset+y*fp.Footprint.RowPitch,source.data()+y*rowBytes,size_t(rowBytes));
        auto upload=Upload(d.Get(),packed),down=Readback(d.Get(),total);
        ComPtr<ID3D12CommandAllocator> alloc,consumerAlloc;ComPtr<ID3D12GraphicsCommandList> producer,consumer;
        HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&alloc)));HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&consumerAlloc)));HR_CHECK(d->CreateCommandList(0,qd.Type,alloc.Get(),nullptr,IID_PPV_ARGS(&producer)));HR_CHECK(d->CreateCommandList(0,qd.Type,consumerAlloc.Get(),nullptr,IID_PPV_ARGS(&consumer)));
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        D3D12_TEXTURE_COPY_LOCATION src{},dst=Recorder::TextureLocation(target.Get());src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;
        producer->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(producer.Get(),target.Get(),D3D12_RESOURCE_STATE_COPY_DEST,readable);HR_CHECK(producer->Close());
        Recorder::Transition(consumer.Get(),target.Get(),readable,D3D12_RESOURCE_STATE_COPY_SOURCE);src=Recorder::TextureLocation(target.Get());dst.pResource=down.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;consumer->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(consumer.Get(),target.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,readable);HR_CHECK(consumer->Close());
        auto* bridge=new ffx_resident::Surface(d.Get(),target.Get());
        ID3D12CommandList* batch[]={producer.Get()};Submit(q.Get(),1,batch);
        auto result=bridge->Run(q.Get(),Submit,exchange);delete bridge;
        batch[0]=consumer.Get();Submit(q.Get(),1,batch);Wait(d.Get(),q.Get());
        auto bytes=ReadGpu(down.Get(),size_t(total));std::vector<uint8_t> actual(source.size());
        for(UINT y=0;y<height;++y)std::memcpy(actual.data()+size_t(y)*width*8,bytes.data()+fp.Offset+size_t(y)*fp.Footprint.RowPitch,size_t(width)*8);
        Write(root/"input.raw",result.input.data(),result.input.size());Write(root/"output.raw",actual.data(),actual.size());
        UINT errors=0,warnings=0;for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t> b(n);auto* msg=reinterpret_cast<D3D12_MESSAGE*>(b.data());info->GetMessage(i,msg,&n);errors+=msg->Severity<=D3D12_MESSAGE_SEVERITY_ERROR;warnings+=msg->Severity==D3D12_MESSAGE_SEVERITY_WARNING;if(msg->Severity<=D3D12_MESSAGE_SEVERITY_WARNING)std::fprintf(stderr,"%s\n",msg->pDescription);}
        bool pass=result.exact&&result.input==source&&actual==result.output&&!errors&&!warnings;
        std::ostringstream report;report<<"{\"status\":\""<<(pass?"PASS":"FAIL")<<"\",\"adapter_vendor\":"<<ad.VendorId<<",\"width\":"<<width<<",\"height\":"<<height<<",\"input_exact\":"<<(result.input==source?"true":"false")<<",\"consumer_output_exact\":"<<(actual==result.output?"true":"false")<<",\"d3d12_errors\":"<<errors<<",\"d3d12_warnings\":"<<warnings<<",\"game_runtime_verified\":false}\n";
        auto text=report.str();Write(root/"summary.json",text.data(),text.size());std::printf("%s",text.c_str());return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"resident surface test: %s\n",e.what());return 1;}
}
