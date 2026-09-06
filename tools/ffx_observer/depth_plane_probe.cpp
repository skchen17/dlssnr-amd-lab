// Independent known-value D32_FLOAT_S8X24_UINT plane-copy test; no game attach.
#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"

int wmain(int argc,wchar_t** argv) {
    if(argc!=2){std::fprintf(stderr,"usage: ffx_depth_plane_probe <new result directory>\n");return 2;}
    auto root=fs::absolute(argv[1]);
    try {
        if(fs::exists(root)||!fs::create_directories(root))throw std::runtime_error("result must be new");
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
            HR_CHECK(adapter->GetDesc1(&ad));if(ad.VendorId==0x1002&&!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){found=true;break;}
        }if(!found)throw std::runtime_error("AMD adapter required");
        ComPtr<ID3D12Device> device;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(device.As(&info));
        D3D12_FEATURE_DATA_FORMAT_INFO format{DXGI_FORMAT_D32_FLOAT_S8X24_UINT,0};HR_CHECK(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO,&format,sizeof(format)));
        if(format.PlaneCount!=2)throw std::runtime_error("unexpected plane count");
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> queue;HR_CHECK(device->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12Fence> fence;HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        std::ofstream records(root/"planes.jsonl");unsigned cases=0;uint64_t mismatches=0;
        for(auto dimensions:{std::pair<UINT,UINT>{321,181},{1552,872},{2560,1440}}) {
            UINT w=dimensions.first,h=dimensions.second;++cases;
            ComPtr<ID3D12CommandAllocator> allocator;HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
            ComPtr<ID3D12GraphicsCommandList> list;HR_CHECK(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
            D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;desc.Width=w;desc.Height=h;desc.DepthOrArraySize=1;desc.MipLevels=1;
            desc.Format=DXGI_FORMAT_D32_FLOAT_S8X24_UINT;desc.SampleDesc.Count=1;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_CLEAR_VALUE clear{};clear.Format=desc.Format;clear.DepthStencil={.25f,0xa7};
            ComPtr<ID3D12Resource> texture;HR_CHECK(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(&texture)));
            auto expected=ffxApiGetResourceDX12(texture.Get(),FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV;hd.NumDescriptors=1;ComPtr<ID3D12DescriptorHeap> heap;
            HR_CHECK(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));auto dsv=heap->GetCPUDescriptorHandleForHeapStart();
            device->CreateDepthStencilView(texture.Get(),nullptr,dsv);
            list->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,.25f,0xa7,0,nullptr);
            // Distinct rectangles make a wrong stride/plane fail, not just a uniform clear.
            D3D12_RECT rect{LONG(w/3),LONG(h/3),LONG(2*w/3),LONG(2*h/3)};
            list->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,.75f,0x3c,1,&rect);
            Recorder::Transition(list.Get(),texture.Get(),D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2]{};UINT rows[2]{};UINT64 rowBytes[2]{},bytes[2]{};
            ComPtr<ID3D12Resource> rb[2];
            for(UINT plane=0;plane<2;++plane) {
                device->GetCopyableFootprints(&desc,plane,1,0,&fp[plane],&rows[plane],&rowBytes[plane],&bytes[plane]);
                rb[plane]=Readback(device.Get(),bytes[plane]);
                D3D12_TEXTURE_COPY_LOCATION src{};src.pResource=texture.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;src.SubresourceIndex=plane;
                D3D12_TEXTURE_COPY_LOCATION dst{};dst.pResource=rb[plane].Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp[plane];
                list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
            }
            Recorder::Transition(list.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_DEPTH_WRITE);
            HR_CHECK(list->Close());ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);
            HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            if(!event||FAILED(queue->Signal(fence.Get(),cases))||FAILED(fence->SetEventOnCompletion(cases,event))||WaitForSingleObject(event,60000)!=WAIT_OBJECT_0)TerminateProcess(GetCurrentProcess(),3);
            CloseHandle(event);HR_CHECK(device->GetDeviceRemovedReason());
            for(UINT plane=0;plane<2;++plane) {
                auto packed=ReadGpu(rb[plane].Get(),size_t(bytes[plane]));std::vector<uint8_t> raw(size_t(rows[plane]*rowBytes[plane]));
                for(UINT y=0;y<rows[plane];++y)std::memcpy(raw.data()+y*rowBytes[plane],packed.data()+fp[plane].Offset+y*fp[plane].Footprint.RowPitch,size_t(rowBytes[plane]));
                if(rows[plane]!=h||rowBytes[plane]!=w*(plane?1u:4u))throw std::runtime_error("unexpected logical plane layout");
                uint64_t bad=0;
                for(UINT y=0;y<h;++y)for(UINT x=0;x<w;++x) {
                    bool inner=x>=UINT(rect.left)&&x<UINT(rect.right)&&y>=UINT(rect.top)&&y<UINT(rect.bottom);
                    if(plane){if(raw[y*w+x]!=(inner?0x3c:0xa7))++bad;}
                    else {float value;std::memcpy(&value,raw.data()+4*(y*w+x),4);if(value!=(inner?.75f:.25f))++bad;}
                }mismatches+=bad;
                std::string filename=std::to_string(w)+"x"+std::to_string(h)+"_plane"+std::to_string(plane)+".raw";Write(root/filename,raw.data(),raw.size());
                records<<"{\"width\":"<<w<<",\"height\":"<<h<<",\"source_dxgi_format\":20,\"plane\":"<<plane<<",\"plane_count\":2,\"copy_format\":"<<fp[plane].Footprint.Format
                  <<",\"footprint_width\":"<<fp[plane].Footprint.Width<<",\"footprint_height\":"<<fp[plane].Footprint.Height<<",\"row_pitch\":"<<fp[plane].Footprint.RowPitch
                  <<",\"row_bytes\":"<<rowBytes[plane]<<",\"footprint_bytes\":"<<bytes[plane]<<",\"raw_bytes\":"<<raw.size()<<",\"mismatched_pixels\":"<<bad
                  <<",\"sdk_ffx_format\":"<<expected.description.format<<",\"sdk_ffx_usage\":"<<expected.description.usage<<",\"file\":\""<<filename<<"\"}\n";
            }
        }
        unsigned errors=0,warnings=0;std::ofstream messages(root/"d3d12_messages.log");
        for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;HR_CHECK(info->GetMessage(i,nullptr,&n));std::vector<uint8_t> b(n);auto* m=reinterpret_cast<D3D12_MESSAGE*>(b.data());HR_CHECK(info->GetMessage(i,m,&n));messages<<m->Severity<<": "<<m->pDescription<<'\n';if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)++errors;if(m->Severity==D3D12_MESSAGE_SEVERITY_WARNING)++warnings;}
        bool pass=!errors&&!mismatches;
        std::ofstream(root/"manifest.json")<<"{\"status\":\""<<(pass?"DEPTH_PLANE_PASS":"DEPTH_PLANE_FAIL")<<"\",\"cases\":"<<cases<<",\"completed_fences\":"<<cases
          <<",\"debug_errors\":"<<errors<<",\"debug_warnings\":"<<warnings<<",\"mismatched_pixels\":"<<mismatches<<",\"adapter_vendor\":"<<ad.VendorId<<",\"adapter_device\":"<<ad.DeviceId<<",\"game_frame\":false}\n";
        std::printf("%s mismatches=%llu debug_errors=%u warnings=%u\n",pass?"DEPTH_PLANE_PASS":"DEPTH_PLANE_FAIL",(unsigned long long)mismatches,errors,warnings);return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
