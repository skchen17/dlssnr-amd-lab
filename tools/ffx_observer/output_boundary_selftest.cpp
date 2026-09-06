// Isolated synthetic output copy at actual GoWR size. No game attach/FFX claims.
#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "output_boundary_copy.h"
#include "output_boundary_roundtrip.h"

int wmain(int argc,wchar_t** argv) {
    if(argc!=2)return 2;
    auto root=fs::absolute(argv[1]);if(fs::exists(root))return 2;fs::create_directories(root);
    try {
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
            HR_CHECK(adapter->GetDesc1(&ad));if(!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&ad.VendorId==0x1002){found=true;break;}
        }if(!found)throw std::runtime_error("AMD adapter absent");
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> q;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)));
        UINT rejected=0,early=0;UINT64 bytes=0;bool exact=true,restore=true;
        for(UINT run=0;run<8;++run) {
            const bool roundTrip=run>=4;const UINT shape=run%4;
            const UINT width=shape<2?640:2342,height=shape<2?360:1317;
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;td.Width=width;td.Height=height;
            td.DepthOrArraySize=1;td.MipLevels=1;td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
            td.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET|D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ComPtr<ID3D12Resource> texture;HR_CHECK(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&texture)));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows;UINT64 rowBytes,total;d->GetCopyableFootprints(&td,0,1,0,&fp,&rows,&rowBytes,&total);
            std::vector<uint8_t> expected(size_t(width)*height*8),packed(size_t(total),0);
            for(size_t i=0;i<expected.size();i+=2){uint16_t value=uint16_t(0x3000+((i/2+run*83)%0xc00));std::memcpy(expected.data()+i,&value,2);}
            for(UINT y=0;y<height;++y)std::memcpy(packed.data()+fp.Offset+y*fp.Footprint.RowPitch,expected.data()+y*rowBytes,size_t(rowBytes));
            auto upload=Upload(d.Get(),packed),downstream=Readback(d.Get(),total);
            ComPtr<ID3D12CommandAllocator> a;HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&a)));
            ComPtr<ID3D12GraphicsCommandList> l;HR_CHECK(d->CreateCommandList(0,qd.Type,a.Get(),nullptr,IID_PPV_ARGS(&l)));
            D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;
            dst=Recorder::TextureLocation(texture.Get());l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
            auto before=shape%2?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            constexpr auto after=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            Recorder::Transition(l.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,before);
            std::unique_ptr<ffx_boundary::OutputCopy> copy;
            std::unique_ptr<ffx_boundary::OutputRoundTrip> roundTripCopy;
            if(roundTrip)roundTripCopy=std::make_unique<ffx_boundary::OutputRoundTrip>(d.Get(),texture.Get());
            else copy=std::make_unique<ffx_boundary::OutputCopy>(d.Get(),texture.Get());
            std::vector<uint8_t> raw;
            auto poll=[&](){return roundTrip?roundTripCopy->Poll(raw):copy->Poll(raw);};
            auto record=[&](const D3D12_RESOURCE_BARRIER& barrier){if(roundTrip)roundTripCopy->Record(l.Get(),barrier);else copy->Record(l.Get(),barrier);};
            if(poll())throw std::runtime_error("poll before recording");++early;
            D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={texture.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};
            for(UINT invalid=0;invalid<6;++invalid) {
                auto bad=b;
                if(invalid==0)bad.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
                if(invalid==1)bad.Flags=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
                if(invalid==2)bad.Transition.Subresource=1;
                if(invalid==3)bad.Transition.pResource=nullptr;
                if(invalid==4)bad.Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                if(invalid==5)bad.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
                bool refused=false;try{record(bad);}catch(const std::invalid_argument&){refused=true;}
                if(!refused)throw std::runtime_error("unsafe boundary accepted");++rejected;
            }
            bool limited=false;try{if(roundTrip){ffx_boundary::OutputRoundTrip limitedCopy(d.Get(),texture.Get(),1);}else{ffx_boundary::OutputCopy limitedCopy(d.Get(),texture.Get(),1);}}catch(const std::invalid_argument&){limited=true;}
            if(!limited)throw std::runtime_error("byte cap accepted");++rejected;
            l->ResourceBarrier(1,&b);record(b);
            bool duplicate=false;try{record(b);}catch(const std::invalid_argument&){duplicate=true;}
            if(!duplicate)throw std::runtime_error("duplicate recording");++rejected;
            // A second independent readback uses the exact restored state.
            Recorder::Transition(l.Get(),texture.Get(),after,D3D12_RESOURCE_STATE_COPY_SOURCE);
            src=Recorder::TextureLocation(texture.Get());dst.pResource=downstream.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;
            l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Recorder::Transition(l.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,after);
            HR_CHECK(l->Close());ComPtr<ID3D12Fence> gate,done;
            HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&gate)));HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&done)));
            HR_CHECK(q->Wait(gate.Get(),1));ID3D12CommandList* submitted[]={l.Get()};q->ExecuteCommandLists(1,submitted);
            if(roundTrip)roundTripCopy->Submitted(q.Get(),l.Get());else copy->Submitted(q.Get(),l.Get());
            if(poll())throw std::runtime_error("mapped before fenced completion");++early;
            HR_CHECK(gate->Signal(1));HR_CHECK(q->Signal(done.Get(),1));HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            if(!event||FAILED(done->SetEventOnCompletion(1,event))||WaitForSingleObject(event,10000)!=WAIT_OBJECT_0)TerminateProcess(GetCurrentProcess(),3);
            CloseHandle(event);HR_CHECK(d->GetDeviceRemovedReason());if(!poll())throw std::runtime_error("copy not complete");
            auto packedDown=ReadGpu(downstream.Get(),size_t(total));std::vector<uint8_t> plain(expected.size());
            for(UINT y=0;y<height;++y)std::memcpy(plain.data()+y*rowBytes,packedDown.data()+fp.Offset+y*fp.Footprint.RowPitch,size_t(rowBytes));
            exact&=raw==expected;restore&=plain==expected;bytes+=raw.size();
            Write(root/((roundTrip?"roundtrip_":"copy_")+std::to_string(shape)+".raw"),raw.data(),raw.size());
        }
        UINT errors=0,warnings=0;
        for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T size=0;info->GetMessage(i,nullptr,&size);std::vector<uint8_t> storage(size);auto m=reinterpret_cast<D3D12_MESSAGE*>(storage.data());HR_CHECK(info->GetMessage(i,m,&size));
            if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)++errors;else if(m->Severity==D3D12_MESSAGE_SEVERITY_WARNING)++warnings;
            if(m->Severity<=D3D12_MESSAGE_SEVERITY_WARNING)std::fprintf(stderr,"D3D12: %s\n",m->pDescription);}
        bool pass=exact&&restore&&!errors&&!warnings&&rejected==64&&early==16;
        std::ostringstream report;report<<"{\"status\":\""<<(pass?"OUTPUT_BOUNDARY_PRIMITIVE_PASS":"FAIL")<<"\",\"adapter_vendor\":"<<ad.VendorId
          <<",\"cases\":8,\"copy_cases\":4,\"roundtrip_cases\":4,\"raw_bytes\":"<<bytes<<",\"captured_exact\":"<<(exact?"true":"false")<<",\"downstream_exact\":"<<(restore?"true":"false")
          <<",\"negative_checks\":"<<rejected<<",\"early_polls_blocked\":"<<early<<",\"d3d12_errors\":"<<errors<<",\"d3d12_warnings\":"<<warnings
          <<",\"synthetic_only\":true,\"live_hook_validated\":false,\"game_frame_capture_verified\":false,\"dlss_nr_verified\":false}\n";
        auto text=report.str();Write(root/"summary.json",text.data(),text.size());std::printf("%s",text.c_str());return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"boundary test failed: %s\n",e.what());return 1;}
}
