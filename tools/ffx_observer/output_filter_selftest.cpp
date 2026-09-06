#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
#include "output_boundary_network.h"
#else
#include "output_boundary_filter.h"
#endif

int wmain(int argc,wchar_t** argv){
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
    if(argc!=4&&argc!=5)return 2;const bool useWarp=argc==5&&!wcscmp(argv[4],L"--warp");if(argc==5&&!useWarp)return 2;auto root=fs::absolute(argv[3]);
#else
    if(argc!=3)return 2;auto root=fs::absolute(argv[2]);
#endif
    if(fs::exists(root))return 2;fs::create_directories(root);try{
    auto shader=Read(fs::absolute(argv[1]));ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
    if(useWarp){HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));HR_CHECK(adapter->GetDesc1(&ad));found=true;}
    else
#endif
    for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()){HR_CHECK(adapter->GetDesc1(&ad));if(!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&ad.VendorId==0x1002){found=true;break;}}if(!found)throw std::runtime_error("AMD absent");
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
    auto weights=ffx_boundary::LoadResidualModel(fs::absolute(argv[2]));
#endif
    ComPtr<ID3D12Device>d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));ComPtr<ID3D12InfoQueue>info;HR_CHECK(d.As(&info));D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue>q;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&q)));uint64_t changed=0,components=0;bool exactBefore=true,alphaExact=true;
    for(UINT run=0;run<2;++run){UINT w=run?2342:640,h=run?1317:360;D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;td.Width=w;td.Height=h;td.DepthOrArraySize=1;td.MipLevels=1;td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;ComPtr<ID3D12Resource>texture;HR_CHECK(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&texture)));D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows;UINT64 rowBytes,total;d->GetCopyableFootprints(&td,0,1,0,&fp,&rows,&rowBytes,&total);std::vector<uint8_t>expected(size_t(w)*h*8),packed(static_cast<size_t>(total),0);for(UINT y=0;y<h;++y)for(UINT x=0;x<w;++x){uint16_t px[4]={uint16_t(0x3000+(x%512)),uint16_t(0x3200+(y%256)),uint16_t(0x3400+((x+y)%256)),0x3c00};std::memcpy(expected.data()+(size_t(y)*w+x)*8,px,8);}for(UINT y=0;y<h;++y)std::memcpy(packed.data()+fp.Offset+y*fp.Footprint.RowPitch,expected.data()+size_t(y)*rowBytes,size_t(rowBytes));auto upload=Upload(d.Get(),packed);ComPtr<ID3D12CommandAllocator>a;HR_CHECK(d->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&a)));ComPtr<ID3D12GraphicsCommandList>l;HR_CHECK(d->CreateCommandList(0,qd.Type,a.Get(),nullptr,IID_PPV_ARGS(&l)));D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;dst=Recorder::TextureLocation(texture.Get());l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);auto before=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;constexpr auto after=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;Recorder::Transition(l.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,before);D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={texture.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};l->ResourceBarrier(1,&b);
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
        ffx_boundary::OutputNetwork filter(d.Get(),texture.Get(),shader,weights);
#else
        ffx_boundary::OutputFilter filter(d.Get(),texture.Get(),shader);
#endif
        filter.Record(b);HR_CHECK(l->Close());ID3D12CommandList* first[]={l.Get()},*second[]={filter.CommandList()};q->ExecuteCommandLists(1,first);q->ExecuteCommandLists(1,second);filter.Submitted(q.Get());std::vector<uint8_t>original,filtered;for(UINT i=0;i<100&&!filter.Poll(original,filtered);++i)Sleep(10);if(original.empty())throw std::runtime_error("filter fence timeout");exactBefore&=original==expected;for(size_t i=0;i<filtered.size();i+=8){if(std::memcmp(original.data()+i,filtered.data()+i,8))++changed;uint16_t a0,a1;std::memcpy(&a0,original.data()+i+6,2);std::memcpy(&a1,filtered.data()+i+6,2);alphaExact&=a0==a1;for(UINT c=0;c<4;++c){uint16_t v,o;std::memcpy(&v,filtered.data()+i+c*2,2);std::memcpy(&o,original.data()+i+c*2,2);if((v&0x7c00)==0x7c00)throw std::runtime_error("non-finite filter");components+=v!=o;}}Write(root/(run?"actual.raw":"small.raw"),filtered.data(),filtered.size());}
    UINT errors=0,warnings=0;for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t>s(n);auto*m=reinterpret_cast<D3D12_MESSAGE*>(s.data());HR_CHECK(info->GetMessage(i,m,&n));errors+=m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR;warnings+=m->Severity==D3D12_MESSAGE_SEVERITY_WARNING;}bool pass=exactBefore&&alphaExact&&changed&&components&&!errors&&!warnings;std::ostringstream j;j<<"{\"status\":\""<<(pass?
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
    "OUTPUT_NETWORK_PASS"
#else
    "OUTPUT_FILTER_PASS"
#endif
    :"FAIL")<<"\",\"cases\":2,\"changed_pixels\":"<<changed<<",\"changed_components\":"<<components<<",\"before_exact\":"<<(exactBefore?"true":"false")<<",\"alpha_exact\":"<<(alphaExact?"true":"false")<<",\"d3d12_errors\":"<<errors<<",\"d3d12_warnings\":"<<warnings<<",\"dynamic_resolution\":true,\"external_weights\":"<<
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
    "true"
#else
    "false"
#endif
    <<",\"trained_weights\":false,\"adapter\":\""<<
#ifdef FFX_OUTPUT_NETWORK_SELFTEST
    (useWarp?"WARP":"AMD")
#else
    "AMD"
#endif
    <<"\",\"dlss_nr_verified\":false}\n";auto text=j.str();Write(root/"summary.json",text.data(),text.size());std::printf("%s",text.c_str());return pass?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"filter test failed: %s\n",e.what());return 1;}}
