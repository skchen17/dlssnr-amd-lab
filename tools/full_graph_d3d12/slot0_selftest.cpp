#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "full_graph_prefix_runtime.h"

namespace {
void Wait(ID3D12Device* device,ID3D12CommandQueue* queue){
    ComPtr<ID3D12Fence> fence;HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(),1));HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event)throw std::runtime_error("event");
    HR_CHECK(fence->SetEventOnCompletion(1,event));auto status=WaitForSingleObject(event,30000);CloseHandle(event);
    if(status!=WAIT_OBJECT_0)throw std::runtime_error("slot0 timeout");HR_CHECK(device->GetDeviceRemovedReason());
}
}

int wmain(int argc,wchar_t** argv){
    if(argc!=4&&argc!=5)return 2;const bool warp=argc==5&&!wcscmp(argv[4],L"--warp");if(argc==5&&!warp)return 2;
    const auto shader=Read(fs::absolute(argv[1])),input=Read(fs::absolute(argv[2]));const auto root=fs::absolute(argv[3]);
    if(input.size()!=full_graph_dx12::ActivationArenaBytes||fs::exists(root))return 2;fs::create_directories(root);
    try{
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
        if(warp){HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));HR_CHECK(adapter->GetDesc1(&ad));found=true;}
        else for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()){HR_CHECK(adapter->GetDesc1(&ad));if(!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&ad.VendorId==0x1002){found=true;break;}}
        if(!found)throw std::runtime_error("adapter absent");ComPtr<ID3D12Device> device;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(device.As(&info));D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> queue;HR_CHECK(device->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
        D3D12_HEAP_PROPERTIES local{};local.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;desc.Width=input.size();desc.Height=1;desc.DepthOrArraySize=1;desc.MipLevels=1;desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> activation;HR_CHECK(device->CreateCommittedResource(&local,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&activation)));
        auto upload=Upload(device.Get(),input),readback=Readback(device.Get(),input.size());ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> list;
        HR_CHECK(device->CreateCommandAllocator(qd.Type,IID_PPV_ARGS(&allocator)));HR_CHECK(device->CreateCommandList(0,qd.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
        Recorder::Transition(list.Get(),activation.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);list->CopyBufferRegion(activation.Get(),0,upload.Get(),0,input.size());Recorder::Transition(list.Get(),activation.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        full_graph_dx12::PrefixSlot0 slot0(device.Get(),shader);slot0.Record(list.Get(),activation.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Recorder::Transition(list.Get(),activation.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);list->CopyBufferRegion(readback.Get(),0,activation.Get(),0,input.size());HR_CHECK(list->Close());
        ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);Wait(device.Get(),queue.Get());auto output=ReadGpu(readback.Get(),input.size());
        uint64_t nonzeroPrefix=0,suffixMismatch=0;for(size_t i=0;i<full_graph_dx12::Slot0Bytes;++i)nonzeroPrefix+=output[i]!=0;
        for(size_t i=full_graph_dx12::Slot0Bytes;i<input.size();++i)suffixMismatch+=output[i]!=input[i];
        UINT errors=0,warnings=0;std::ostringstream messages;for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T n=0;info->GetMessage(i,nullptr,&n);std::vector<uint8_t>s(n);auto*m=reinterpret_cast<D3D12_MESSAGE*>(s.data());HR_CHECK(info->GetMessage(i,m,&n));errors+=m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR;warnings+=m->Severity==D3D12_MESSAGE_SEVERITY_WARNING;messages<<unsigned(m->Severity)<<": "<<(m->pDescription?m->pDescription:"")<<"\n";}auto messageText=messages.str();Write(root/"d3d12_messages.log",messageText.data(),messageText.size());
        const bool pass=!nonzeroPrefix&&!suffixMismatch&&!errors&&!warnings;std::ostringstream summary;summary<<"{\"status\":\""<<(pass?"FULL_GRAPH_SLOT0_D3D12_PASS":"FAIL")<<"\",\"adapter\":\""<<(warp?"WARP":"AMD")<<"\",\"activation_bytes\":"<<input.size()<<",\"cleared_bytes\":"<<full_graph_dx12::Slot0Bytes<<",\"nonzero_prefix_bytes\":"<<nonzeroPrefix<<",\"suffix_mismatches\":"<<suffixMismatch<<",\"d3d12_errors\":"<<errors<<",\"d3d12_warnings\":"<<warnings<<",\"complete_network\":false,\"game_runtime_ready\":false}\n";auto text=summary.str();Write(root/"summary.json",text.data(),text.size());std::printf("%s",text.c_str());return pass?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"slot0 test failed: %s\n",e.what());return 1;}
}
