#pragma once
// Dynamic-resolution residual compute on a private command list. The caller
// inserts this list between the proven FFX producer and its downstream suffix.
#include "texture_capture.h"

namespace ffx_boundary {
using Microsoft::WRL::ComPtr;
using ffx_capture::Check;

class OutputFilter {
    ComPtr<ID3D12Device> device_;ComPtr<ID3D12Resource> output_,scratch_,before_,after_;
    ComPtr<ID3D12DescriptorHeap> descriptors_;ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;ComPtr<ID3D12CommandQueue> queue_;ComPtr<ID3D12Fence> fence_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};UINT rows_=0,width_=0,height_=0;UINT64 rowBytes_=0,bytes_=0;
    bool recorded_=false,submitted_=false,retired_=false;
    static void Transition(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={resource,0,before,after};list->ResourceBarrier(1,&b);}
public:
    OutputFilter(ID3D12Device* device,ID3D12Resource* output,const std::vector<uint8_t>& shader,UINT64 budget=128*1024*1024)
        :device_(device),output_(output){
        if(!device||!output||shader.empty()||shader.size()>1024*1024)throw std::invalid_argument("filter null/invalid input");
        ComPtr<ID3D12Device> owner;Check(output->GetDevice(IID_PPV_ARGS(&owner)));if(!ffx_capture::Same(owner.Get(),device))throw std::invalid_argument("filter device mismatch");
        auto d=output->GetDesc();width_=UINT(d.Width);height_=d.Height;
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.MipLevels!=1||d.DepthOrArraySize!=1||d.SampleDesc.Count!=1||
           d.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT||!width_||!height_||width_>8192||height_>8192||
           (d.Flags&(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS|D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)))throw std::invalid_argument("filter unsupported output");
        device->GetCopyableFootprints(&d,0,1,0,&footprint_,&rows_,&rowBytes_,&bytes_);if(!bytes_||bytes_>budget/2||rowBytes_!=UINT64(width_)*8||rows_!=height_)throw std::invalid_argument("filter byte/shape budget");
        D3D12_HEAP_PROPERTIES local{};local.Type=D3D12_HEAP_TYPE_DEFAULT;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(device->CreateCommittedResource(&local,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&scratch_)));
        D3D12_HEAP_PROPERTIES read{};read.Type=D3D12_HEAP_TYPE_READBACK;D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes_;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Check(device->CreateCommittedResource(&read,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&before_)));
        Check(device->CreateCommittedResource(&read,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&after_)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=2;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;Check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&descriptors_)));
        auto cpu=descriptors_->GetCPUDescriptorHandleForHeapStart();auto step=device->GetDescriptorHandleIncrementSize(hd.Type);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};srv.Format=d.Format;srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;srv.Texture2D.MipLevels=1;device->CreateShaderResourceView(output,&srv,cpu);
        cpu.ptr+=step;D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};uav.Format=d.Format;uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;device->CreateUnorderedAccessView(scratch_.Get(),nullptr,&uav,cpu);
        D3D12_DESCRIPTOR_RANGE ranges[2]{};D3D12_ROOT_PARAMETER params[2]{};
        for(UINT i=0;i<2;++i){ranges[i].RangeType=i?D3D12_DESCRIPTOR_RANGE_TYPE_UAV:D3D12_DESCRIPTOR_RANGE_TYPE_SRV;ranges[i].NumDescriptors=1;ranges[i].BaseShaderRegister=0;ranges[i].OffsetInDescriptorsFromTableStart=0;params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[i].DescriptorTable={1,&ranges[i]};}
        D3D12_ROOT_SIGNATURE_DESC rd{};rd.NumParameters=2;rd.pParameters=params;ComPtr<ID3DBlob> blob,error;Check(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error));Check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root_)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};pd.pRootSignature=root_.Get();pd.CS={shader.data(),shader.size()};Check(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(&pipeline_)));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator_)));Check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator_.Get(),nullptr,IID_PPV_ARGS(&list_)));Check(list_->Close());Check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)));
    }
    ~OutputFilter(){if(recorded_&&!retired_)std::terminate();}
    void Record(const D3D12_RESOURCE_BARRIER& forwarded){
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if(recorded_||forwarded.Type!=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION||forwarded.Flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE||forwarded.Transition.pResource!=output_.Get()||
           (forwarded.Transition.Subresource!=0&&forwarded.Transition.Subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)||
           (forwarded.Transition.StateBefore!=D3D12_RESOURCE_STATE_UNORDERED_ACCESS&&forwarded.Transition.StateBefore!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)||forwarded.Transition.StateAfter!=readable)
            throw std::invalid_argument("filter requires exact complete output transition");
        Check(allocator_->Reset());Check(list_->Reset(allocator_.Get(),nullptr));recorded_=true;
        D3D12_TEXTURE_COPY_LOCATION src{},dst{};Transition(list_.Get(),output_.Get(),readable,D3D12_RESOURCE_STATE_COPY_SOURCE);
        src.pResource=output_.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;dst.pResource=before_.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=footprint_;list_->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Transition(list_.Get(),output_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,readable);
        ID3D12DescriptorHeap* heaps[]={descriptors_.Get()};list_->SetDescriptorHeaps(1,heaps);list_->SetComputeRootSignature(root_.Get());list_->SetPipelineState(pipeline_.Get());auto gpu=descriptors_->GetGPUDescriptorHandleForHeapStart();auto step=device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);list_->SetComputeRootDescriptorTable(0,gpu);gpu.ptr+=step;list_->SetComputeRootDescriptorTable(1,gpu);list_->Dispatch((width_+7)/8,(height_+7)/8,1);
        D3D12_RESOURCE_BARRIER u{};u.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;u.UAV.pResource=scratch_.Get();list_->ResourceBarrier(1,&u);Transition(list_.Get(),scratch_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);Transition(list_.Get(),output_.Get(),readable,D3D12_RESOURCE_STATE_COPY_DEST);
        src.pResource=scratch_.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;dst.pResource=output_.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;list_->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Transition(list_.Get(),output_.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
        src.pResource=output_.Get();dst.pResource=after_.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=footprint_;list_->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Transition(list_.Get(),output_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,readable);Transition(list_.Get(),scratch_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);Check(list_->Close());
    }
    ID3D12CommandList* CommandList()const{return list_.Get();}
    void Submitted(ID3D12CommandQueue* queue){if(!recorded_||submitted_||!queue||queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)throw std::invalid_argument("filter submission");ComPtr<ID3D12Device> owner;Check(queue->GetDevice(IID_PPV_ARGS(&owner)));if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("filter queue device");queue_=queue;Check(queue->Signal(fence_.Get(),1));submitted_=true;}
    bool Poll(std::vector<uint8_t>& before,std::vector<uint8_t>& after){if(retired_)throw std::invalid_argument("filter retired");if(!submitted_)return false;Check(device_->GetDeviceRemovedReason());auto completed=fence_->GetCompletedValue();if(completed==UINT64_MAX)throw std::runtime_error("filter device removed");if(completed<1)return false;auto get=[&](ID3D12Resource* r,std::vector<uint8_t>& raw){raw.resize(size_t(rows_*rowBytes_));void* data=nullptr;D3D12_RANGE range{0,size_t(bytes_)};Check(r->Map(0,&range,&data));for(UINT y=0;y<rows_;++y)std::memcpy(raw.data()+y*rowBytes_,static_cast<uint8_t*>(data)+footprint_.Offset+y*footprint_.Footprint.RowPitch,size_t(rowBytes_));D3D12_RANGE none{0,0};r->Unmap(0,&none);};get(before_.Get(),before);get(after_.Get(),after);retired_=true;return true;}
};
}
