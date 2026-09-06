#pragma once
// Exact output -> scratch -> output write-back at a proven readable boundary.
// This is a no-op transport proof: it never supplies new pixels or NR math.
#include "texture_capture.h"

namespace ffx_boundary {
using Microsoft::WRL::ComPtr;
using ffx_capture::Check;

class OutputRoundTrip {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12Resource> output_, scratch_, readback_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    UINT rows_=0;
    UINT64 rowBytes_=0,bytes_=0;
    bool recorded_=false,submitted_=false,retired_=false;
public:
    OutputRoundTrip(ID3D12Device* device,ID3D12Resource* output,UINT64 budget=64*1024*1024)
        : device_(device),output_(output) {
        if(!device||!output)throw std::invalid_argument("roundtrip null resource/device");
        ComPtr<ID3D12Device> owner;Check(output->GetDevice(IID_PPV_ARGS(&owner)));
        if(!ffx_capture::Same(owner.Get(),device))throw std::invalid_argument("roundtrip device mismatch");
        auto d=output->GetDesc();
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.MipLevels!=1||d.DepthOrArraySize!=1||
           d.SampleDesc.Count!=1||d.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT||!d.Width||!d.Height||
           d.Width>8192||d.Height>8192||(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS))
            throw std::invalid_argument("roundtrip unsupported output");
        device->GetCopyableFootprints(&d,0,1,0,&footprint_,&rows_,&rowBytes_,&bytes_);
        if(!bytes_||bytes_>budget||rowBytes_!=d.Width*8||rows_!=d.Height)
            throw std::invalid_argument("roundtrip byte/shape budget");
        D3D12_HEAP_PROPERTIES local{};local.Type=D3D12_HEAP_TYPE_DEFAULT;
        Check(device->CreateCommittedResource(&local,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&scratch_)));
        D3D12_HEAP_PROPERTIES read{};read.Type=D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes_;
        b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Check(device->CreateCommittedResource(&read,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback_)));
        Check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)));
    }
    OutputRoundTrip(const OutputRoundTrip&)=delete;
    OutputRoundTrip& operator=(const OutputRoundTrip&)=delete;
    ~OutputRoundTrip(){if(recorded_&&!retired_)std::terminate();}

    void Record(ID3D12GraphicsCommandList* list,const D3D12_RESOURCE_BARRIER& forwarded) {
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if(recorded_||!list||forwarded.Type!=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION||
           forwarded.Flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE||forwarded.Transition.pResource!=output_.Get()||
           (forwarded.Transition.Subresource!=0&&forwarded.Transition.Subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)||
           (forwarded.Transition.StateBefore!=D3D12_RESOURCE_STATE_UNORDERED_ACCESS&&forwarded.Transition.StateBefore!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)||
           forwarded.Transition.StateAfter!=readable||list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT)
            throw std::invalid_argument("roundtrip requires exact complete output transition");
        ComPtr<ID3D12Device> owner;Check(list->GetDevice(IID_PPV_ARGS(&owner)));
        if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("roundtrip list device");
        list_=list;recorded_=true;
        D3D12_RESOURCE_BARRIER toSource{};toSource.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toSource.Transition={output_.Get(),0,readable,D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1,&toSource);
        D3D12_TEXTURE_COPY_LOCATION source{},scratchDestination{},readbackDestination{};
        source.pResource=output_.Get();source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        scratchDestination.pResource=scratch_.Get();scratchDestination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        readbackDestination.pResource=readback_.Get();readbackDestination.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        readbackDestination.PlacedFootprint=footprint_;
        list->CopyTextureRegion(&scratchDestination,0,0,0,&source,nullptr);
        list->CopyTextureRegion(&readbackDestination,0,0,0,&source,nullptr);
        D3D12_RESOURCE_BARRIER swapStates[2]{};
        swapStates[0].Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        swapStates[0].Transition={output_.Get(),0,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COPY_DEST};
        swapStates[1].Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        swapStates[1].Transition={scratch_.Get(),0,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(2,swapStates);
        D3D12_TEXTURE_COPY_LOCATION scratchSource{},outputDestination{};
        scratchSource.pResource=scratch_.Get();scratchSource.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        outputDestination.pResource=output_.Get();outputDestination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&outputDestination,0,0,0,&scratchSource,nullptr);
        D3D12_RESOURCE_BARRIER restore{};restore.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        restore.Transition={output_.Get(),0,D3D12_RESOURCE_STATE_COPY_DEST,readable};
        list->ResourceBarrier(1,&restore);
    }

    void Submitted(ID3D12CommandQueue* queue,ID3D12GraphicsCommandList* list) {
        if(!recorded_||submitted_||!queue||!list||!ffx_capture::Same(list,list_.Get())||
           queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)
            throw std::invalid_argument("roundtrip submission association");
        ComPtr<ID3D12Device> owner;Check(queue->GetDevice(IID_PPV_ARGS(&owner)));
        if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("roundtrip queue device");
        queue_=queue;Check(queue->Signal(fence_.Get(),1));submitted_=true;
    }

    bool Poll(std::vector<uint8_t>& raw) {
        if(retired_)throw std::invalid_argument("roundtrip already retired");
        if(!submitted_)return false;
        Check(device_->GetDeviceRemovedReason());auto completed=fence_->GetCompletedValue();
        if(completed==UINT64_MAX)throw std::runtime_error("roundtrip device removed fence");
        if(completed<1)return false;
        raw.resize(size_t(rows_*rowBytes_));void* data=nullptr;D3D12_RANGE range{0,size_t(bytes_)};
        Check(readback_->Map(0,&range,&data));
        for(UINT y=0;y<rows_;++y)std::memcpy(raw.data()+y*rowBytes_,static_cast<uint8_t*>(data)+footprint_.Offset+y*footprint_.Footprint.RowPitch,size_t(rowBytes_));
        D3D12_RANGE none{0,0};readback_->Unmap(0,&none);retired_=true;return true;
    }
};
}
