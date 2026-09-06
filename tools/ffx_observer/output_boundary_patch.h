#pragma once
// One-frame 8x8 supplied-pixel patch at the proven post-FFX output boundary.
// This validates replacement transport only; it performs no NR computation.
#include "texture_capture.h"

namespace ffx_boundary {
using Microsoft::WRL::ComPtr;
using ffx_capture::Check;

class OutputPatch {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12Resource> output_,upload_,before_,after_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    UINT rows_=0;
    UINT64 rowBytes_=0,bytes_=0;
    bool recorded_=false,submitted_=false,retired_=false;
public:
    static constexpr UINT PatchWidth=8,PatchHeight=8,PatchRowPitch=256;
    static constexpr uint16_t PatchPixel[4]={0x3c00,0x0000,0x3c00,0x3c00};
    OutputPatch(ID3D12Device* device,ID3D12Resource* output,UINT64 budget=128*1024*1024)
        :device_(device),output_(output) {
        if(!device||!output)throw std::invalid_argument("patch null resource/device");
        ComPtr<ID3D12Device> owner;Check(output->GetDevice(IID_PPV_ARGS(&owner)));
        if(!ffx_capture::Same(owner.Get(),device))throw std::invalid_argument("patch device mismatch");
        auto d=output->GetDesc();
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.MipLevels!=1||d.DepthOrArraySize!=1||
           d.SampleDesc.Count!=1||d.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT||d.Width<PatchWidth||d.Height<PatchHeight||
           d.Width>8192||d.Height>8192||(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS))
            throw std::invalid_argument("patch unsupported output");
        device->GetCopyableFootprints(&d,0,1,0,&footprint_,&rows_,&rowBytes_,&bytes_);
        if(!bytes_||bytes_>budget/2||rowBytes_!=d.Width*8||rows_!=d.Height)
            throw std::invalid_argument("patch byte/shape budget");
        D3D12_HEAP_PROPERTIES read{};read.Type=D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes_;
        b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Check(device->CreateCommittedResource(&read,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&before_)));
        Check(device->CreateCommittedResource(&read,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&after_)));
        D3D12_HEAP_PROPERTIES upload{};upload.Type=D3D12_HEAP_TYPE_UPLOAD;b.Width=UINT64(PatchRowPitch)*PatchHeight;
        Check(device->CreateCommittedResource(&upload,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload_)));
        void* mapped=nullptr;D3D12_RANGE none{0,0};Check(upload_->Map(0,&none,&mapped));std::memset(mapped,0,size_t(b.Width));
        for(UINT y=0;y<PatchHeight;++y)for(UINT x=0;x<PatchWidth;++x)
            std::memcpy(static_cast<uint8_t*>(mapped)+y*PatchRowPitch+x*8,PatchPixel,8);
        upload_->Unmap(0,nullptr);Check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)));
    }
    OutputPatch(const OutputPatch&)=delete;
    OutputPatch& operator=(const OutputPatch&)=delete;
    ~OutputPatch(){if(recorded_&&!retired_)std::terminate();}

    void Record(ID3D12GraphicsCommandList* list,const D3D12_RESOURCE_BARRIER& forwarded) {
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if(recorded_||!list||forwarded.Type!=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION||
           forwarded.Flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE||forwarded.Transition.pResource!=output_.Get()||
           (forwarded.Transition.Subresource!=0&&forwarded.Transition.Subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)||
           (forwarded.Transition.StateBefore!=D3D12_RESOURCE_STATE_UNORDERED_ACCESS&&forwarded.Transition.StateBefore!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)||
           forwarded.Transition.StateAfter!=readable||list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT)
            throw std::invalid_argument("patch requires exact complete output transition");
        ComPtr<ID3D12Device> owner;Check(list->GetDevice(IID_PPV_ARGS(&owner)));
        if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("patch list device");
        list_=list;recorded_=true;
        D3D12_RESOURCE_BARRIER transition{};transition.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition={output_.Get(),0,readable,D3D12_RESOURCE_STATE_COPY_SOURCE};list->ResourceBarrier(1,&transition);
        D3D12_TEXTURE_COPY_LOCATION source{},destination{};
        source.pResource=output_.Get();source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource=before_.Get();destination.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;destination.PlacedFootprint=footprint_;
        list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;transition.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_DEST;list->ResourceBarrier(1,&transition);
        source.pResource=upload_.Get();source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint.Offset=0;source.PlacedFootprint.Footprint={DXGI_FORMAT_R16G16B16A16_FLOAT,PatchWidth,PatchHeight,1,PatchRowPitch};
        destination.pResource=output_.Get();destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;transition.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;list->ResourceBarrier(1,&transition);
        source.pResource=output_.Get();source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource=after_.Get();destination.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;destination.PlacedFootprint=footprint_;
        list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;transition.Transition.StateAfter=readable;list->ResourceBarrier(1,&transition);
    }
    void Submitted(ID3D12CommandQueue* queue,ID3D12GraphicsCommandList* list) {
        if(!recorded_||submitted_||!queue||!list||!ffx_capture::Same(list,list_.Get())||queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)
            throw std::invalid_argument("patch submission association");
        ComPtr<ID3D12Device> owner;Check(queue->GetDevice(IID_PPV_ARGS(&owner)));
        if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("patch queue device");
        queue_=queue;Check(queue->Signal(fence_.Get(),1));submitted_=true;
    }
    bool Poll(std::vector<uint8_t>& before,std::vector<uint8_t>& after) {
        if(retired_)throw std::invalid_argument("patch already retired");if(!submitted_)return false;
        Check(device_->GetDeviceRemovedReason());auto completed=fence_->GetCompletedValue();
        if(completed==UINT64_MAX)throw std::runtime_error("patch device removed fence");if(completed<1)return false;
        auto read=[&](ID3D12Resource* resource,std::vector<uint8_t>& raw){raw.resize(size_t(rows_*rowBytes_));void* data=nullptr;D3D12_RANGE range{0,size_t(bytes_)};
            Check(resource->Map(0,&range,&data));for(UINT y=0;y<rows_;++y)std::memcpy(raw.data()+y*rowBytes_,static_cast<uint8_t*>(data)+footprint_.Offset+y*footprint_.Footprint.RowPitch,size_t(rowBytes_));
            D3D12_RANGE none{0,0};resource->Unmap(0,&none);};
        read(before_.Get(),before);read(after_.Get(),after);retired_=true;return true;
    }
};
}
