#pragma once
#include "texture_capture.h"
#include "resident_exchange.h"

namespace ffx_resident {
using Microsoft::WRL::ComPtr;
using ffx_capture::Check;
using Submit=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,UINT,ID3D12CommandList*const*);

// Bring-up bridge: submit producer prefix, CPU wait for a SIGNALED read fence,
// exchange the SAME surface, submit output, then resume the original suffix.
// Caller invokes construction/recording/submission OUTSIDE the session guard,
// under InternalCommandScope. No queue Wait, GPU spin, or future-fence dependency.
class Surface {
    ComPtr<ID3D12Device> device_;ComPtr<ID3D12Resource> target_,read_,upload_,verify_;
    ComPtr<ID3D12CommandAllocator> readAlloc_,writeAlloc_;
    ComPtr<ID3D12GraphicsCommandList> reader_,writer_;
    ComPtr<ID3D12Fence> fence_;HANDLE event_=nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_{};UINT width_=0,height_=0;UINT64 bytes_=0;
    uint64_t fenceValue_=0;bool uncertain_=false;
    static constexpr auto Readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    static void Transition(ID3D12GraphicsCommandList* list,ID3D12Resource* target,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to){
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={target,0,from,to};list->ResourceBarrier(1,&b);}
    void WaitSubmitted(ID3D12CommandQueue* q){
        const auto value=++fenceValue_;Check(q->Signal(fence_.Get(),value));
        Check(fence_->SetEventOnCompletion(value,event_));
        if(WaitForSingleObject(event_,2000)!=WAIT_OBJECT_0)throw std::runtime_error("surface GPU completion timeout; retained");
        Check(device_->GetDeviceRemovedReason());
        if(fence_->GetCompletedValue()==UINT64_MAX||fence_->GetCompletedValue()<value)throw std::runtime_error("surface fence invalid");
        uncertain_=false;
    }
    std::vector<uint8_t> Read(ID3D12Resource* buffer){
        void* p=nullptr;D3D12_RANGE range{0,size_t(bytes_)};Check(buffer->Map(0,&range,&p));
        std::vector<uint8_t> raw(size_t(width_)*height_*8);
        for(UINT y=0;y<height_;++y)std::memcpy(raw.data()+size_t(y)*width_*8,static_cast<uint8_t*>(p)+fp_.Offset+size_t(y)*fp_.Footprint.RowPitch,size_t(width_)*8);
        D3D12_RANGE none{};buffer->Unmap(0,&none);return raw;
    }
public:
    Surface(ID3D12Device* d,ID3D12Resource* target):device_(d),target_(target){
        if(!d||!target)throw std::invalid_argument("surface null");
        ComPtr<ID3D12Device> owner;Check(target->GetDevice(IID_PPV_ARGS(&owner)));if(!ffx_capture::Same(owner.Get(),d))throw std::invalid_argument("surface device");
        auto desc=target->GetDesc();
        if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||desc.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT||!desc.Width||!desc.Height||desc.Width>8192||desc.Height>8192||desc.MipLevels!=1||desc.DepthOrArraySize!=1||desc.SampleDesc.Count!=1||
           (desc.Flags&D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)||desc.Width*desc.Height*8>MaxBytes)throw std::invalid_argument("surface unsupported shape");
        width_=UINT(desc.Width);height_=desc.Height;UINT rows;UINT64 rowBytes;
        d->GetCopyableFootprints(&desc,0,1,0,&fp_,&rows,&rowBytes,&bytes_);
        if(rows!=height_||rowBytes!=UINT64(width_)*8||bytes_>MaxBytes+8192*256)throw std::invalid_argument("surface footprint");
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes_;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_READBACK;
        Check(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&read_)));
        Check(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&verify_)));
        heap.Type=D3D12_HEAP_TYPE_UPLOAD;Check(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload_)));
        Check(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&readAlloc_)));
        Check(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&writeAlloc_)));
        Check(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,readAlloc_.Get(),nullptr,IID_PPV_ARGS(&reader_)));
        Check(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,writeAlloc_.Get(),nullptr,IID_PPV_ARGS(&writer_)));
        D3D12_TEXTURE_COPY_LOCATION texture{},buffer{};texture.pResource=target;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buffer.pResource=read_.Get();buffer.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;buffer.PlacedFootprint=fp_;
        Transition(reader_.Get(),target,Readable,D3D12_RESOURCE_STATE_COPY_SOURCE);reader_->CopyTextureRegion(&buffer,0,0,0,&texture,nullptr);Transition(reader_.Get(),target,D3D12_RESOURCE_STATE_COPY_SOURCE,Readable);Check(reader_->Close());
        buffer.pResource=upload_.Get();Transition(writer_.Get(),target,Readable,D3D12_RESOURCE_STATE_COPY_DEST);writer_->CopyTextureRegion(&texture,0,0,0,&buffer,nullptr);
        Transition(writer_.Get(),target,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);buffer.pResource=verify_.Get();writer_->CopyTextureRegion(&buffer,0,0,0,&texture,nullptr);Transition(writer_.Get(),target,D3D12_RESOURCE_STATE_COPY_SOURCE,Readable);Check(writer_->Close());
        Check(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)));event_=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event_)throw std::runtime_error("surface event");
    }
    ~Surface(){if(uncertain_)std::terminate();if(event_)CloseHandle(event_);}
    bool SafeToDestroy()const{return !uncertain_;}
    struct Result {std::vector<uint8_t> input,output;bool exact=false;UINT width=0,height=0;};
    Result Run(ID3D12CommandQueue* queue,Submit original,Exchange& exchange){
        if(uncertain_||!queue||!original||queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)throw std::invalid_argument("surface queue");
        ComPtr<ID3D12Device> owner;Check(queue->GetDevice(IID_PPV_ARGS(&owner)));if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("surface queue device");
        ID3D12CommandList* lists[]={reader_.Get()};uncertain_=true;original(queue,1,lists);WaitSubmitted(queue);
        Result result;result.width=width_;result.height=height_;result.input=Read(read_.Get());
        auto output=exchange.Run(result.input,width_,height_);
        void* p=nullptr;D3D12_RANGE none{};Check(upload_->Map(0,&none,&p));
        for(UINT y=0;y<height_;++y)std::memcpy(static_cast<uint8_t*>(p)+fp_.Offset+size_t(y)*fp_.Footprint.RowPitch,output.data()+size_t(y)*width_*8,size_t(width_)*8);
        upload_->Unmap(0,nullptr);lists[0]=writer_.Get();uncertain_=true;original(queue,1,lists);WaitSubmitted(queue);
        result.output=Read(verify_.Get());result.exact=result.output==output;
        if(!result.exact)throw std::runtime_error("surface output readback differs from worker");
        return result;
    }
};
}
