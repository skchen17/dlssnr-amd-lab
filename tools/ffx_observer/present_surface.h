#pragma once
// Diagnostic DISPLAY-REFERRED bridge. Not the NR pre-tonemap color contract.
// Every call reads the current swap-chain buffer, runs the real network, writes
// that same buffer, verifies it, restores PRESENT, then permits Present.
#include "texture_capture.h"
#include "resident_exchange.h"
#include <DirectXPackedVector.h>
#include <algorithm>

namespace nr_present {
using Microsoft::WRL::ComPtr;
using ffx_capture::Check;
inline bool Supported(DXGI_FORMAT f){return f==DXGI_FORMAT_R8G8B8A8_UNORM||f==DXGI_FORMAT_B8G8R8A8_UNORM||f==DXGI_FORMAT_R10G10B10A2_UNORM;}
inline std::vector<uint8_t> Decode(const std::vector<uint8_t>& raw,DXGI_FORMAT format){
    if(!Supported(format)||raw.empty()||raw.size()%4)throw std::invalid_argument("present decode format/size");
    std::vector<uint8_t> half(raw.size()*2);
    for(size_t i=0;i<raw.size()/4;++i){uint32_t p;std::memcpy(&p,raw.data()+i*4,4);float v[4];
        if(format==DXGI_FORMAT_R10G10B10A2_UNORM){v[0]=float(p&1023)/1023;v[1]=float((p>>10)&1023)/1023;v[2]=float((p>>20)&1023)/1023;v[3]=float(p>>30)/3;}
        else {v[0]=float(p&255)/255;v[1]=float((p>>8)&255)/255;v[2]=float((p>>16)&255)/255;v[3]=float(p>>24)/255;if(format==DXGI_FORMAT_B8G8R8A8_UNORM)std::swap(v[0],v[2]);}
        for(unsigned c=0;c<4;++c){auto h=DirectX::PackedVector::XMConvertFloatToHalf(v[c]);std::memcpy(half.data()+i*8+c*2,&h,2);}
    }return half;
}
inline std::vector<uint8_t> Encode(const std::vector<uint8_t>& half,const std::vector<uint8_t>& base,DXGI_FORMAT format){
    if(!Supported(format)||half.size()!=base.size()*2||base.empty()||base.size()%4)throw std::invalid_argument("present encode size/format");
    std::vector<uint8_t> raw(base.size());
    for(size_t i=0;i<base.size()/4;++i){float v[4];for(unsigned c=0;c<4;++c){uint16_t h;std::memcpy(&h,half.data()+i*8+c*2,2);v[c]=DirectX::PackedVector::XMConvertHalfToFloat(h);if(!std::isfinite(v[c]))throw std::invalid_argument("present encode nonfinite");}
        uint32_t previous;std::memcpy(&previous,base.data()+i*4,4);uint32_t p;
        const auto unit=[](float f,uint32_t m){return uint32_t(std::lround(std::clamp(f,0.0f,1.0f)*m));};
        if(format==DXGI_FORMAT_R10G10B10A2_UNORM)p=unit(v[0],1023)|(unit(v[1],1023)<<10)|(unit(v[2],1023)<<20)|(previous&0xc0000000);
        else {if(format==DXGI_FORMAT_B8G8R8A8_UNORM)std::swap(v[0],v[2]);p=unit(v[0],255)|(unit(v[1],255)<<8)|(unit(v[2],255)<<16)|(previous&0xff000000);}
        std::memcpy(raw.data()+i*4,&p,4);
    }return raw;
}
class Surface {
    ComPtr<ID3D12Device> device_;ComPtr<ID3D12Resource> target_,read_,upload_,verify_;
    ComPtr<ID3D12CommandAllocator> ra_,wa_;ComPtr<ID3D12GraphicsCommandList> reader_,writer_;
    ComPtr<ID3D12Fence> fence_;HANDLE event_=nullptr;uint64_t fenceValue_=0;bool uncertain_=false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_{};UINT w_=0,h_=0;UINT64 bytes_=0;DXGI_FORMAT format_{};
    static void Transition(ID3D12GraphicsCommandList* l,ID3D12Resource* r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){D3D12_RESOURCE_BARRIER x{};x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;x.Transition={r,0,a,b};l->ResourceBarrier(1,&x);}
    void Submit(ID3D12CommandQueue* q,ID3D12GraphicsCommandList* l){
        ID3D12CommandList* lists[]={l};uncertain_=true;q->ExecuteCommandLists(1,lists);
        const auto value=++fenceValue_;Check(q->Signal(fence_.Get(),value));Check(fence_->SetEventOnCompletion(value,event_));
        if(WaitForSingleObject(event_,2000)!=WAIT_OBJECT_0)throw std::runtime_error("present GPU timeout; retain uncertain resources");
        Check(device_->GetDeviceRemovedReason());auto completed=fence_->GetCompletedValue();if(completed==UINT64_MAX||completed<value)throw std::runtime_error("present fence invalid");uncertain_=false;
    }
    std::vector<uint8_t> Read(ID3D12Resource* r){void* p=nullptr;D3D12_RANGE range{0,size_t(bytes_)};Check(r->Map(0,&range,&p));std::vector<uint8_t> raw(size_t(w_)*h_*4);for(UINT y=0;y<h_;++y)std::memcpy(raw.data()+size_t(y)*w_*4,static_cast<uint8_t*>(p)+fp_.Offset+size_t(y)*fp_.Footprint.RowPitch,size_t(w_)*4);D3D12_RANGE none{};r->Unmap(0,&none);return raw;}
public:
    struct Result {std::vector<uint8_t> encodedInput,input,output,encodedOutput;UINT width=0,height=0;DXGI_FORMAT format{};};
    Surface(ID3D12Device* d,ID3D12Resource* r):device_(d),target_(r){
        if(!d||!r)throw std::invalid_argument("present resource null");ComPtr<ID3D12Device> owner;Check(r->GetDevice(IID_PPV_ARGS(&owner)));if(!ffx_capture::Same(d,owner.Get()))throw std::invalid_argument("present resource device");auto td=r->GetDesc();
        if(td.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||!Supported(td.Format)||!td.Width||!td.Height||td.Width>8192||td.Height>8192||td.Width*td.Height*8>ffx_resident::MaxBytes||td.MipLevels!=1||td.DepthOrArraySize!=1||td.SampleDesc.Count!=1)throw std::invalid_argument("present unsupported surface");
        w_=UINT(td.Width);h_=td.Height;format_=td.Format;UINT rows;UINT64 rowBytes;d->GetCopyableFootprints(&td,0,1,0,&fp_,&rows,&rowBytes,&bytes_);if(rows!=h_||rowBytes!=UINT64(w_)*4)throw std::invalid_argument("present footprint");
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes_;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_READBACK;
        Check(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&read_)));Check(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&verify_)));heap.Type=D3D12_HEAP_TYPE_UPLOAD;Check(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload_)));
        Check(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&ra_)));Check(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&wa_)));Check(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,ra_.Get(),nullptr,IID_PPV_ARGS(&reader_)));Check(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,wa_.Get(),nullptr,IID_PPV_ARGS(&writer_)));
        D3D12_TEXTURE_COPY_LOCATION texture{},buffer{};texture.pResource=r;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;buffer.pResource=read_.Get();buffer.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;buffer.PlacedFootprint=fp_;
        Transition(reader_.Get(),r,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_COPY_SOURCE);reader_->CopyTextureRegion(&buffer,0,0,0,&texture,nullptr);Transition(reader_.Get(),r,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_PRESENT);Check(reader_->Close());
        buffer.pResource=upload_.Get();Transition(writer_.Get(),r,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_COPY_DEST);writer_->CopyTextureRegion(&texture,0,0,0,&buffer,nullptr);Transition(writer_.Get(),r,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);buffer.pResource=verify_.Get();writer_->CopyTextureRegion(&buffer,0,0,0,&texture,nullptr);Transition(writer_.Get(),r,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_PRESENT);Check(writer_->Close());
        Check(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)));event_=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event_)throw std::runtime_error("present event");
    }
    ~Surface(){if(uncertain_)std::terminate();if(event_)CloseHandle(event_);}
    bool SafeToDestroy()const{return !uncertain_;}
    Result Run(ID3D12CommandQueue* q,ffx_resident::Exchange& exchange){
        if(!q||uncertain_||q->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)throw std::invalid_argument("present queue");ComPtr<ID3D12Device> owner;Check(q->GetDevice(IID_PPV_ARGS(&owner)));if(!ffx_capture::Same(owner.Get(),device_.Get()))throw std::invalid_argument("present queue device");
        Submit(q,reader_.Get());Result r;r.width=w_;r.height=h_;r.format=format_;r.encodedInput=Read(read_.Get());r.input=Decode(r.encodedInput,format_);r.output=exchange.Run(r.input,w_,h_);auto encoded=Encode(r.output,r.encodedInput,format_);
        void* p=nullptr;D3D12_RANGE none{};Check(upload_->Map(0,&none,&p));for(UINT y=0;y<h_;++y)std::memcpy(static_cast<uint8_t*>(p)+fp_.Offset+size_t(y)*fp_.Footprint.RowPitch,encoded.data()+size_t(y)*w_*4,size_t(w_)*4);upload_->Unmap(0,nullptr);Submit(q,writer_.Get());r.encodedOutput=Read(verify_.Get());if(r.encodedOutput!=encoded)throw std::runtime_error("present readback differs");return r;
    }
};
}
