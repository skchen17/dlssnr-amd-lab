#pragma once
// Immutable offline pixels. No inference, GPU waits, allocator resets or
// root-state changes. Live instances and uploads are retained until exit.
#include "texture_capture.h"
#include <fstream>
namespace ffx_boundary {
class StaticPreview {
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12Resource> input_,output_;
public:
    static constexpr UINT Width=644,Height=384,RowPitch=5376;
    static std::vector<uint8_t> Load(const std::filesystem::path& path) {
        std::ifstream f(path,std::ios::binary|std::ios::ate);
        if(!f||f.tellg()!=std::streamoff(Width*Height*8))throw std::invalid_argument("preview pixel size");
        std::vector<uint8_t> raw(Width*Height*8);f.seekg(0);
        if(!f.read(reinterpret_cast<char*>(raw.data()),raw.size()))throw std::invalid_argument("preview pixel read");
        for(size_t i=0;i<raw.size();i+=2){uint16_t h;std::memcpy(&h,raw.data()+i,2);
            if((h&0x7c00)==0x7c00)throw std::invalid_argument("preview nonfinite half");}
        return raw;
    }
    StaticPreview(ID3D12Device* device,const std::vector<uint8_t>& input,const std::vector<uint8_t>& output):device_(device){
        if(!device||input.size()!=Width*Height*8||output.size()!=input.size())throw std::invalid_argument("preview input");
        auto upload=[&](const std::vector<uint8_t>& raw,Microsoft::WRL::ComPtr<ID3D12Resource>& resource){
            D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=UINT64(RowPitch)*Height;
            b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            ffx_capture::Check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&resource)));
            void* mapped=nullptr;D3D12_RANGE none{};ffx_capture::Check(resource->Map(0,&none,&mapped));std::memset(mapped,0,size_t(b.Width));
            for(UINT y=0;y<Height;++y)std::memcpy(static_cast<uint8_t*>(mapped)+y*RowPitch,raw.data()+y*Width*8,Width*8);
            resource->Unmap(0,nullptr);
        };upload(input,input_);upload(output,output_);
    }
    // Stop only prevents future copies; it must not free in-flight uploads.
    void Record(ID3D12GraphicsCommandList* list,const D3D12_RESOURCE_BARRIER& b,bool network) {
        constexpr auto readable=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if(!list||list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT||b.Type!=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION||
           b.Flags!=D3D12_RESOURCE_BARRIER_FLAG_NONE||!b.Transition.pResource||
           (b.Transition.Subresource!=0&&b.Transition.Subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)||
           (b.Transition.StateBefore!=D3D12_RESOURCE_STATE_UNORDERED_ACCESS&&b.Transition.StateBefore!=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)||
           b.Transition.StateAfter!=readable)throw std::invalid_argument("preview boundary");
        auto* target=b.Transition.pResource;auto d=target->GetDesc();
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT||
           d.Width<Width||d.Height<Height||d.Width>8192||d.Height>8192||d.MipLevels!=1||d.DepthOrArraySize!=1||
           d.SampleDesc.Count!=1||(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS))throw std::invalid_argument("preview output shape");
        Microsoft::WRL::ComPtr<ID3D12Device> owner,listOwner;
        ffx_capture::Check(target->GetDevice(IID_PPV_ARGS(&owner)));ffx_capture::Check(list->GetDevice(IID_PPV_ARGS(&listOwner)));
        if(!ffx_capture::Same(owner.Get(),device_.Get())||!ffx_capture::Same(listOwner.Get(),device_.Get()))throw std::invalid_argument("preview device");
        D3D12_RESOURCE_BARRIER transition{};transition.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition={target,0,readable,D3D12_RESOURCE_STATE_COPY_DEST};list->ResourceBarrier(1,&transition);
        D3D12_TEXTURE_COPY_LOCATION source{},destination{};
        source.pResource=network?output_.Get():input_.Get();source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint.Footprint={DXGI_FORMAT_R16G16B16A16_FLOAT,Width,Height,1,RowPitch};
        destination.pResource=target;destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&destination,UINT((d.Width-Width)/2),(d.Height-Height)/2,0,&source,nullptr);
        transition.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;transition.Transition.StateAfter=readable;list->ResourceBarrier(1,&transition);
    }
};
}
