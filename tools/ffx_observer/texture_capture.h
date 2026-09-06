#pragma once
// Bounded, caller-coordinated collector. This is NOT a COM/IAT hook.
// Caller must attest live COM pointers, legacy resource states and a single
// direct queue; record before/after the original FFX call, submit the list ONCE,
// then call Submitted before resetting/reusing it. Poll performs disk IO only
// after a collector-owned queue fence, never from inside the recording callback.
#include <d3d12.h>
#include <wrl/client.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale>
#include <cmath>
#include <cstring>
#include <stdexcept>
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"

namespace ffx_capture {
using Microsoft::WRL::ComPtr;
inline void Check(HRESULT h) { if(FAILED(h))throw std::runtime_error("capture D3D12 operation failed"); }
inline bool Same(IUnknown* a,IUnknown* b) {
    ComPtr<IUnknown> x,y;Check(a->QueryInterface(IID_PPV_ARGS(&x)));Check(b->QueryInterface(IID_PPV_ARGS(&y)));return x.Get()==y.Get();
}
enum class DepthPlanePolicy { Reject, D32S8DepthOnly };
class Collector {
    struct Surface {
        const char* role=nullptr;ComPtr<ID3D12Resource> source,readback;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows=0;UINT64 rowBytes=0,bytes=0;
        D3D12_RESOURCE_STATES state{};FfxApiResource metadata{};
        DXGI_FORMAT sourceFormat{};UINT planeCount=1;
    };
    struct Job {
        uint64_t ticket=0,frame=0,fence=0;bool after=false;
        ComPtr<ID3D12GraphicsCommandList> list;std::vector<Surface> surfaces;
        ffxDispatchDescUpscale metadata{};
    };
    ComPtr<ID3D12Device> device_;ComPtr<ID3D12Fence> fence_;ComPtr<ID3D12CommandQueue> queue_;
    std::filesystem::path root_;std::vector<Job> jobs_;
    uint64_t maxFrames_,maxBytes_,accepted_=0,dropped_=0,completed_=0,totalBytes_=0,nextFence_=0;
    ffxCreateContextDescUpscale context_{};uint64_t providerId_=0;
    DepthPlanePolicy depthPolicy_;
    bool gameFrame_=false;
    static D3D12_RESOURCE_STATES State(uint32_t s) {
        switch(s) {
        case FFX_API_RESOURCE_STATE_COMPUTE_READ:return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ:return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case FFX_API_RESOURCE_STATE_UNORDERED_ACCESS:return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        default:throw std::invalid_argument("capture unsupported/unverified resource state");
        }
    }
    static void Transition(ID3D12GraphicsCommandList* l,ID3D12Resource* r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER x{};x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        // Single mip/array only: capture plane 0, never transition stencil plane 1.
        x.Transition={r,0,a,b};l->ResourceBarrier(1,&x);
    }
    static void Copy(ID3D12GraphicsCommandList* l,const Surface& s) {
        Transition(l,s.source.Get(),s.state,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=s.source.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource=s.readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=s.fp;
        l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);Transition(l,s.source.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,s.state);
    }
    Surface Validate(const char* role,const FfxApiResource& r) {
        Surface s;s.role=role;s.source=static_cast<ID3D12Resource*>(r.resource);s.metadata=r;s.state=State(r.state);
        ComPtr<ID3D12Device> owner;Check(s.source->GetDevice(IID_PPV_ARGS(&owner)));
        if(!Same(owner.Get(),device_.Get()))throw std::invalid_argument("capture wrong resource device");
        auto d=s.source->GetDesc();s.sourceFormat=d.Format;
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.DepthOrArraySize!=1||d.MipLevels!=1||d.SampleDesc.Count!=1||!d.Width||!d.Height||d.Width>8192||d.Height>8192)
            throw std::invalid_argument("capture unsupported texture shape");
        switch(d.Format) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:case DXGI_FORMAT_R32_FLOAT:case DXGI_FORMAT_R16G16_FLOAT:
        case DXGI_FORMAT_R32G32_FLOAT:case DXGI_FORMAT_R8_UNORM:break;
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            if(depthPolicy_!=DepthPlanePolicy::D32S8DepthOnly||std::strcmp(role,"depth")||
               (r.state!=FFX_API_RESOURCE_STATE_COMPUTE_READ&&r.state!=FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ))
                throw std::invalid_argument("capture requires explicit depth-plane-only read policy");
            s.planeCount=2;break;
        default:throw std::invalid_argument("capture unsupported texture format");
        }
        auto actual=ffxApiGetResourceDX12(s.source.Get(),r.state);
        // GoWR binds the depth view of D32S8 and omits stencil usage. This is the
        // ONLY permitted metadata normalization; all other fields remain exact.
        if(s.planeCount==2&&r.description.usage==FFX_API_RESOURCE_USAGE_DEPTHTARGET)
            actual.description.usage=FFX_API_RESOURCE_USAGE_DEPTHTARGET;
        if(std::memcmp(&actual.description,&r.description,sizeof(r.description)))throw std::invalid_argument("capture resource descriptor mismatch");
        device_->GetCopyableFootprints(&d,0,1,0,&s.fp,&s.rows,&s.rowBytes,&s.bytes);
        if(s.planeCount==2&&(s.fp.Footprint.Format!=DXGI_FORMAT_R32_TYPELESS||s.rowBytes!=d.Width*4||s.rows!=d.Height))
            throw std::invalid_argument("capture unexpected depth-plane footprint");
        if(!s.bytes||s.bytes>maxBytes_)throw std::invalid_argument("capture resource byte cap");
        return s;
    }
    void Allocate(Surface& s) {
        D3D12_HEAP_PROPERTIES h{};h.Type=D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=s.bytes;d.Height=1;
        d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Check(device_->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&s.readback)));
    }
    Job& Find(uint64_t ticket) {
        for(auto& j:jobs_)if(j.ticket==ticket)return j;throw std::invalid_argument("capture unknown ticket");
    }
    static void WriteNew(const std::filesystem::path& p,const void* data,size_t bytes) {
        HANDLE file=CreateFileW(p.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE)throw std::runtime_error("capture file already exists or cannot be created");
        DWORD written=0;BOOL ok=WriteFile(file,data,DWORD(bytes),&written,nullptr);CloseHandle(file);
        if(!ok||written!=bytes)throw std::runtime_error("capture file write failed");
    }
    void Save(const Job& j) {
        auto dir=root_/std::to_string(j.frame);
        if(!std::filesystem::create_directory(dir))throw std::runtime_error("capture frame directory already exists");
        std::ostringstream json;json.imbue(std::locale::classic());json<<std::setprecision(9);const auto& d=j.metadata;
        json<<"{\"frame\":"<<j.frame<<",\"fence_value\":"<<j.fence<<",\"fence_completed\":true,\"context_create_flags\":"<<context_.flags
            <<",\"requested_provider_id\":"<<providerId_<<",\"provider_selection_verified\":"<<(providerId_?"true":"false")<<",\"context_max_render\":["<<context_.maxRenderSize.width<<','<<context_.maxRenderSize.height
            <<"],\"context_max_upscale\":["<<context_.maxUpscaleSize.width<<','<<context_.maxUpscaleSize.height<<"],\"render_size\":["<<d.renderSize.width<<','<<d.renderSize.height
            <<"],\"upscale_size\":["<<d.upscaleSize.width<<','<<d.upscaleSize.height
            <<"],\"effective_upscale_size\":["<<(d.upscaleSize.width?d.upscaleSize.width:context_.maxUpscaleSize.width)<<','<<(d.upscaleSize.height?d.upscaleSize.height:context_.maxUpscaleSize.height)
            <<"],\"jitter\":["<<d.jitterOffset.x<<','<<d.jitterOffset.y
            <<"],\"motion_scale\":["<<d.motionVectorScale.x<<','<<d.motionVectorScale.y<<"],\"reset\":"<<(d.reset?"true":"false")
            <<",\"pre_exposure\":"<<d.preExposure<<",\"frame_time_ms\":"<<d.frameTimeDelta<<",\"camera_near\":"<<d.cameraNear
            <<",\"camera_far\":"<<d.cameraFar<<",\"camera_fov_y\":"<<d.cameraFovAngleVertical<<",\"view_to_meters\":"<<d.viewSpaceToMetersFactor
            <<",\"dispatch_flags\":"<<d.flags<<",\"enable_sharpening\":"<<(d.enableSharpening?"true":"false")<<",\"sharpness\":"<<d.sharpness<<",\"resources\":[";
        bool first=true;
        for(const auto& s:j.surfaces) {
            std::vector<uint8_t> raw(size_t(s.rows*s.rowBytes));void* mapped=nullptr;D3D12_RANGE range{0,size_t(s.bytes)};
            Check(s.readback->Map(0,&range,&mapped));
            for(UINT y=0;y<s.rows;++y)std::memcpy(raw.data()+y*s.rowBytes,static_cast<uint8_t*>(mapped)+s.fp.Offset+y*s.fp.Footprint.RowPitch,size_t(s.rowBytes));
            D3D12_RANGE none{0,0};s.readback->Unmap(0,&none);
            WriteNew(dir/(std::string(s.role)+".raw"),raw.data(),raw.size());
            if(!first)json<<',';first=false;
            json<<"{\"role\":\""<<s.role<<"\",\"width\":"<<s.fp.Footprint.Width<<",\"height\":"<<s.rows<<",\"dxgi_format\":"<<s.fp.Footprint.Format
                <<",\"ffx_format\":"<<s.metadata.description.format<<",\"ffx_state_restored\":"<<s.metadata.state
                <<",\"source_dxgi_format\":"<<s.sourceFormat<<",\"plane_count\":"<<s.planeCount<<",\"copied_subresource\":0,\"barrier_subresource\":0"
                <<",\"row_pitch\":"<<s.fp.Footprint.RowPitch<<",\"row_bytes\":"<<s.rowBytes<<",\"raw_bytes\":"<<raw.size()<<'}';
        }
        json<<"],\"game_frame\":"<<(gameFrame_?"true":"false")<<",\"caller_validated_synthetic_host\":"<<(gameFrame_?"false":"true")<<"}\n";
        const auto text=json.str();WriteNew(dir/"manifest.json",text.data(),text.size());
    }
public:
    Collector(ID3D12Device* device,const std::filesystem::path& root,const ffxCreateContextDescUpscale& context,uint64_t providerId,
              uint64_t maxFrames=3,uint64_t maxBytes=64*1024*1024,DepthPlanePolicy depthPolicy=DepthPlanePolicy::Reject,bool gameFrame=false)
        :device_(device),root_(root),maxFrames_(maxFrames),maxBytes_(maxBytes),context_(context),providerId_(providerId),depthPolicy_(depthPolicy),gameFrame_(gameFrame) {
        if(!device||!root.is_absolute()||!maxFrames||maxFrames>32||!maxBytes||maxBytes>512*1024*1024)throw std::invalid_argument("capture invalid configuration");
        if(context.header.type!=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE||(!providerId&&!gameFrame))throw std::invalid_argument("capture missing context contract");
        for(auto n:{context.maxRenderSize.width,context.maxRenderSize.height,context.maxUpscaleSize.width,context.maxUpscaleSize.height})
            if(!n||n>8192)throw std::invalid_argument("capture invalid context dimensions");
        if(!std::filesystem::create_directory(root))throw std::invalid_argument("capture directory must be new");
        Check(device_->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)));
    }
    Collector(const Collector&)=delete;Collector& operator=(const Collector&)=delete;
    ~Collector() {
        // Abandonment is not a drain. Retain referenced objects to process exit,
        // even for recorded-but-not-submitted lists, rather than cause GPU UAF.
        if(!jobs_.empty()) {
            for(auto& j:jobs_){j.list.Detach();for(auto& s:j.surfaces){s.source.Detach();s.readback.Detach();}}
            fence_.Detach();queue_.Detach();device_.Detach();
        }
    }
    uint64_t Before(const ffxDispatchDescUpscale& d,uint64_t frame) {
        if(accepted_>=maxFrames_){++dropped_;return 0;}
        if(d.header.type!=FFX_API_DISPATCH_DESC_TYPE_UPSCALE||d.header.pNext||!d.commandList||!d.color.resource||!d.depth.resource||!d.motionVectors.resource||!d.output.resource)
            throw std::invalid_argument("capture unsupported dispatch contract");
        for(float f:{d.jitterOffset.x,d.jitterOffset.y,d.motionVectorScale.x,d.motionVectorScale.y,d.preExposure,d.frameTimeDelta,d.cameraNear,d.cameraFar,d.cameraFovAngleVertical,d.viewSpaceToMetersFactor,d.sharpness})
            if(!std::isfinite(f))throw std::invalid_argument("capture nonfinite metadata");
        if(reinterpret_cast<const unsigned char*>(&d.reset)[0]>1||reinterpret_cast<const unsigned char*>(&d.enableSharpening)[0]>1)
            throw std::invalid_argument("capture invalid bool encoding");
        if(!d.renderSize.width||!d.renderSize.height||bool(d.upscaleSize.width)!=bool(d.upscaleSize.height))throw std::invalid_argument("capture missing/mixed dimensions");
        auto effective=d.upscaleSize;if(!effective.width)effective=context_.maxUpscaleSize;
        if(d.renderSize.width>context_.maxRenderSize.width||d.renderSize.height>context_.maxRenderSize.height||
           effective.width>context_.maxUpscaleSize.width||effective.height>context_.maxUpscaleSize.height)
            throw std::invalid_argument("capture stale or exceeded context dimensions");
        if(d.renderSize.width>d.color.description.width||d.renderSize.height>d.color.description.height||
           d.renderSize.width>d.depth.description.width||d.renderSize.height>d.depth.description.height||
           effective.width>d.output.description.width||effective.height>d.output.description.height)
            throw std::invalid_argument("capture dimensions exceed textures");
        Job j;j.frame=frame;j.ticket=accepted_+1;j.metadata=d;j.list=static_cast<ID3D12GraphicsCommandList*>(d.commandList);
        if(j.list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT)throw std::invalid_argument("capture requires direct command list");
        ComPtr<ID3D12Device> owner;Check(j.list->GetDevice(IID_PPV_ARGS(&owner)));if(!Same(owner.Get(),device_.Get()))throw std::invalid_argument("capture wrong list device");
        for(const auto& old:jobs_)if(old.frame==frame||old.list.Get()==j.list.Get())throw std::invalid_argument("capture duplicate pending list/frame");
        const char* names[]={"color","depth","motion","exposure","reactive","transparency","output"};
        const FfxApiResource* resources[]={&d.color,&d.depth,&d.motionVectors,&d.exposure,&d.reactive,&d.transparencyAndComposition,&d.output};
        uint64_t bytes=0;
        for(unsigned i=0;i<7;++i)if(resources[i]->resource) {
            if(i<6&&resources[i]->resource==d.output.resource)throw std::invalid_argument("capture aliased output/input");
            j.surfaces.push_back(Validate(names[i],*resources[i]));bytes+=j.surfaces.back().bytes;
        }
        if(bytes>maxBytes_-totalBytes_){++dropped_;return 0;}
        // Complete validation/allocation/storage BEFORE emitting any copy commands.
        for(auto& s:j.surfaces)Allocate(s);jobs_.push_back(std::move(j));auto& job=jobs_.back();
        ++accepted_;totalBytes_+=bytes;
        for(size_t i=0;i+1<job.surfaces.size();++i)Copy(job.list.Get(),job.surfaces[i]);
        return job.ticket;
    }
    void After(uint64_t ticket,ID3D12GraphicsCommandList* list) {
        if(!ticket)return;auto& j=Find(ticket);
        if(j.after||j.fence||j.list.Get()!=list)throw std::invalid_argument("capture wrong/duplicate after");
        Copy(list,j.surfaces.back());j.after=true;
    }
    void Submitted(uint64_t ticket,ID3D12CommandQueue* queue,ID3D12CommandList* list) {
        if(!ticket)return;auto& j=Find(ticket);
        if(!j.after||j.fence||!queue||j.list.Get()!=list||queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)
            throw std::invalid_argument("capture invalid submission notification");
        ComPtr<ID3D12Device> owner;Check(queue->GetDevice(IID_PPV_ARGS(&owner)));
        if(!Same(owner.Get(),device_.Get())||(queue_&&!Same(queue_.Get(),queue)))throw std::invalid_argument("capture queue ownership mismatch");
        queue_=queue;const uint64_t value=++nextFence_;Check(queue->Signal(fence_.Get(),value));j.fence=value;
    }
    uint64_t Poll() {
        uint64_t value=fence_->GetCompletedValue();if(value==UINT64_MAX)throw std::runtime_error("capture device removed");
        uint64_t saved=0;
        for(auto it=jobs_.begin();it!=jobs_.end();) {
            if(it->fence&&it->fence<=value){Save(*it);it=jobs_.erase(it);++saved;++completed_;}else ++it;
        }return saved;
    }
    uint64_t Accepted()const{return accepted_;}uint64_t Dropped()const{return dropped_;}
    uint64_t Completed()const{return completed_;}uint64_t Pending()const{return jobs_.size();}
};
}
