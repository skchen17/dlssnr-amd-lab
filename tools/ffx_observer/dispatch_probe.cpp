// Independent real-provider test. No game process, IAT hooks or deployment.
#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"
#include "texture_capture.h"
#include "live_inspector.h"
#include "capture_session.h"
#include "output_boundary_copy.h"
#include "output_boundary_roundtrip.h"
#include "output_boundary_patch.h"
#include <memory>

namespace {
UINT RW=320, RH=180, OW=640, OH=360;
void Write(const fs::path& p,const std::vector<uint8_t>& b) { Write(p,b.data(),b.size()); }
void Api(uint32_t result, const char* name) {
    if(result) throw std::runtime_error(std::string(name)+" returned "+std::to_string(result));
}
void Message(uint32_t type,const wchar_t* message) {
    std::fwprintf(stderr,L"FFX[%u] %ls\n",type,message);
}
ComPtr<ID3D12Resource> Tex(ID3D12Device* d,UINT w,UINT h,DXGI_FORMAT fmt,bool output=false) {
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=w;desc.Height=h;desc.DepthOrArraySize=1;desc.MipLevels=1;
    desc.Format=fmt;desc.SampleDesc.Count=1;
    if(output)desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> r;
    HR_CHECK(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
        output?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&r)));
    return r;
}
struct Layout {D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows=0;UINT64 rowBytes=0,bytes=0;};
Layout Footprint(ID3D12Device* d,ID3D12Resource* r,UINT plane=0) {
    Layout x;auto desc=r->GetDesc();d->GetCopyableFootprints(&desc,plane,1,0,&x.fp,&x.rows,&x.rowBytes,&x.bytes);return x;
}
D3D12_TEXTURE_COPY_LOCATION Linear(ID3D12Resource* r,const Layout& x) {
    D3D12_TEXTURE_COPY_LOCATION l{};l.pResource=r;l.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;l.PlacedFootprint=x.fp;return l;
}
ComPtr<ID3D12Resource> UploadTex(ID3D12Device* d,ID3D12GraphicsCommandList* l,ID3D12Resource* r,const std::vector<uint8_t>& raw) {
    auto fp=Footprint(d,r);if(raw.size()!=fp.rows*fp.rowBytes)throw std::runtime_error("input size mismatch");
    std::vector<uint8_t> packed(size_t(fp.bytes),0);
    for(UINT y=0;y<fp.rows;++y)std::memcpy(packed.data()+fp.fp.Offset+y*fp.fp.Footprint.RowPitch,raw.data()+y*fp.rowBytes,size_t(fp.rowBytes));
    auto up=Upload(d,packed);auto src=Linear(up.Get(),fp),dst=Recorder::TextureLocation(r);
    l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    Recorder::Transition(l,r,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return up;
}
std::vector<uint8_t> Unpack(ID3D12Resource* r,const Layout& fp) {
    auto packed=ReadGpu(r,size_t(fp.bytes));std::vector<uint8_t> raw(size_t(fp.rows*fp.rowBytes));
    for(UINT y=0;y<fp.rows;++y)std::memcpy(raw.data()+y*fp.rowBytes,packed.data()+fp.fp.Offset+y*fp.fp.Footprint.RowPitch,size_t(fp.rowBytes));
    return raw;
}
std::vector<uint8_t> Color(unsigned variant) {
    std::vector<uint8_t> b(RW*RH*8);auto* h=reinterpret_cast<uint16_t*>(b.data());
    for(UINT y=0;y<RH;++y)for(UINT x=0;x<RW;++x) {
        size_t p=(y*RW+x)*4;
        h[p]=UnitHalf(variant?.8f-.5f*float(x)/RW:.1f+.6f*float(x)/RW);
        h[p+1]=UnitHalf(.15f+.5f*float(y)/RH);
        h[p+2]=UnitHalf(((x/16+y/16+variant)%2)?.7f:.2f);h[p+3]=0x3c00;
    }return b;
}
}
int wmain(int argc,wchar_t** argv) {
    std::wstring mode=argc==5?argv[4]:L"";
    const bool boundaryMode=mode==L"output-boundary";
    const bool roundTripMode=mode==L"output-roundtrip";
    const bool windowSession=mode==L"session-window";
    const bool adversarialSession=mode==L"session-boundary-negative";
    const bool boundaryHookSession=mode==L"session-output-boundary";
    const bool roundTripHookSession=mode==L"session-output-roundtrip";
    const bool patchHookSession=mode==L"session-output-patch";
    const bool filterHookSession=mode==L"session-output-filter";
    const bool networkHookSession=mode==L"session-output-network";
    const bool observeSession=mode==L"session-observe"||windowSession||adversarialSession||boundaryHookSession||roundTripHookSession||patchHookSession||filterHookSession||networkHookSession;
    if(observeSession)mode=L"session";
    if(argc!=4 && !(argc==5&&(boundaryMode||roundTripMode||mode==L"capture"||mode==L"inspect"||mode==L"contract"||mode==L"session"||mode==L"depth-base"||mode==L"depth-capture"||mode==L"depth-aligned-base"||mode==L"depth-aligned-capture"))){std::fprintf(stderr,"usage: ffx_dispatch_probe <verified AMD DLL> <exact provider name> <new result directory> [output-boundary|output-roundtrip|capture|inspect|contract|session|depth-base|depth-capture|depth-aligned-base|depth-aligned-capture]\n");return 2;}
    const bool alignedMode=mode==L"depth-aligned-base"||mode==L"depth-aligned-capture";
    const bool depthMode=mode==L"depth-base"||mode==L"depth-capture"||alignedMode;
    const bool captureMode=mode==L"capture"||mode==L"depth-capture"||mode==L"depth-aligned-capture";
    const bool inspectMode=mode==L"inspect"||mode==L"contract";
    const bool sessionMode=mode==L"session";
    if(depthMode){RW=1552;RH=872;OW=2342;OH=1317;}
    if(alignedMode){RW=1536;RH=864;OW=2304;OH=1296;}
    fs::path result=fs::absolute(argv[3]);
    if(fs::exists(result)){std::fprintf(stderr,"result directory already exists\n");return 2;}
    fs::create_directories(result);std::ofstream journal(result/"stages.log");
    auto stage=[&](const std::string& s){journal<<s<<std::endl;std::printf("%s\n",s.c_str());std::fflush(stdout);};
    try {
        std::wstring requestedW=argv[2];std::string requested(requestedW.begin(),requestedW.end());
        // No arbitrary quoted/control-character names in the JSON manifest.
        if(requested.find_first_not_of("0123456789. *")!=std::string::npos)throw std::invalid_argument("unexpected provider name");
        auto library=fs::absolute(argv[1]);
        HMODULE backend=LoadLibraryExW(library.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!backend)throw std::runtime_error("provider load failed");
        auto create=reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(backend,"ffxCreateContext"));
        auto destroy=reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(backend,"ffxDestroyContext"));
        auto query=reinterpret_cast<PfnFfxQuery>(GetProcAddress(backend,"ffxQuery"));
        auto dispatch=reinterpret_cast<PfnFfxDispatch>(GetProcAddress(backend,"ffxDispatch"));
        if(!create||!destroy||!query||!dispatch)throw std::runtime_error("required exports absent");
        DWORD(WINAPI* sessionWatch)(void*)=nullptr;DWORD(WINAPI* sessionCommand)(void*)=nullptr;DWORD sessionPoll=0;
        unsigned windowIdleFrames=0,windowTriggers=0;bool windowCapRejected=false,windowBusyRejected=false;
        if(sessionMode) {
            wchar_t executable[32768]{};if(!GetModuleFileNameW(nullptr,executable,32768))throw std::runtime_error("executable path");
            auto library=fs::path(executable).parent_path()/"ffx_capture_session.dll";
            HMODULE session=LoadLibraryExW(library.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);if(!session)throw std::runtime_error("session load");
            auto init=reinterpret_cast<DWORD(WINAPI*)(void*)>(GetProcAddress(session,"FfxSession_TestInitialize"));
            FfxSessionTestConfigV1 c{};c.common.size=sizeof(c.common);c.common.version=1;c.common.sample_limit=64;c.common.capture_enabled=1;
            if(observeSession)c.common.capture_enabled=0;
            if(windowSession)c.common.version=2;
            if(boundaryHookSession||roundTripHookSession||patchHookSession||filterHookSession||networkHookSession)c.common.version=2;
            wcscpy_s(c.common.log_path,(result/L"session.jsonl").c_str());c.original_create=reinterpret_cast<uintptr_t>(create);c.original_destroy=reinterpret_cast<uintptr_t>(destroy);c.original_dispatch=reinterpret_cast<uintptr_t>(dispatch);
            if(!init||init(&c)!=1)throw std::runtime_error("session initialization");
            create=reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(session,"FfxSession_TestCreate"));destroy=reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(session,"FfxSession_TestDestroy"));dispatch=reinterpret_cast<PfnFfxDispatch>(GetProcAddress(session,"FfxSession_TestDispatch"));
            sessionWatch=reinterpret_cast<DWORD(WINAPI*)(void*)>(GetProcAddress(session,"FfxSession_TestWatch"));sessionCommand=reinterpret_cast<DWORD(WINAPI*)(void*)>(GetProcAddress(session,"FfxSession_Command"));
            if(!create||!destroy||!dispatch||!sessionWatch||!sessionCommand)throw std::runtime_error("session exports");
            if(!observeSession)for(uint32_t action:{3u,4u}){FfxSessionCommandV1 command{sizeof(command),action};if(sessionCommand(&command)!=0)throw std::runtime_error("observation canceled armed capture");}
        }
        if(inspectMode) {
            wchar_t executable[32768]{};
            if(!GetModuleFileNameW(nullptr,executable,32768))throw std::runtime_error("executable path");
            auto inspectorPath=fs::path(executable).parent_path()/(mode==L"contract"?"ffx_contract_inspector.dll":"ffx_live_inspector.dll");
            auto inspector=LoadLibraryExW(inspectorPath.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
            if(!inspector)throw std::runtime_error("inspector load");
            auto initialize=reinterpret_cast<DWORD(WINAPI*)(void*)>(GetProcAddress(inspector,"FfxLive_TestInitialize"));
            auto observed=reinterpret_cast<PfnFfxDispatch>(GetProcAddress(inspector,"FfxLive_TestDispatch"));
            FfxLiveTestConfigV1 config{};config.common.size=sizeof(config.common);config.common.version=1;config.common.sample_limit=4;
            wcscpy_s(config.common.log_path,(result/L"live.jsonl").c_str());config.original_dispatch=reinterpret_cast<uintptr_t>(dispatch);
            if(!initialize||!observed||initialize(&config)!=1)throw std::runtime_error("inspector initialization");
            dispatch=observed;
        }
        ComPtr<ID3D12Debug> debug;HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 ad{};bool found=false;
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
            HR_CHECK(adapter->GetDesc1(&ad));if(!(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&ad.VendorId==0x1002){found=true;break;}
        }if(!found)throw std::runtime_error("AMD adapter absent");
        ComPtr<ID3D12Device> d;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)));
        ComPtr<ID3D12InfoQueue> info;HR_CHECK(d.As(&info));
        std::unique_ptr<ffx_capture::Collector> capture;
        unsigned captureRejected=0,earlyPolls=0,depthRejected=0;bool byteBudgetRejected=false;uint64_t stencilMismatches=0;
        unsigned boundaryCompleted=0;bool boundaryExact=true;
        uint64_t count=0;ffxQueryDescGetVersions v{};v.header.type=FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        v.createDescType=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;v.device=d.Get();v.outputCount=&count;
        Api(query(nullptr,&v.header),"query count");if(!count||count>32)throw std::runtime_error("invalid provider count");
        std::vector<uint64_t> ids(count);std::vector<const char*> names(count);v.versionIds=ids.data();v.versionNames=names.data();
        Api(query(nullptr,&v.header),"query versions");if(count>ids.size())throw std::runtime_error("provider capacity exceeded");
        bool matched=false;uint64_t selected=0;
        for(size_t i=0;i<count;++i)if(names[i]&&requested==names[i]){if(matched)throw std::runtime_error("ambiguous provider");matched=true;selected=ids[i];}
        if(!matched)throw std::runtime_error("requested provider not enumerated");
        stage("provider "+requested+" id "+std::to_string(selected));
        D3D12_COMMAND_QUEUE_DESC qd{};ComPtr<ID3D12CommandQueue> queue;HR_CHECK(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12Fence> fence;HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        ffxOverrideVersion version{};version.header.type=FFX_API_DESC_TYPE_OVERRIDE_VERSION;version.versionId=selected;
        ffxCreateBackendDX12Desc dx{};dx.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;dx.header.pNext=&version.header;dx.device=d.Get();
        ffxCreateContextDescUpscale desc{};desc.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;desc.header.pNext=&dx.header;
        desc.flags=FFX_UPSCALE_ENABLE_DEBUG_CHECKING;desc.maxRenderSize={RW,RH};desc.maxUpscaleSize={OW,OH};desc.fpMessage=Message;
        if(depthMode)desc.flags|=FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE|FFX_UPSCALE_ENABLE_DEPTH_INVERTED|FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        ffxContext ctx=nullptr;stage("create begin");Api(create(&ctx,&desc.header,nullptr),"create");if(!ctx)throw std::runtime_error("null context");stage("create OK");
        const auto depthPolicy=depthMode?ffx_capture::DepthPlanePolicy::D32S8DepthOnly:ffx_capture::DepthPlanePolicy::Reject;
        if(captureMode)capture=std::make_unique<ffx_capture::Collector>(d.Get(),result/"capture",desc,selected,3,192*1024*1024,depthPolicy);
        std::vector<uint8_t> depth(RW*RH*4),motion(RW*RH*4,0),exposure(4);float half=.5f,one=1;
        for(size_t i=0;i<depth.size();i+=4)std::memcpy(depth.data()+i,&half,4);std::memcpy(exposure.data(),&one,4);
        Write(result/"depth_r32f.raw",depth);Write(result/"motion_rg16f.raw",motion);Write(result/"exposure_r32f.raw",exposure);
        std::vector<std::vector<uint8_t>> outputs;
        // A-reset, identical A-reset, B-reset, A-reset: test repeatability and response.
        // All frames reset: this does not validate temporal history or jitter integration.
        const unsigned frameCount=(filterHookSession||networkHookSession)?1u:4u;
        for(unsigned frame=0;frame<frameCount;++frame) {
            if((boundaryHookSession||roundTripHookSession||patchHookSession||filterHookSession||networkHookSession)&&frame==0){FfxSessionCommandV1 arm{sizeof(arm),networkHookSession?13u:filterHookSession?11u:patchHookSession?9u:roundTripHookSession?7u:5u};if(sessionCommand(&arm)!=1)throw std::runtime_error("boundary output arm");}
            if(windowSession) {
                if(frame==2){FfxSessionCommandV1 stop{sizeof(stop),4};if(sessionCommand(&stop)!=1)throw std::runtime_error("window stop");}
                if(frame==1||frame==3){FfxSessionCommandV1 begin{sizeof(begin),3};if(sessionCommand(&begin)!=1)throw std::runtime_error("window begin");++windowTriggers;
                    windowBusyRejected=sessionCommand(&begin)==0;if(!windowBusyRejected)throw std::runtime_error("busy window accepted");}
                else ++windowIdleFrames;
            }
            // Auto-exposure providers may retain state despite reset=true.
            // Fresh contexts isolate each synthetic capture control; temporal
            // behavior requires a separate sequence test, not this equality gate.
            if(depthMode&&frame){Api(destroy(&ctx,nullptr),"destroy between controls");Api(create(&ctx,&desc.header,nullptr),"fresh control context");}
            ComPtr<ID3D12CommandAllocator> allocator;HR_CHECK(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
            ComPtr<ID3D12GraphicsCommandList> list;HR_CHECK(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
            if(sessionMode){HR_CHECK(list->Close());HR_CHECK(list->Reset(allocator.Get(),nullptr));}
            ComPtr<ID3D12CommandAllocator> producerAllocator;ComPtr<ID3D12GraphicsCommandList> producerList;
            if(windowSession){HR_CHECK(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&producerAllocator)));
                HR_CHECK(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,producerAllocator.Get(),nullptr,IID_PPV_ARGS(&producerList)));
                HR_CHECK(producerList->Close());HR_CHECK(producerList->Reset(producerAllocator.Get(),nullptr));}
            auto* depthMotionProducer=windowSession?producerList.Get():list.Get();
            auto color=Tex(d.Get(),RW,RH,DXGI_FORMAT_R16G16B16A16_FLOAT);ComPtr<ID3D12Resource> z;
            ComPtr<ID3D12DescriptorHeap> depthHeap;
            if(depthMode) {
                D3D12_RESOURCE_DESC zd{};zd.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;zd.Width=RW;zd.Height=RH;zd.DepthOrArraySize=1;zd.MipLevels=1;
                zd.Format=DXGI_FORMAT_D32_FLOAT_S8X24_UINT;zd.SampleDesc.Count=1;zd.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                D3D12_HEAP_PROPERTIES zh{};zh.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_CLEAR_VALUE clear{};clear.Format=zd.Format;clear.DepthStencil={.5f,0xa7};
                HR_CHECK(d->CreateCommittedResource(&zh,D3D12_HEAP_FLAG_NONE,&zd,D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(&z)));
                D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV;hd.NumDescriptors=1;HR_CHECK(d->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&depthHeap)));
                auto handle=depthHeap->GetCPUDescriptorHandleForHeapStart();d->CreateDepthStencilView(z.Get(),nullptr,handle);
                list->ClearDepthStencilView(handle,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,.5f,0xa7,0,nullptr);
                Recorder::Transition(list.Get(),z.Get(),D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }else z=Tex(d.Get(),RW,RH,DXGI_FORMAT_R32_FLOAT);
            auto mv=Tex(d.Get(),RW,RH,DXGI_FORMAT_R16G16_FLOAT),exp=Tex(d.Get(),1,1,DXGI_FORMAT_R32_FLOAT),out=Tex(d.Get(),OW,OH,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
            if(sessionMode) {
                ffxDispatchDescUpscale watched{};watched.color.resource=color.Get();watched.depth.resource=z.Get();watched.motionVectors.resource=mv.Get();watched.exposure.resource=exp.Get();watched.output.resource=out.Get();
                if(sessionWatch(&watched)!=1)throw std::runtime_error("session fixture watch");
            }
            auto input=Color(frame==2);Write(result/("input_"+std::to_string(frame)+".raw"),input);
            auto uc=UploadTex(d.Get(),list.Get(),color.Get(),input);ComPtr<ID3D12Resource> uz;
            if(!depthMode)uz=UploadTex(d.Get(),depthMotionProducer,z.Get(),depth);
            auto um=UploadTex(d.Get(),depthMotionProducer,mv.Get(),motion),ue=UploadTex(d.Get(),list.Get(),exp.Get(),exposure);
            ComPtr<ID3D12Resource> reactive,ur;
            if(depthMode) {
                reactive=Tex(d.Get(),RW,RH,DXGI_FORMAT_R8_UNORM);ur=UploadTex(d.Get(),list.Get(),reactive.Get(),std::vector<uint8_t>(RW*RH,0));
                for(auto r:{color.Get(),mv.Get(),reactive.Get()})Recorder::Transition(list.Get(),r,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }
            // Poison every output channel so an unwritten/partially written surface fails.
            std::vector<uint8_t> poison(OW*OH*8);auto* poisonHalf=reinterpret_cast<uint16_t*>(poison.data());
            std::fill(poisonHalf,poisonHalf+poison.size()/2,uint16_t(0x7e00));
            Recorder::Transition(list.Get(),out.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
            auto uo=UploadTex(d.Get(),list.Get(),out.Get(),poison);
            Recorder::Transition(list.Get(),out.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ffxDispatchDescUpscale a{};a.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;a.commandList=list.Get();
            a.color=ffxApiGetResourceDX12(color.Get());a.depth=ffxApiGetResourceDX12(z.Get());a.motionVectors=ffxApiGetResourceDX12(mv.Get());
            a.exposure=ffxApiGetResourceDX12(exp.Get());a.output=ffxApiGetResourceDX12(out.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            a.renderSize={RW,RH};a.upscaleSize={OW,OH};a.motionVectorScale={float(RW),float(RH)};
            a.reset=true;a.preExposure=1;a.frameTimeDelta=16.666667f;a.cameraNear=.1f;a.cameraFar=1000;a.cameraFovAngleVertical=1.04719755f;a.viewSpaceToMetersFactor=1;
            if(depthMode) {
                a.color=ffxApiGetResourceDX12(color.Get(),FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                a.depth=ffxApiGetResourceDX12(z.Get(),FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);a.depth.description.usage=FFX_API_RESOURCE_USAGE_DEPTHTARGET;
                a.motionVectors=ffxApiGetResourceDX12(mv.Get(),FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                a.reactive=ffxApiGetResourceDX12(reactive.Get(),FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);a.exposure={};
                a.motionVectorScale={-float(RW)/2,float(RH)/2};a.cameraNear=1000;a.cameraFar=.1f;
                if(frame==0||frame==2)a.upscaleSize={0,0}; // Same caller context; preserve original descriptor.
            }
            uint64_t captureTicket=0;
            if(capture) {
                auto reject=[&](const ffxDispatchDescUpscale& bad) {
                    try{capture->Before(bad,100);}catch(const std::invalid_argument&){++captureRejected;return;}
                    throw std::runtime_error("invalid capture descriptor accepted");
                };
                if(frame==0) {
                    auto bad=a;bad.color.state=FFX_API_RESOURCE_STATE_COMMON;reject(bad);
                    bad=a;bad.color.description.width++;reject(bad);
                    bad=a;bad.header.type=123;reject(bad);
                    bad=a;bad.header.pNext=&bad.header;reject(bad);
                    bad=a;bad.depth.resource=nullptr;reject(bad);
                    bad=a;bad.jitterOffset.x=std::numeric_limits<float>::quiet_NaN();reject(bad);
                    bad=a;bad.output=bad.color;reject(bad);
                    if(depthMode) {
                        auto rejectDepth=[&](const ffxDispatchDescUpscale& value){reject(value);++depthRejected;};
                        bad=a;bad.depth.description.usage|=FFX_API_RESOURCE_USAGE_UAV;rejectDepth(bad);
                        bad=a;bad.depth.description.width++;rejectDepth(bad);
                        bad=a;bad.depth.state=FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;rejectDepth(bad);
                        bad=a;bad.color=bad.depth;rejectDepth(bad);
                        bad=a;bad.upscaleSize={OW,0};rejectDepth(bad);
                        bad=a;bad.upscaleSize={OW+1,OH};rejectDepth(bad);
                        bad=a;bad.renderSize={RW+1,RH};rejectDepth(bad);
                        ffx_capture::Collector rejectPolicy(d.Get(),result/"depth_policy_rejected",desc,selected);
                        bool rejected=false;try{rejectPolicy.Before(a,0);}catch(const std::invalid_argument&){rejected=true;}
                        if(!rejected||rejectPolicy.Accepted())throw std::runtime_error("implicit depth-plane policy accepted");++depthRejected;
                        auto stale=desc;stale.maxUpscaleSize={OW+10,OH+10};
                        ffx_capture::Collector rejectContext(d.Get(),result/"stale_context_rejected",stale,selected,3,192*1024*1024,depthPolicy);
                        rejected=false;try{rejectContext.Before(a,0);}catch(const std::invalid_argument&){rejected=true;}
                        if(!rejected||rejectContext.Accepted())throw std::runtime_error("stale default context accepted");++depthRejected;
                    }
                    ffx_capture::Collector limitedCapture(d.Get(),result/"capture_byte_budget",desc,selected,3,depthMode?40*1024*1024:2000000,depthPolicy);
                    byteBudgetRejected=limitedCapture.Before(a,0)==0&&limitedCapture.Accepted()==0&&limitedCapture.Dropped()==1&&limitedCapture.Pending()==0;
                    if(!byteBudgetRejected)throw std::runtime_error("capture byte budget did not reject");
                }
                captureTicket=capture->Before(a,frame);
                if(frame==0)reject(a); // duplicate pending list
                if(capture->Poll()!=0)throw std::runtime_error("capture mapped before submission");++earlyPolls;
            }
            if(windowSession) {
                // Independent positive control for the optional enhanced hook.
                // A global ordering barrier changes no resource layout or bytes.
                ComPtr<ID3D12GraphicsCommandList7> extended;HR_CHECK(list.As(&extended));
                D3D12_GLOBAL_BARRIER barrier{D3D12_BARRIER_SYNC_ALL,D3D12_BARRIER_SYNC_ALL,D3D12_BARRIER_ACCESS_COMMON,D3D12_BARRIER_ACCESS_COMMON};
                D3D12_BARRIER_GROUP group{};group.Type=D3D12_BARRIER_TYPE_GLOBAL;group.NumBarriers=1;group.pGlobalBarriers=&barrier;
                extended->Barrier(1,&group);
            }
            stage("dispatch "+std::to_string(frame)+" begin");Api(dispatch(&ctx,&a.header),"dispatch");stage("dispatch recorded OK");
            if(capture)capture->After(captureTicket,list.Get());
            std::unique_ptr<ffx_boundary::OutputCopy> boundaryCopy;
            std::unique_ptr<ffx_boundary::OutputRoundTrip> roundTripCopy;
            auto outputReadState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            if(boundaryMode||roundTripMode||observeSession) {
                if(boundaryMode)boundaryCopy=std::make_unique<ffx_boundary::OutputCopy>(d.Get(),out.Get());
                if(roundTripMode)roundTripCopy=std::make_unique<ffx_boundary::OutputRoundTrip>(d.Get(),out.Get());
                D3D12_RESOURCE_BARRIER boundary{};boundary.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                outputReadState=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                boundary.Transition={out.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,outputReadState};
                if(adversarialSession&&(frame==1||frame==2)){
                    D3D12_RESOURCE_BARRIER pair[]={boundary,boundary};
                    if(frame==1){pair[0].Transition.StateAfter=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;pair[1].Transition.StateBefore=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;}
                    else{pair[0].Flags=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;pair[1].Flags=D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;}
                    list->ResourceBarrier(2,pair);
                }else list->ResourceBarrier(1,&boundary);
                if(boundaryCopy)boundaryCopy->Record(list.Get(),boundary);
                if(roundTripCopy)roundTripCopy->Record(list.Get(),boundary);
            }
            auto fp=Footprint(d.Get(),out.Get());auto rb=Readback(d.Get(),fp.bytes);
            ComPtr<ID3D12CommandAllocator> suffixAllocator,suffix2Allocator;ComPtr<ID3D12GraphicsCommandList> suffixList,suffix2List;
            ID3D12GraphicsCommandList* readbackList=list.Get();
            if(filterHookSession||networkHookSession){HR_CHECK(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&suffixAllocator)));HR_CHECK(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,suffixAllocator.Get(),nullptr,IID_PPV_ARGS(&suffixList)));
                HR_CHECK(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&suffix2Allocator)));HR_CHECK(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,suffix2Allocator.Get(),nullptr,IID_PPV_ARGS(&suffix2List)));readbackList=suffixList.Get();}
            Recorder::Transition(readbackList,out.Get(),outputReadState,D3D12_RESOURCE_STATE_COPY_SOURCE);
            auto src=Recorder::TextureLocation(out.Get()),dst=Linear(rb.Get(),fp);readbackList->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
            Layout stencilFp;ComPtr<ID3D12Resource> stencilReadback;
            if(depthMode) {
                stencilFp=Footprint(d.Get(),z.Get(),1);stencilReadback=Readback(d.Get(),stencilFp.bytes);
                D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={z.Get(),1,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE};
                list->ResourceBarrier(1,&b);auto zs=Recorder::TextureLocation(z.Get());zs.SubresourceIndex=1;auto zd=Linear(stencilReadback.Get(),stencilFp);list->CopyTextureRegion(&zd,0,0,0,&zs,nullptr);
                std::swap(b.Transition.StateBefore,b.Transition.StateAfter);list->ResourceBarrier(1,&b);
            }
            if(producerList)HR_CHECK(producerList->Close());if(suffixList)HR_CHECK(suffixList->Close());if(suffix2List)HR_CHECK(suffix2List->Close());
            HR_CHECK(list->Close());ID3D12CommandList* lists[]={windowSession?producerList.Get():list.Get(),(filterHookSession||networkHookSession)?suffixList.Get():list.Get(),suffix2List.Get()};
            ComPtr<ID3D12Fence> gate;
            // The residual-filter fixture deliberately avoids an unsignaled GPU
            // queue wait. Earlier gate-based fault injection made a host hang
            // capable of stalling the display adapter and is not a safe proof.
            if((capture||boundaryHookSession||roundTripHookSession||patchHookSession)&&frame==0){HR_CHECK(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&gate)));HR_CHECK(queue->Wait(gate.Get(),1));}
            queue->ExecuteCommandLists((filterHookSession||networkHookSession)?3u:windowSession?2u:1u,lists);
            if(capture)capture->Submitted(captureTicket,queue.Get(),list.Get());
            if(boundaryCopy)boundaryCopy->Submitted(queue.Get(),list.Get());
            if(roundTripCopy)roundTripCopy->Submitted(queue.Get(),list.Get());
            if(gate) {
                FfxSessionCommandV1 poll{sizeof(poll),networkHookSession?14u:filterHookSession?12u:patchHookSession?10u:roundTripHookSession?8u:boundaryHookSession?6u:2u};const auto saved=capture?capture->Poll():sessionCommand(&poll);HR_CHECK(gate->Signal(1));
                if(saved!=(capture?0u:2u))throw std::runtime_error("capture mapped while GPU queue was gated");++earlyPolls;
            }
            // Fail-stop on uncertain queue completion; never unwind/recycle pending resources.
            HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            if(!event||FAILED(queue->Signal(fence.Get(),frame+1))||FAILED(fence->SetEventOnCompletion(frame+1,event))||WaitForSingleObject(event,60000)!=WAIT_OBJECT_0) {
                stage("FATAL uncertain GPU completion");TerminateProcess(GetCurrentProcess(),3);
            }CloseHandle(event);HR_CHECK(d->GetDeviceRemovedReason());
            outputs.push_back(Unpack(rb.Get(),fp));Write(result/("output_"+std::to_string(frame)+".raw"),outputs.back());stage("fence/readback OK");
            if(boundaryCopy){std::vector<uint8_t> boundaryRaw;if(!boundaryCopy->Poll(boundaryRaw))throw std::runtime_error("boundary fence incomplete");
                boundaryExact&=boundaryRaw==outputs.back();++boundaryCompleted;Write(result/("boundary_"+std::to_string(frame)+".raw"),boundaryRaw);}
            if(roundTripCopy){std::vector<uint8_t> boundaryRaw;if(!roundTripCopy->Poll(boundaryRaw))throw std::runtime_error("roundtrip fence incomplete");
                boundaryExact&=boundaryRaw==outputs.back();++boundaryCompleted;Write(result/("boundary_"+std::to_string(frame)+".raw"),boundaryRaw);}
            if(depthMode) {
                auto stencil=Unpack(stencilReadback.Get(),stencilFp);Write(result/("stencil_"+std::to_string(frame)+".raw"),stencil);
                for(auto v:stencil)if(v!=0xa7)++stencilMismatches;
            }
            if(capture && capture->Poll()!=(captureTicket?1u:0u))throw std::runtime_error("capture completion count mismatch");
            if(sessionMode){FfxSessionCommandV1 command{sizeof(command),networkHookSession?14u:filterHookSession?12u:patchHookSession?10u:roundTripHookSession?8u:boundaryHookSession?6u:2u};sessionPoll=sessionCommand(&command);if(sessionPoll!=((observeSession&&!boundaryHookSession&&!roundTripHookSession&&!patchHookSession&&!filterHookSession&&!networkHookSession)?2u:1u))throw std::runtime_error("session completion/observe status");}
        }
        if(windowSession){FfxSessionCommandV1 stop{sizeof(stop),4},begin{sizeof(begin),3};
            if(sessionCommand(&stop)!=1)throw std::runtime_error("final window stop");
            for(unsigned i=2;i<8;++i){if(sessionCommand(&begin)!=1||sessionCommand(&stop)!=1)throw std::runtime_error("bounded window cycling");}
            windowCapRejected=sessionCommand(&begin)==0;if(!windowCapRejected)throw std::runtime_error("window cap not enforced");
            FfxSessionCommandV1 invalid{sizeof(invalid),99};if(sessionCommand(&invalid)!=0)throw std::runtime_error("invalid command accepted");
        }
        Api(destroy(&ctx,nullptr),"destroy");stage("destroy OK");
        unsigned errors=0,warnings=0;std::ofstream messages(result/"d3d12_messages.log");
        for(UINT64 i=0;i<info->GetNumStoredMessages();++i) {
            SIZE_T size=0;HR_CHECK(info->GetMessage(i,nullptr,&size));std::vector<uint8_t> b(size);auto* m=reinterpret_cast<D3D12_MESSAGE*>(b.data());HR_CHECK(info->GetMessage(i,m,&size));
            messages<<m->Severity<<": "<<m->pDescription<<'\n';if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)++errors;if(m->Severity==D3D12_MESSAGE_SEVERITY_WARNING)++warnings;
        }
        bool finite=true;for(const auto& b:outputs)for(size_t i=0;i<b.size();i+=2){uint16_t h;std::memcpy(&h,b.data()+i,2);if((h&0x7c00)==0x7c00)finite=false;}
        bool repeat=(filterHookSession||networkHookSession)?true:patchHookSession?(outputs[1]==outputs[3]):(outputs[0]==outputs[1]&&outputs[0]==outputs[3]);
        bool sensitive=(filterHookSession||networkHookSession)?true:patchHookSession?(outputs[1]!=outputs[2]):(outputs[0]!=outputs[2]);
        const bool capturePass=!capture||(capture->Completed()==3&&capture->Pending()==0&&capture->Dropped()==1&&captureRejected==(depthMode?15u:8u)&&earlyPolls==5&&byteBudgetRejected&&(!depthMode||depthRejected==9));
        bool pass=!errors&&finite&&repeat&&sensitive&&capturePass&&!stencilMismatches&&boundaryExact&&(!(boundaryMode||roundTripMode)||boundaryCompleted==4);
        std::ofstream manifest(result/"manifest.json");manifest<<"{\n\"status\":\""<<(pass?"DISPATCH_READBACK_PASS":"DISPATCH_READBACK_FAIL")<<"\",\n"
          <<"\"requested_provider\":\""<<requested<<"\",\"requested_provider_id\":"<<selected<<",\"version_override_create_status\":0,\n"
          <<"\"adapter_vendor\":"<<ad.VendorId<<",\"adapter_device\":"<<ad.DeviceId<<",\"render_size\":["<<RW<<','<<RH<<"],\"output_size\":["<<OW<<','<<OH<<"],\n"
          <<"\"depth_plane_mode\":"<<(depthMode?"true":"false")<<",\"depth_invalid_rejected\":"<<depthRejected<<",\"stencil_mismatched_bytes\":"<<stencilMismatches<<",\n"
          <<"\"fresh_context_per_frame\":"<<(depthMode?"true":"false")<<",\n"
          <<"\"session_enabled\":"<<(sessionMode?"true":"false")<<",\"session_poll_status\":"<<sessionPoll<<",\n"
          <<"\"output_boundary_enabled\":"<<(boundaryMode?"true":"false")<<",\"output_boundary_completed\":"<<boundaryCompleted<<",\"output_boundary_exact\":"<<(boundaryExact?"true":"false")<<",\n"
          <<"\"output_roundtrip_enabled\":"<<(roundTripMode?"true":"false")<<",\"output_roundtrip_completed\":"<<(roundTripMode?boundaryCompleted:0)<<",\"output_roundtrip_exact\":"<<(boundaryExact?"true":"false")<<",\n"
          <<"\"boundary_return_fixture\":"<<(observeSession?"true":"false")<<",\n"
          <<"\"boundary_adversarial_fixture\":"<<(adversarialSession?"true":"false")<<",\n"
          <<"\"boundary_hook_capture_fixture\":"<<(boundaryHookSession?"true":"false")<<",\n"
          <<"\"boundary_roundtrip_hook_fixture\":"<<(roundTripHookSession?"true":"false")<<",\n"
          <<"\"boundary_patch_hook_fixture\":"<<(patchHookSession?"true":"false")<<",\n"
          <<"\"boundary_filter_hook_fixture\":"<<(filterHookSession?"true":"false")<<",\n"
          <<"\"boundary_network_hook_fixture\":"<<(networkHookSession?"true":"false")<<",\n"
          <<"\"session_observe_only\":"<<(observeSession?"true":"false")<<",\n"
          <<"\"session_window_control\":"<<(windowSession?"true":"false")<<",\"window_idle_frames\":"<<windowIdleFrames<<",\"window_triggers_with_frames\":"<<windowTriggers<<",\"window_cap_rejected\":"<<(windowCapRejected?"true":"false")<<",\"window_busy_rejected\":"<<(windowBusyRejected?"true":"false")<<",\n"
          <<"\"window_split_producer_lists\":"<<(windowSession?"true":"false")<<",\n"
          <<"\"window_enhanced_global_barrier_probe\":"<<(windowSession?"true":"false")<<",\n"
          <<"\"dispatches\":"<<frameCount<<",\"completed_fences\":"<<frameCount<<",\"debug_layer\":true,\"debug_errors\":"<<errors<<",\"debug_warnings\":"<<warnings<<",\n"
          <<"\"finite\":"<<(finite?"true":"false")<<",\"reset_repeat_equal\":"<<(repeat?"true":"false")<<",\"input_sensitive\":"<<(sensitive?"true":"false")<<",\n"
          <<"\"capture_enabled\":"<<(captureMode?"true":"false")<<",\"capture_completed\":"<<(capture?capture->Completed():0)<<",\"capture_pending\":"<<(capture?capture->Pending():0)
          <<",\"capture_dropped\":"<<(capture?capture->Dropped():0)<<",\"capture_invalid_rejected\":"<<captureRejected<<",\"capture_early_poll_checks\":"<<earlyPolls
          <<",\"capture_byte_budget_rejected\":"<<(byteBudgetRejected?"true":"false")<<",\n"
          <<"\"output_poisoned_nan\":true,\"all_frames_reset\":true,\"temporal_history_verified\":false,\"game_launched\":false,\"dlss_nr_verified\":false,\"game_runtime_ready\":false\n}\n";
        manifest.close();if(!manifest)throw std::runtime_error("manifest write failed");stage(pass?"PASS":"FAIL");return pass?0:1;
    }catch(const std::exception& e){stage(std::string("FAIL: ")+e.what());std::ofstream(result/"manifest.json")<<"{\"status\":\"PROBE_FAILED\",\"game_launched\":false,\"game_runtime_ready\":false}\n";return 1;}
}
