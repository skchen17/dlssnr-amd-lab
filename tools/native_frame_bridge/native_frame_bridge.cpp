// Isolated D3D12 texture/shared-buffer <-> current PyTorch HIP runtime harness.
// CPU upload/readback exports are OFFLINE TEST FIXTURES, never game transport.
// No NVIDIA imports, neural kernels, pointer-identity assumptions or TDR edits.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define __HIP_PLATFORM_AMD__
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>
using Microsoft::WRL::ComPtr;
static thread_local std::string error;
static void hr(HRESULT v){if(FAILED(v))throw std::runtime_error("D3D12 HRESULT "+std::to_string(uint32_t(v)));}
static void hc(hipError_t v){if(v!=hipSuccess)throw std::runtime_error("HIP error "+std::to_string(int(v)));}
#define STR_(x) #x
#define STR(x) STR_(x)
template<class T> static T api(const char* n){
    auto module=GetModuleHandleW(L"amdhip64_7.dll");
    if(!module)throw std::runtime_error("existing PyTorch HIP runtime not loaded");
    auto p=GetProcAddress(module,n);if(!p)throw std::runtime_error(std::string("missing HIP API ")+n);
    return reinterpret_cast<T>(p);
}
#define HIP(fn) api<decltype(&fn)>(STR(fn))
static void transition(ID3D12GraphicsCommandList* c,ID3D12Resource* r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){
    D3D12_RESOURCE_BARRIER x{};x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;x.Transition={r,0,a,b};c->ResourceBarrier(1,&x);
}
struct Shared {
    ComPtr<ID3D12Heap> heap;ComPtr<ID3D12Resource> buffer;HANDLE handle=nullptr;
    hipExternalMemory_t imported=nullptr;void* pointer=nullptr;
};
struct Bridge {
    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;ComPtr<ID3D12GraphicsCommandList> cmd;
    ComPtr<ID3D12Resource> texture,upload,readback;
    ComPtr<ID3D12Fence> inputFence,outputFence,hostFence;
    HANDLE inHandle=nullptr,outHandle=nullptr,event=nullptr;
    hipExternalSemaphore_t inSem=nullptr,outSem=nullptr;
    Shared input,output;D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT width=0,height=0;UINT64 bytes=0,seq=0,hostValue=0;
    DWORD owner=GetCurrentThreadId();int state=0; //0 idle,1 D3D submitted,2 HIP imported,3 HIP output,4 consumed,-1 poisoned
    bool importLocal=true;
    void require(int expected){if(owner!=GetCurrentThreadId()||state!=expected)throw std::runtime_error("bridge owner/state violation");}
    void shared(Shared& s){
        D3D12_HEAP_DESC hd{};hd.SizeInBytes=(bytes+65535)&~UINT64(65535);hd.Properties.Type=D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags=D3D12_HEAP_FLAG_SHARED|D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        hr(device->CreateHeap(&hd,IID_PPV_ARGS(&s.heap)));
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr(device->CreatePlacedResource(s.heap.Get(),0,&b,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&s.buffer)));
        hr(device->CreateSharedHandle(s.heap.Get(),nullptr,GENERIC_ALL,nullptr,&s.handle));
        if(!importLocal)return;
        hipExternalMemoryHandleDesc md{};md.type=hipExternalMemoryHandleTypeD3D12Heap;md.handle.win32.handle=s.handle;md.size=hd.SizeInBytes;
        hc(HIP(hipImportExternalMemory)(&s.imported,&md));
        hipExternalMemoryBufferDesc bd{};bd.size=bytes;
        hc(HIP(hipExternalMemoryGetMappedBuffer)(&s.pointer,s.imported,&bd));
    }
    void fence(ComPtr<ID3D12Fence>& f,HANDLE& handle,hipExternalSemaphore_t& sem){
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&f)));
        hr(device->CreateSharedHandle(f.Get(),nullptr,GENERIC_ALL,nullptr,&handle));
        if(!importLocal)return;
        hipExternalSemaphoreHandleDesc d{};d.type=hipExternalSemaphoreHandleTypeD3D12Fence;d.handle.win32.handle=handle;
        hc(HIP(hipImportExternalSemaphore)(&sem,&d));
    }
    void init(UINT w,UINT h,bool localHip=true,UINT64 selectedLuid=0,ID3D12Device* borrowedDevice=nullptr,ID3D12CommandQueue* borrowedQueue=nullptr){
        if(!w||!h||w>(borrowedDevice?3840:768)||h>(borrowedDevice?2160:512))throw std::runtime_error("unreviewed bridge test dimensions");
        width=w;height=h;importLocal=localHip;
        ComPtr<IDXGIFactory4> factory;hr(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        if(importLocal){
            int index=-1;hc(HIP(hipGetDevice)(&index));hipDeviceProp_t properties{};hc(HIP(hipGetDeviceProperties)(&properties,index));
            if(std::string(properties.gcnArchName).find("gfx1201")==std::string::npos)throw std::runtime_error("unreviewed HIP architecture");
            LUID luid{};std::memcpy(&luid,properties.luid,8);hr(factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter)));
            DXGI_ADAPTER_DESC1 ad{};hr(adapter->GetDesc1(&ad));
            if(ad.VendorId!=0x1002||std::memcmp(&ad.AdapterLuid,properties.luid,8))throw std::runtime_error("HIP/D3D adapter mismatch");
        }else{
            if(!selectedLuid)throw std::runtime_error("explicit authenticated worker adapter LUID required");
            LUID luid{};std::memcpy(&luid,&selectedLuid,8);hr(factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter)));
            DXGI_ADAPTER_DESC1 ad{};hr(adapter->GetDesc1(&ad));
            if(ad.VendorId!=0x1002||ad.DeviceId!=0x7550||(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)||std::memcmp(&ad.AdapterLuid,&luid,8))
                throw std::runtime_error("standalone producer adapter identity differs");
        }
        D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        if(borrowedDevice){
            if(localHip||!borrowedQueue||borrowedQueue->GetDesc().Type!=q.Type)throw std::runtime_error("borrowed queue invalid");
            auto luid=borrowedDevice->GetAdapterLuid();if(std::memcmp(&luid,&selectedLuid,8))throw std::runtime_error("borrowed adapter mismatch");
            ComPtr<ID3D12Device> owner;hr(borrowedQueue->GetDevice(IID_PPV_ARGS(&owner)));
            ComPtr<IUnknown> a,b;hr(owner.As(&a));hr(borrowedDevice->QueryInterface(IID_PPV_ARGS(&b)));if(a!=b)throw std::runtime_error("queue device identity differs");
            device=borrowedDevice;queue=borrowedQueue;
        }else{hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));}
        hr(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&alloc)));
        hr(device->CreateCommandList(0,q.Type,alloc.Get(),nullptr,IID_PPV_ARGS(&cmd)));hr(cmd->Close());
        D3D12_RESOURCE_DESC t{};t.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;t.Width=w;t.Height=h;t.DepthOrArraySize=1;t.MipLevels=1;t.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;t.SampleDesc.Count=1;
        if(borrowedDevice)t.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
        hr(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&t,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&texture)));
        UINT rows=0;UINT64 rowBytes=0;device->GetCopyableFootprints(&t,0,1,0,&footprint,&rows,&rowBytes,&bytes);
        if(rows!=h||rowBytes!=w*8ULL)throw std::runtime_error("texture footprint mismatch");
        D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=bytes;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hp.Type=D3D12_HEAP_TYPE_UPLOAD;hr(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload)));
        hp.Type=D3D12_HEAP_TYPE_READBACK;hr(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)));
        shared(input);shared(output);fence(inputFence,inHandle,inSem);fence(outputFence,outHandle,outSem);
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&hostFence)));
        event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event)throw std::runtime_error("event creation failed");
    }
    void begin(){hr(alloc->Reset());hr(cmd->Reset(alloc.Get(),nullptr));}
    void submit(){hr(cmd->Close());ID3D12CommandList* p[]={cmd.Get()};queue->ExecuteCommandLists(1,p);}
    void waitHost(){
        hr(queue->Signal(hostFence.Get(),++hostValue));hr(hostFence->SetEventOnCompletion(hostValue,event));
        if(WaitForSingleObject(event,2000)!=WAIT_OBJECT_0)throw std::runtime_error("D3D timeout; retain resources; no GPU cancellation");
        hr(device->GetDeviceRemovedReason());
        if(hostFence->GetCompletedValue()!=hostValue)throw std::runtime_error("invalid completion fence");
    }
    void uploadFixture(const void* raw,UINT64 size){
        require(0);if(!raw||size!=width*height*8ULL)throw std::runtime_error("fixture size mismatch");
        void* p=nullptr;D3D12_RANGE none{};hr(upload->Map(0,&none,&p));
        for(UINT y=0;y<height;++y)std::memcpy(static_cast<char*>(p)+footprint.Offset+y*footprint.Footprint.RowPitch,static_cast<const char*>(raw)+y*width*8ULL,width*8ULL);
        upload->Unmap(0,nullptr);begin();
        D3D12_TEXTURE_COPY_LOCATION t{};t.pResource=texture.Get();t.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION b{};b.pResource=upload.Get();b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;b.PlacedFootprint=footprint;
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);cmd->CopyTextureRegion(&t,0,0,0,&b,nullptr);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(cmd.Get(),input.buffer.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
        b.pResource=input.buffer.Get();cmd->CopyTextureRegion(&b,0,0,0,&t,nullptr);
        transition(cmd.Get(),input.buffer.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        state=1;submit();hr(queue->Signal(inputFence.Get(),++seq));waitHost();
    }
    void toTensor(void* target,UINT64 size,void* stream){
        require(1);if(!target||size!=width*height*8ULL)throw std::runtime_error("tensor size mismatch");
        hipExternalSemaphoreWaitParams params{};params.params.fence.value=seq;
        hc(HIP(hipWaitExternalSemaphoresAsync)(&inSem,&params,1,static_cast<hipStream_t>(stream)));
        hc(HIP(hipMemcpy2DAsync)(target,width*8ULL,static_cast<char*>(input.pointer)+footprint.Offset,footprint.Footprint.RowPitch,width*8ULL,height,hipMemcpyDeviceToDevice,static_cast<hipStream_t>(stream)));
        state=2;
    }
    // Borrowed-device callers populate texture with GPU rendering, restore COMMON,
    // and retire that producer before this call. No CPU image upload/readback.
    void submitTexture(){
        require(0);begin();D3D12_TEXTURE_COPY_LOCATION t{},b{};t.pResource=texture.Get();t.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        b.pResource=input.buffer.Get();b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;b.PlacedFootprint=footprint;
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(cmd.Get(),input.buffer.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyTextureRegion(&b,0,0,0,&t,nullptr);
        transition(cmd.Get(),input.buffer.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        state=1;submit();hr(queue->Signal(inputFence.Get(),++seq));waitHost();
    }
    void receiveTexture(){
        require(3);auto done=outputFence->GetCompletedValue();if(done==UINT64_MAX||done<seq)throw std::runtime_error("output not complete");
        begin();D3D12_TEXTURE_COPY_LOCATION t{},b{};t.pResource=texture.Get();t.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        b.pResource=output.buffer.Get();b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;b.PlacedFootprint=footprint;
        transition(cmd.Get(),output.buffer.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);cmd->CopyTextureRegion(&t,0,0,0,&b,nullptr);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
        transition(cmd.Get(),output.buffer.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);submit();waitHost();state=4;
    }
    void fromTensor(const void* source,UINT64 size,void* stream){
        require(2);if(!source||size!=width*height*8ULL)throw std::runtime_error("tensor size mismatch");
        hc(HIP(hipMemcpy2DAsync)(static_cast<char*>(output.pointer)+footprint.Offset,footprint.Footprint.RowPitch,source,width*8ULL,width*8ULL,height,hipMemcpyDeviceToDevice,static_cast<hipStream_t>(stream)));
        hipExternalSemaphoreSignalParams params{};params.params.fence.value=seq;
        hc(HIP(hipSignalExternalSemaphoresAsync)(&outSem,&params,1,static_cast<hipStream_t>(stream)));state=3;
    }
    void readFixture(void* raw,UINT64 size){
        require(3);if(!raw||size!=width*height*8ULL)throw std::runtime_error("fixture size mismatch");
        // Never park D3D on a future HIP fence after uncertain submission.
        auto done=outputFence->GetCompletedValue();if(done==UINT64_MAX||done<seq)throw std::runtime_error("HIP output not completed; retain");
        hr(queue->Wait(outputFence.Get(),seq));begin();
        D3D12_TEXTURE_COPY_LOCATION t{};t.pResource=texture.Get();t.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION b{};b.pResource=output.buffer.Get();b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;b.PlacedFootprint=footprint;
        transition(cmd.Get(),output.buffer.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);cmd->CopyTextureRegion(&t,0,0,0,&b,nullptr);
        transition(cmd.Get(),output.buffer.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
        b.pResource=readback.Get();cmd->CopyTextureRegion(&b,0,0,0,&t,nullptr);
        transition(cmd.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);submit();waitHost();
        void* p=nullptr;D3D12_RANGE range{0,size_t(bytes)};hr(readback->Map(0,&range,&p));
        for(UINT y=0;y<height;++y)std::memcpy(static_cast<char*>(raw)+y*width*8ULL,static_cast<const char*>(p)+footprint.Offset+y*footprint.Footprint.RowPitch,width*8ULL);
        D3D12_RANGE none{};readback->Unmap(0,&none);state=4;
    }
    void close(){
        if(owner!=GetCurrentThreadId()||(state!=0&&state!=4))throw std::runtime_error("uncertain resources must remain retained");
        for(auto* s:{&input,&output}){
            if(s->pointer){hc(HIP(hipFree)(s->pointer));s->pointer=nullptr;}
            if(s->imported){hc(HIP(hipDestroyExternalMemory)(s->imported));s->imported=nullptr;}
            if(s->handle){CloseHandle(s->handle);s->handle=nullptr;}
        }
        for(auto* s:{&inSem,&outSem})if(*s){hc(HIP(hipDestroyExternalSemaphore)(*s));*s=nullptr;}
        for(auto* h:{&inHandle,&outHandle,&event})if(*h){CloseHandle(*h);*h=nullptr;}
    }
};
#define EXPORT extern "C" __declspec(dllexport)
EXPORT const char* nr_bridge_error(){return error.c_str();}
EXPORT void* nr_bridge_create(UINT w,UINT h){
    Bridge* p=new Bridge;
    try{p->init(w,h);return p;}catch(const std::exception& e){error=e.what();
        // No work submitted during init. Release any successfully imported objects.
        try{p->close();delete p;}catch(...){/* retain on uncertain teardown */}return nullptr;}
}
EXPORT void* nr_bridge_create_producer(UINT w,UINT h,UINT64 luid){
    Bridge* p=new Bridge;
    try{p->init(w,h,false,luid);return p;}catch(const std::exception& e){error=e.what();try{p->close();delete p;}catch(...){}return nullptr;}
}
EXPORT UINT64 nr_current_hip_luid(){try{int index=0;hc(HIP(hipGetDevice)(&index));hipDeviceProp_t p{};hc(HIP(hipGetDeviceProperties)(&p,index));UINT64 v=0;std::memcpy(&v,p.luid,8);return v;}catch(const std::exception& e){error=e.what();return 0;}}
template<class F> int call(Bridge* p,F f){
    if(!p){error="null bridge";return -1;}
    try{f();return 0;}catch(const std::exception& e){error=e.what();p->state=-1;return -1;}
}
EXPORT int nr_bridge_upload(Bridge* p,const void* raw,UINT64 n){return call(p,[&]{p->uploadFixture(raw,n);});}
EXPORT int nr_bridge_to_tensor(Bridge* p,void* target,UINT64 n,void* stream){return call(p,[&]{p->toTensor(target,n,stream);});}
EXPORT int nr_bridge_from_tensor(Bridge* p,const void* source,UINT64 n,void* stream){return call(p,[&]{p->fromTensor(source,n,stream);});}
EXPORT int nr_bridge_read(Bridge* p,void* raw,UINT64 n){return call(p,[&]{p->readFixture(raw,n);});}
EXPORT int nr_bridge_next(Bridge* p){return call(p,[&]{p->require(4);p->state=0;});}
EXPORT int nr_bridge_close(Bridge* p){int code=call(p,[&]{p->close();});if(!code)delete p;return code;}
EXPORT UINT nr_bridge_pitch(Bridge* p){return p?p->footprint.Footprint.RowPitch:0;}
EXPORT UINT64 nr_bridge_imported_pointer(Bridge* p){return p?reinterpret_cast<UINT64>(p->input.pointer):0;}
EXPORT UINT64 nr_bridge_d3d_address(Bridge* p){return p?p->input.buffer->GetGPUVirtualAddress():0;}

// Cross-process metadata ABI: sixteen uint64 fields, NO pixel payload or GPU VA.
// Handles are duplicated into the exact worker PID, never interpreted as pointers.
// [version,producer,worker,generation,w,h,pitch,span,heap,luid,in,out,ready,done,offset,reserved]
EXPORT int nr_bridge_export(Bridge* p,DWORD worker,UINT64 generation,UINT64* fields){
    return call(p,[&]{
        p->require(0);if(!fields||!generation||!worker||worker==GetCurrentProcessId())throw std::runtime_error("invalid remote owner");
        HANDLE process=OpenProcess(PROCESS_DUP_HANDLE|PROCESS_QUERY_LIMITED_INFORMATION,FALSE,worker);
        if(!process)throw std::runtime_error("worker handle open failed");
        UINT64 result[16]={1,GetCurrentProcessId(),worker,generation,p->width,p->height,p->footprint.Footprint.RowPitch,p->bytes,p->input.heap->GetDesc().SizeInBytes};
        auto luid=p->device->GetAdapterLuid();std::memcpy(&result[9],&luid,8);
        HANDLE sources[]={p->input.handle,p->output.handle,p->inHandle,p->outHandle};int count=0;
        for(;count<4;++count){HANDLE target=nullptr;if(!DuplicateHandle(GetCurrentProcess(),sources[count],process,&target,0,FALSE,DUPLICATE_SAME_ACCESS))break;result[10+count]=reinterpret_cast<UINT64>(target);}
        if(count!=4){
            for(int i=0;i<count;++i){HANDLE local=nullptr;if(DuplicateHandle(process,reinterpret_cast<HANDLE>(result[10+i]),GetCurrentProcess(),&local,0,FALSE,DUPLICATE_CLOSE_SOURCE|DUPLICATE_SAME_ACCESS))CloseHandle(local);}
            CloseHandle(process);throw std::runtime_error("remote handle duplication failed");
        }
        CloseHandle(process);std::memcpy(fields,result,sizeof(result));
    });
}
EXPORT int nr_bridge_remote_done(Bridge* p,UINT64 sequence){
    return call(p,[&]{p->require(1);auto value=p->outputFence->GetCompletedValue();
        if(sequence!=p->seq||value==UINT64_MAX||value<sequence)throw std::runtime_error("remote fence/sequence incomplete");p->state=3;});
}

struct Remote {
    UINT64 d[16]{};Shared input,output;hipExternalSemaphore_t ready=nullptr,done=nullptr;
    UINT64 sequence=0;int state=0;DWORD owner=GetCurrentThreadId();
    void require(int value){if(owner!=GetCurrentThreadId()||state!=value)throw std::runtime_error("remote state/owner");}
    void init(const UINT64* descriptor,DWORD producer){
        if(!descriptor)throw std::runtime_error("null descriptor");std::memcpy(d,descriptor,sizeof(d));
        if(d[0]!=1||d[1]!=producer||!producer||d[2]!=GetCurrentProcessId()||!d[3]||!d[4]||!d[5]||d[4]>3840||d[5]>2160||
           d[6]<d[4]*8||d[6]%256||d[7]<(d[5]-1)*d[6]+d[4]*8||d[8]<d[7]||d[8]>128*1024*1024||d[8]%65536||d[14]||d[15])throw std::runtime_error("remote descriptor bounds/identity");
        int index=0;hc(HIP(hipGetDevice)(&index));hipDeviceProp_t prop{};hc(HIP(hipGetDeviceProperties)(&prop,index));
        if(std::memcmp(&d[9],prop.luid,8)||std::string(prop.gcnArchName).find("gfx1201")==std::string::npos)throw std::runtime_error("remote adapter mismatch");
        for(int i=0;i<2;++i){auto& s=i?output:input;s.handle=reinterpret_cast<HANDLE>(d[10+i]);
            hipExternalMemoryHandleDesc md{};md.type=hipExternalMemoryHandleTypeD3D12Heap;md.handle.win32.handle=s.handle;md.size=d[8];
            hc(HIP(hipImportExternalMemory)(&s.imported,&md));hipExternalMemoryBufferDesc bd{};bd.size=d[7];hc(HIP(hipExternalMemoryGetMappedBuffer)(&s.pointer,s.imported,&bd));}
        for(int i=0;i<2;++i){hipExternalSemaphoreHandleDesc sd{};sd.type=hipExternalSemaphoreHandleTypeD3D12Fence;sd.handle.win32.handle=reinterpret_cast<HANDLE>(d[12+i]);hc(HIP(hipImportExternalSemaphore)(i?&done:&ready,&sd));}
    }
    void begin(void* tensor,UINT64 bytes,UINT64 generation,UINT64 seq,void* stream){
        require(0);if(!tensor||bytes!=d[4]*d[5]*8||generation!=d[3]||seq!=sequence+1)throw std::runtime_error("remote stale input/geometry");
        state=1;sequence=seq;hipExternalSemaphoreWaitParams p{};p.params.fence.value=seq;
        hc(HIP(hipWaitExternalSemaphoresAsync)(&ready,&p,1,static_cast<hipStream_t>(stream)));
        hc(HIP(hipMemcpy2DAsync)(tensor,d[4]*8,input.pointer,d[6],d[4]*8,d[5],hipMemcpyDeviceToDevice,static_cast<hipStream_t>(stream)));
    }
    void finish(const void* tensor,UINT64 bytes,UINT64 generation,UINT64 seq,void* stream){
        require(1);if(!tensor||bytes!=d[4]*d[5]*8||generation!=d[3]||seq!=sequence)throw std::runtime_error("remote stale output/geometry");
        hc(HIP(hipMemcpy2DAsync)(output.pointer,d[6],tensor,d[4]*8,d[4]*8,d[5],hipMemcpyDeviceToDevice,static_cast<hipStream_t>(stream)));
        hipExternalSemaphoreSignalParams p{};p.params.fence.value=seq;
        hc(HIP(hipSignalExternalSemaphoresAsync)(&done,&p,1,static_cast<hipStream_t>(stream)));state=2;
    }
    void consumed(UINT64 seq){require(2);if(sequence!=seq)throw std::runtime_error("stale consumer acknowledgement");state=0;}
    void close(){require(0);
        for(auto* s:{&input,&output}){if(s->pointer){hc(HIP(hipFree)(s->pointer));s->pointer=nullptr;}if(s->imported){hc(HIP(hipDestroyExternalMemory)(s->imported));s->imported=nullptr;}}
        if(ready){hc(HIP(hipDestroyExternalSemaphore)(ready));ready=nullptr;}if(done){hc(HIP(hipDestroyExternalSemaphore)(done));done=nullptr;}
        for(int i=10;i<14;++i)if(d[i]){CloseHandle(reinterpret_cast<HANDLE>(d[i]));d[i]=0;}
    }
};
template<class F> static int remoteCall(Remote* p,F f){if(!p){error="null remote";return -1;}try{f();return 0;}catch(const std::exception& e){p->state=-1;error=e.what();return -1;}}
EXPORT void* nr_remote_open(const UINT64* descriptor,DWORD producer){
    auto* p=new Remote;try{p->init(descriptor,producer);return p;}catch(const std::exception& e){error=e.what();
        // No work has been submitted. Failed imports remain isolated to this
        // worker process; caller stops the session, never retries this lease.
        return nullptr;}
}
EXPORT int nr_remote_begin(Remote* p,void* tensor,UINT64 bytes,UINT64 generation,UINT64 seq,void* stream){return remoteCall(p,[&]{p->begin(tensor,bytes,generation,seq,stream);});}
EXPORT int nr_remote_finish(Remote* p,const void* tensor,UINT64 bytes,UINT64 generation,UINT64 seq,void* stream){return remoteCall(p,[&]{p->finish(tensor,bytes,generation,seq,stream);});}
EXPORT int nr_remote_consumed(Remote* p,UINT64 seq){return remoteCall(p,[&]{p->consumed(seq);});}
EXPORT int nr_remote_close(Remote* p){int code=remoteCall(p,[&]{p->close();});if(!code)delete p;return code;}
