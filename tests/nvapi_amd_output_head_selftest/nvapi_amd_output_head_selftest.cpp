#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../third_party/nvapi/nvapi.h"
#include "../../tools/nvapi_amd/nvapi_amd_diag.h"

using Microsoft::WRL::ComPtr;
namespace fs=std::filesystem;

#define HR_CHECK(x) do{HRESULT h_=(x);if(FAILED(h_))throw std::runtime_error(#x);}while(0)
using Query_t=void*(__cdecl*)(uint32_t);
using CreateModule_t=NvAPI_Status(__cdecl*)(ID3D12Device*,const void*,NvU32,NVDX_ObjectHandle*);
using CreateFunction_t=NvAPI_Status(__cdecl*)(ID3D12Device*,NVDX_ObjectHandle,const char*,NVDX_ObjectHandle*);
using Launch_t=NvAPI_Status(__cdecl*)(ID3D12GraphicsCommandList*,const NVAPI_CU_KERNEL_LAUNCH_PARAMS*,NvU32);
using Destroy_t=NvAPI_Status(__cdecl*)(ID3D12Device*,NVDX_ObjectHandle);
using Independent_t=NvAPI_Status(__cdecl*)(NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS*);
using TraceInit_t=int(WINAPI*)(DWORD);
using TraceDevice_t=int(__cdecl*)(ID3D12Device*);
using TraceWrap_t=void*(__cdecl*)(void*);

std::vector<uint8_t> Read(const fs::path&p){std::ifstream s(p,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error("cannot open "+p.string());size_t n=size_t(s.tellg());s.seekg(0);std::vector<uint8_t>b(n);if(!s.read(reinterpret_cast<char*>(b.data()),std::streamsize(n)))throw std::runtime_error("read failed");return b;}
void Write(const fs::path&p,const void*d,size_t n){if(p.has_parent_path())fs::create_directories(p.parent_path());std::ofstream s(p,std::ios::binary);if(!s||!s.write(reinterpret_cast<const char*>(d),std::streamsize(n)))throw std::runtime_error("write failed");}
D3D12_RESOURCE_DESC BufferDesc(UINT64 n){D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=n;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;return d;}
void Barrier(ID3D12GraphicsCommandList*l,ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){D3D12_RESOURCE_BARRIER x{};x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;x.Transition.pResource=r;x.Transition.StateBefore=a;x.Transition.StateAfter=b;x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;l->ResourceBarrier(1,&x);}
uint64_t Hash(const void*d,size_t n){auto*p=static_cast<const uint8_t*>(d);uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}

int main(int argc,char**argv){try{
  if(argc!=9){std::fprintf(stderr,"usage: nvapi_amd_output_head_selftest <module_trace.dll> <nvapi64_amd.dll> <activation_arena.raw> <model_arena.raw> <slot154_params.raw> <reference_surface.raw> <output_surface.raw> <result.json>\n");return 2;}
  const auto activation=Read(argv[3]),model=Read(argv[4]),capturedParams=Read(argv[5]),reference=Read(argv[6]);
  constexpr size_t surfaceBytes=size_t(640)*360*4*2;
  constexpr size_t outputOffset=27807744,mainOffset=13873152,skipOffset=110592,headOffset=147429888,blendOffset=147429376;
  if(activation.size()!=outputOffset||model.size()<headOffset+21808||capturedParams.size()!=184||reference.size()!=surfaceBytes)throw std::runtime_error("unexpected input size");
  HMODULE trace=LoadLibraryW(fs::path(argv[1]).wstring().c_str());if(!trace)throw std::runtime_error("module_trace LoadLibrary failed");auto traceInit=reinterpret_cast<TraceInit_t>(GetProcAddress(trace,"ModuleTrace_InitializeAndWait"));auto traceDevice=reinterpret_cast<TraceDevice_t>(GetProcAddress(trace,"ModuleTrace_RegisterD3D12Device"));auto traceWrap=reinterpret_cast<TraceWrap_t>(GetProcAddress(trace,"ModuleTrace_TestWrapIndependentDescriptor"));if(!traceInit||!traceDevice||!traceWrap)throw std::runtime_error("module_trace exports missing");
  HMODULE library=LoadLibraryW(fs::path(argv[2]).wstring().c_str());if(!library)throw std::runtime_error("LoadLibrary failed");
  auto query=reinterpret_cast<Query_t>(GetProcAddress(library,"nvapi_QueryInterface"));
  auto getDiag=reinterpret_cast<NvapiAmdGetDiagnostics_t>(GetProcAddress(library,"NvapiAmd_GetDiagnostics"));
  auto resetDiag=reinterpret_cast<NvapiAmdResetDiagnostics_t>(GetProcAddress(library,"NvapiAmd_ResetDiagnostics"));
  auto registerBuffer=reinterpret_cast<NvapiAmdRegisterExternalBuffer_t>(GetProcAddress(library,"NvapiAmd_RegisterExternalBuffer"));
  auto unregisterBuffer=reinterpret_cast<NvapiAmdUnregisterExternalBuffer_t>(GetProcAddress(library,"NvapiAmd_UnregisterExternalBuffer"));
  if(!query||!getDiag||!resetDiag||!registerBuffer||!unregisterBuffer||!resetDiag())throw std::runtime_error("missing exports");if(!traceInit(10000))throw std::runtime_error("module_trace initialization failed");
  auto createModule=reinterpret_cast<CreateModule_t>(query(0xAD1A677D));auto createFunction=reinterpret_cast<CreateFunction_t>(query(0xE2436E22));auto launch=reinterpret_cast<Launch_t>(query(0x24973538));auto destroyFunction=reinterpret_cast<Destroy_t>(query(0xDF295EA6));auto destroyModule=reinterpret_cast<Destroy_t>(query(0x41C65285));auto getIndependent=reinterpret_cast<Independent_t>(traceWrap(query(0x0DDAC234)));
  if(!createModule||!createFunction||!launch||!destroyFunction||!destroyModule||!getIndependent)throw std::runtime_error("missing NVAPI surface");

  ComPtr<IDXGIFactory6> factory;HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 adapterDesc{};
  for(UINT i=0;;++i){ComPtr<IDXGIAdapter1> candidate;if(factory->EnumAdapters1(i,&candidate)==DXGI_ERROR_NOT_FOUND)break;HR_CHECK(candidate->GetDesc1(&adapterDesc));if(!(adapterDesc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&adapterDesc.VendorId==0x1002){adapter=candidate;break;}}
  if(!adapter)throw std::runtime_error("AMD adapter not found");ComPtr<ID3D12Device> device;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)));if(!traceDevice(device.Get()))throw std::runtime_error("module_trace device hook failed");
  D3D12_COMMAND_QUEUE_DESC qd{};qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;ComPtr<ID3D12CommandQueue> queue;HR_CHECK(device->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)));
  ComPtr<ID3D12Fence> fence;HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));uint64_t fenceValue=0;
  auto execute=[&](ID3D12GraphicsCommandList*l){HR_CHECK(l->Close());ID3D12CommandList*lists[]={l};queue->ExecuteCommandLists(1,lists);HR_CHECK(queue->Signal(fence.Get(),++fenceValue));while(fence->GetCompletedValue()<fenceValue)Sleep(0);};
  D3D12_HEAP_PROPERTIES def{};def.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_HEAP_PROPERTIES up{};up.Type=D3D12_HEAP_TYPE_UPLOAD;D3D12_HEAP_PROPERTIES rb{};rb.Type=D3D12_HEAP_TYPE_READBACK;
  const size_t activationBytes=activation.size();
  ComPtr<ID3D12Resource> activationGpu,modelGpu,activationUpload,modelUpload,readback,outputTexture;
  auto ad=BufferDesc(activationBytes),md=BufferDesc(model.size()),rd=BufferDesc(surfaceBytes);
  HR_CHECK(device->CreateCommittedResource(&def,D3D12_HEAP_FLAG_NONE,&ad,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&activationGpu)));
  HR_CHECK(device->CreateCommittedResource(&def,D3D12_HEAP_FLAG_NONE,&md,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&modelGpu)));
  HR_CHECK(device->CreateCommittedResource(&up,D3D12_HEAP_FLAG_NONE,&ad,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&activationUpload)));
  HR_CHECK(device->CreateCommittedResource(&up,D3D12_HEAP_FLAG_NONE,&md,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&modelUpload)));
  HR_CHECK(device->CreateCommittedResource(&rb,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)));
  void*mapped=nullptr;HR_CHECK(activationUpload->Map(0,nullptr,&mapped));std::memcpy(mapped,activation.data(),activation.size());activationUpload->Unmap(0,nullptr);
  HR_CHECK(modelUpload->Map(0,nullptr,&mapped));std::memcpy(mapped,model.data(),model.size());modelUpload->Unmap(0,nullptr);
  ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> list;HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));HR_CHECK(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
  list->CopyResource(activationGpu.Get(),activationUpload.Get());list->CopyResource(modelGpu.Get(),modelUpload.Get());Barrier(list.Get(),activationGpu.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);Barrier(list.Get(),modelGpu.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);execute(list.Get());
  D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;td.Width=640;td.Height=360;td.DepthOrArraySize=1;td.MipLevels=1;td.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;td.SampleDesc.Count=1;td.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  HR_CHECK(device->CreateCommittedResource(&def,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&outputTexture)));
  D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=1;ComPtr<ID3D12DescriptorHeap> descriptorHeap;HR_CHECK(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&descriptorHeap)));D3D12_CPU_DESCRIPTOR_HANDLE outputDescriptor=descriptorHeap->GetCPUDescriptorHandleForHeapStart();D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;ud.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;device->CreateUnorderedAccessView(outputTexture.Get(),nullptr,&ud,outputDescriptor);
  NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS independent{};independent.structSizeIn=independent.structSizeOut=sizeof(independent);independent.pDevice=device.Get();independent.type=NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SURFACE;independent.desc=outputDescriptor;if(getIndependent(&independent)!=NVAPI_OK||!independent.handle)throw std::runtime_error("surface object creation failed");

  uint64_t blob[2]={0x504F5354424C4F43ull,0x4B5F4E4154495645ull};NVDX_ObjectHandle module=nullptr,function=nullptr;
  if(createModule(device.Get(),blob,sizeof(blob),&module)!=NVAPI_OK||createFunction(device.Get(),module,"cc_tinlayout_fused_post_block_swin_1h_32_fp8",&function)!=NVAPI_OK)throw std::runtime_error("create module/function failed");
  std::vector<uint8_t>params=capturedParams;const uint64_t activationVa=activationGpu->GetGPUVirtualAddress(),modelVa=modelGpu->GetGPUVirtualAddress();
  const uint64_t addresses[4]={activationVa+mainOffset,activationVa+skipOffset,modelVa+headOffset,modelVa+blendOffset};
  std::memcpy(params.data(),&addresses[0],8);std::memcpy(params.data()+8,&addresses[1],8);std::memcpy(params.data()+16,&independent.handle,8);std::memcpy(params.data()+24,&addresses[2],8);std::memcpy(params.data()+104,&addresses[3],8);
  NVAPI_CU_KERNEL_LAUNCH_PARAMS launchParams{};launchParams.hFunction=function;launchParams.gridDim={81,49,1};launchParams.blockDim={32,1,1};launchParams.pParams=params.data();launchParams.paramSize=184;
  ComPtr<ID3D12CommandAllocator> emptyAllocator;ComPtr<ID3D12GraphicsCommandList> emptyList;HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&emptyAllocator)));HR_CHECK(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,emptyAllocator.Get(),nullptr,IID_PPV_ARGS(&emptyList)));
  const auto start=std::chrono::steady_clock::now();const NvAPI_Status launchStatus=launch(emptyList.Get(),&launchParams,1);const auto end=std::chrono::steady_clock::now();
  if(launchStatus!=NVAPI_OK)throw std::runtime_error("output-head launch failed");execute(emptyList.Get());
  ComPtr<ID3D12CommandAllocator> readAllocator;ComPtr<ID3D12GraphicsCommandList> readList;HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&readAllocator)));HR_CHECK(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,readAllocator.Get(),nullptr,IID_PPV_ARGS(&readList)));
  Barrier(readList.Get(),outputTexture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);D3D12_TEXTURE_COPY_LOCATION src{};src.pResource=outputTexture.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;src.SubresourceIndex=0;D3D12_TEXTURE_COPY_LOCATION dst{};dst.pResource=readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint.Footprint.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;dst.PlacedFootprint.Footprint.Width=640;dst.PlacedFootprint.Footprint.Height=360;dst.PlacedFootprint.Footprint.Depth=1;dst.PlacedFootprint.Footprint.RowPitch=640*8;readList->CopyTextureRegion(&dst,0,0,0,&src,nullptr);execute(readList.Get());
  std::vector<uint8_t>output(surfaceBytes);HR_CHECK(readback->Map(0,nullptr,&mapped));std::memcpy(output.data(),mapped,surfaceBytes);readback->Unmap(0,nullptr);
  uint64_t mismatches=0;for(size_t i=0;i<surfaceBytes;++i)mismatches+=output[i]!=reference[i];NvapiAmdDiagnosticsV1 diag{};diag.struct_size=sizeof(diag);if(!getDiag(&diag))throw std::runtime_error("diagnostics failed");
  const bool pass=mismatches==0&&diag.neural_math_executed==1&&diag.kernels_submitted==1&&diag.address_translations==4&&diag.translation_failures==0&&diag.independent_descriptor_calls==1&&diag.d3d12_copy_dispatches==1&&diag.counts_as_s6==0;
  Write(argv[7],output.data(),output.size());char hash[32]{};std::snprintf(hash,sizeof(hash),"%016llX",(unsigned long long)Hash(output.data(),output.size()));const double milliseconds=std::chrono::duration<double,std::milli>(end-start).count();
  const std::string json="{\n  \"schema\": 1,\n  \"experiment\": \"nvapi_amd_output_surface_dispatch\",\n  \"status\": \""+std::string(pass?"PASS":"FAIL")+"\",\n  \"classification\": \"REAL_SURFACE_OBJECT_PATH\",\n  \"device\": \"AMD Radeon RX 9070 XT\",\n  \"function\": \"cc_tinlayout_fused_post_block_swin_1h_32_fp8\",\n  \"parameter_bytes\": 184,\n  \"registered_d3d12_buffers\": 2,\n  \"buffer_resource_registration\": \"AUTOMATIC_MODULE_TRACE\",\n  \"registered_surface_descriptors\": 1,\n  \"descriptor_resource_registration\": \"AUTOMATIC_MODULE_TRACE\",\n  \"address_translations\": "+std::to_string(diag.address_translations)+",\n  \"translation_failures\": "+std::to_string(diag.translation_failures)+",\n  \"independent_descriptor_calls\": "+std::to_string(diag.independent_descriptor_calls)+",\n  \"d3d12_texture_copies\": "+std::to_string(diag.d3d12_copy_dispatches)+",\n  \"neural_math_executed\": "+std::to_string(diag.neural_math_executed)+",\n  \"surface_byte_mismatches\": "+std::to_string(mismatches)+",\n  \"host_observed_dispatch_ms\": "+std::to_string(milliseconds)+",\n  \"output_fnv1a64\": \""+hash+"\",\n  \"counts_as_s6\": false,\n  \"game_runtime_ready\": false,\n  \"next_gate\": \"Provide queue-ordered HIP synchronization and native upstream decoder activations\"\n}\n";Write(argv[8],json.data(),json.size());
  destroyFunction(device.Get(),function);destroyModule(device.Get(),module);unregisterBuffer(modelGpu.Get());unregisterBuffer(activationGpu.Get());FreeLibrary(library);
  std::printf("[%s] NVAPI AMD output head: mismatches=%llu neural=%u translations=%u %.3f ms\n",pass?"PASS":"FAIL",(unsigned long long)mismatches,diag.neural_math_executed,diag.address_translations,milliseconds);return pass?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}}
