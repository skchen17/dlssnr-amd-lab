// Resident AMD output head with a D3D12 texture staging boundary in the same
// process. The included implementation supplies the already validated kernels;
// this file replaces its CLI with the D3D12/HIP interop validation harness.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define main output_head_resident_file_cli
#include "../output_head_resident/output_head_resident.cpp"
#undef main

using Microsoft::WRL::ComPtr;

#define HR_CHECK(x) do { HRESULT h_=(x); if(FAILED(h_)){char b_[160]; \
  std::snprintf(b_,sizeof(b_),#x " failed: 0x%08lX",(unsigned long)h_); \
  throw std::runtime_error(b_);} } while(0)

namespace bridge {
size_t Align(size_t value,size_t alignment){return (value+alignment-1)&~(alignment-1);}

D3D12_RESOURCE_DESC BufferDesc(UINT64 bytes){
  D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;
  d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;
  d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;return d;
}

D3D12_RESOURCE_DESC TextureDesc(){
  D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width=WIDTH;d.Height=HEIGHT;d.DepthOrArraySize=1;d.MipLevels=1;
  d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d.SampleDesc.Count=1;
  d.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN;return d;
}

void Transition(ID3D12GraphicsCommandList*list,ID3D12Resource*resource,
                D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){
  D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition.pResource=resource;b.Transition.StateBefore=before;
  b.Transition.StateAfter=after;b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  list->ResourceBarrier(1,&b);
}
}

int main(int argc,char**argv){try{
  if(argc<8||argc>9){
    std::fprintf(stderr,"usage: output_head_resident_d3d12 <activation_arena.raw> <model_arena.raw> <base_rgba16f.raw|-> <reference_residual.raw|-> <residual_out.raw> <surface_out.raw> <result.json> [iterations]\n");
    return 2;
  }
  const uint32_t iterations=argc==9?uint32_t(std::stoul(argv[8])):20;
  if(!iterations)throw std::runtime_error("iterations must be positive");
  const auto mainFeature=Slice(argv[1],MAIN_OFF,MAIN_BYTES);
  const auto skipFeature=Slice(argv[1],SKIP_OFF,SKIP_BYTES);
  const auto head=Slice(argv[2],HEAD_OFF,HEAD_BYTES);
  const bool hasBase=std::string(argv[3])!="-",hasReference=std::string(argv[4])!="-";
  const auto baseBytes=hasBase?Read(argv[3]):std::vector<uint8_t>(SURFACE_VALUES*2,0);
  const auto reference=hasReference?Read(argv[4]):std::vector<uint8_t>{};
  if(baseBytes.size()!=SURFACE_VALUES*2)throw std::runtime_error("unexpected base size");
  if(hasReference&&reference.size()!=RESIDUAL_VALUES*2)throw std::runtime_error("unexpected reference size");

  ComPtr<IDXGIFactory7> factory;HR_CHECK(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory)));
  ComPtr<IDXGIAdapter1> adapter;DXGI_ADAPTER_DESC1 adapterDesc{};bool found=false;
  for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()){
    HR_CHECK(adapter->GetDesc1(&adapterDesc));
    if(!(adapterDesc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&adapterDesc.VendorId==0x1002){found=true;break;}
  }
  if(!found)throw std::runtime_error("AMD D3D12 adapter not found");
  HIP_CHECK(hipSetDevice(0));hipDeviceProp_t hipProp{};HIP_CHECK(hipGetDeviceProperties(&hipProp,0));
  ComPtr<ID3D12Device> device;HR_CHECK(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)));
  D3D12_COMMAND_QUEUE_DESC queueDesc{};queueDesc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
  ComPtr<ID3D12CommandQueue> queue;HR_CHECK(device->CreateCommandQueue(&queueDesc,IID_PPV_ARGS(&queue)));
  ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> list;
  HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
  HR_CHECK(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
  HR_CHECK(list->Close());
  ComPtr<ID3D12Fence> sharedFence,cpuFence;
  HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&sharedFence)));
  HR_CHECK(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&cpuFence)));
  UINT64 cpuValue=0;
  auto executeAndWait=[&](){ID3D12CommandList*lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);HR_CHECK(queue->Signal(cpuFence.Get(),++cpuValue));while(cpuFence->GetCompletedValue()<cpuValue)Sleep(0);};

  const auto textureDesc=bridge::TextureDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows=0;UINT64 rowBytes=0,textureCopyBytes=0;
  device->GetCopyableFootprints(&textureDesc,0,1,0,&footprint,&rows,&rowBytes,&textureCopyBytes);
  if(rows!=HEIGHT||rowBytes!=WIDTH*8)throw std::runtime_error("unexpected RGBA16F footprint");
  const size_t baseOffset=0;
  const size_t outputOffset=bridge::Align(size_t(textureCopyBytes),D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
  const size_t heapBytes=bridge::Align(outputOffset+size_t(textureCopyBytes),D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
  D3D12_HEAP_DESC heapDesc{};heapDesc.SizeInBytes=heapBytes;heapDesc.Properties.Type=D3D12_HEAP_TYPE_DEFAULT;
  heapDesc.Flags=D3D12_HEAP_FLAG_SHARED|D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
  ComPtr<ID3D12Heap> sharedHeap;HR_CHECK(device->CreateHeap(&heapDesc,IID_PPV_ARGS(&sharedHeap)));
  const auto sharedDesc=bridge::BufferDesc(heapBytes);ComPtr<ID3D12Resource> sharedBuffer;
  HR_CHECK(device->CreatePlacedResource(sharedHeap.Get(),0,&sharedDesc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&sharedBuffer)));
  D3D12_HEAP_PROPERTIES defaultHeap{};defaultHeap.Type=D3D12_HEAP_TYPE_DEFAULT;
  ComPtr<ID3D12Resource> baseTexture,outputTexture;
  HR_CHECK(device->CreateCommittedResource(&defaultHeap,D3D12_HEAP_FLAG_NONE,&textureDesc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&baseTexture)));
  HR_CHECK(device->CreateCommittedResource(&defaultHeap,D3D12_HEAP_FLAG_NONE,&textureDesc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&outputTexture)));
  D3D12_HEAP_PROPERTIES uploadHeap{};uploadHeap.Type=D3D12_HEAP_TYPE_UPLOAD;
  D3D12_HEAP_PROPERTIES readbackHeap{};readbackHeap.Type=D3D12_HEAP_TYPE_READBACK;
  const auto textureBufferDesc=bridge::BufferDesc(textureCopyBytes);
  ComPtr<ID3D12Resource> upload,readback;
  HR_CHECK(device->CreateCommittedResource(&uploadHeap,D3D12_HEAP_FLAG_NONE,&textureBufferDesc,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload)));
  HR_CHECK(device->CreateCommittedResource(&readbackHeap,D3D12_HEAP_FLAG_NONE,&textureBufferDesc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)));
  void*mapped=nullptr;HR_CHECK(upload->Map(0,nullptr,&mapped));std::memset(mapped,0,size_t(textureCopyBytes));
  for(uint32_t y=0;y<HEIGHT;++y)std::memcpy(static_cast<uint8_t*>(mapped)+size_t(footprint.Footprint.RowPitch)*y,baseBytes.data()+size_t(y)*WIDTH*8,WIDTH*8);
  upload->Unmap(0,nullptr);
  HR_CHECK(allocator->Reset());HR_CHECK(list->Reset(allocator.Get(),nullptr));
  bridge::Transition(list.Get(),baseTexture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
  D3D12_TEXTURE_COPY_LOCATION baseDst{},uploadSrc{};baseDst.pResource=baseTexture.Get();baseDst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  uploadSrc.pResource=upload.Get();uploadSrc.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;uploadSrc.PlacedFootprint=footprint;
  list->CopyTextureRegion(&baseDst,0,0,0,&uploadSrc,nullptr);
  bridge::Transition(list.Get(),baseTexture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
  bridge::Transition(list.Get(),sharedBuffer.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
  D3D12_TEXTURE_COPY_LOCATION sharedBase{},baseSrc{};sharedBase.pResource=sharedBuffer.Get();sharedBase.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;sharedBase.PlacedFootprint=footprint;sharedBase.PlacedFootprint.Offset=baseOffset;
  baseSrc.pResource=baseTexture.Get();baseSrc.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  list->CopyTextureRegion(&sharedBase,0,0,0,&baseSrc,nullptr);
  bridge::Transition(list.Get(),sharedBuffer.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
  bridge::Transition(list.Get(),baseTexture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
  HR_CHECK(list->Close());executeAndWait();HR_CHECK(queue->Signal(sharedFence.Get(),1));

  HANDLE heapHandle=nullptr,fenceHandle=nullptr;
  HR_CHECK(device->CreateSharedHandle(sharedHeap.Get(),nullptr,GENERIC_ALL,nullptr,&heapHandle));
  HR_CHECK(device->CreateSharedHandle(sharedFence.Get(),nullptr,GENERIC_ALL,nullptr,&fenceHandle));
  hipExternalMemoryHandleDesc memoryDesc{};memoryDesc.type=hipExternalMemoryHandleTypeD3D12Heap;memoryDesc.handle.win32.handle=heapHandle;memoryDesc.size=heapBytes;
  hipExternalMemory_t externalMemory=nullptr;HIP_CHECK(hipImportExternalMemory(&externalMemory,&memoryDesc));
  hipExternalMemoryBufferDesc mappingDesc{};mappingDesc.size=heapBytes;void*sharedPointer=nullptr;
  HIP_CHECK(hipExternalMemoryGetMappedBuffer(&sharedPointer,externalMemory,&mappingDesc));
  hipExternalSemaphoreHandleDesc semaphoreDesc{};semaphoreDesc.type=hipExternalSemaphoreHandleTypeD3D12Fence;semaphoreDesc.handle.win32.handle=fenceHandle;
  hipExternalSemaphore_t externalFence=nullptr;HIP_CHECK(hipImportExternalSemaphore(&externalFence,&semaphoreDesc));CloseHandle(heapHandle);CloseHandle(fenceHandle);
  auto*sharedBytes=static_cast<uint8_t*>(sharedPointer);
  auto*deviceBase=reinterpret_cast<const uint16_t*>(sharedBytes+baseOffset);
  auto*deviceSurface=reinterpret_cast<uint16_t*>(sharedBytes+outputOffset);

  uint8_t *dm=nullptr,*ds=nullptr,*dh=nullptr,*da=nullptr,*dhidden=nullptr,*dq=nullptr,*dk=nullptr,*dv=nullptr,*dsoft=nullptr;
  uint16_t *dah=nullptr,*dfirst=nullptr,*dp176=nullptr,*dqk=nullptr,*datt=nullptr,*dproj=nullptr,*dres=nullptr;
  HIP_CHECK(hipMalloc(&dm,mainFeature.size()));HIP_CHECK(hipMalloc(&ds,skipFeature.size()));HIP_CHECK(hipMalloc(&dh,head.size()));
  HIP_CHECK(hipMalloc(&da,A_BYTES));HIP_CHECK(hipMalloc(&dah,H_BYTES));HIP_CHECK(hipMalloc(&dhidden,HIDDEN_BYTES));HIP_CHECK(hipMalloc(&dfirst,H_BYTES));
  HIP_CHECK(hipMalloc(&dp176,PROJ176_VALUES*2));HIP_CHECK(hipMalloc(&dq,A_BYTES));HIP_CHECK(hipMalloc(&dk,A_BYTES));HIP_CHECK(hipMalloc(&dv,A_BYTES));
  HIP_CHECK(hipMalloc(&dqk,QK_VALUES*2));HIP_CHECK(hipMalloc(&dsoft,SOFT_BYTES));HIP_CHECK(hipMalloc(&datt,H_BYTES));HIP_CHECK(hipMalloc(&dproj,H_BYTES));HIP_CHECK(hipMalloc(&dres,RESIDUAL_VALUES*2));
  HIP_CHECK(hipMemcpy(dm,mainFeature.data(),mainFeature.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(ds,skipFeature.data(),skipFeature.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dh,head.data(),head.size(),hipMemcpyHostToDevice));
  float sf=0;std::memcpy(&sf,head.data()+19664,4);const uint16_t scale=__half_as_ushort(__float2half_rn(sf));
  const uint16_t eps=__half_as_ushort(__float2half_rn(6.199999916134402e-05f));
  hipStream_t stream{};HIP_CHECK(hipStreamCreate(&stream));
  hipExternalSemaphoreWaitParams waitParams{};waitParams.params.fence.value=1;
  HIP_CHECK(hipWaitExternalSemaphoresAsync(&externalFence,&waitParams,1,stream));
  auto grid=[](size_t n){return dim3(uint32_t((n+255)/256));};const dim3 block(256);
  auto run=[&](){
    Activation<<<grid(VALUES),block,0,stream>>>(dm,ds,dh,da,dah);Hidden<<<grid(VALUES*4),block,0,stream>>>(da,dh,dhidden);First128<<<grid(VALUES),block,0,stream>>>(dah,dhidden,dh,dfirst);
    Pack176<<<grid(VALUES),block,0,stream>>>(dfirst,da);Project176<<<grid(PROJ176_VALUES),block,0,stream>>>(da,dh,dp176);
    PrepQK<<<grid(CTAS*64),block,0,stream>>>(dp176,dq,dk,scale,eps);PrepV<<<grid(VALUES),block,0,stream>>>(dp176,dv);QK<<<grid(QK_VALUES),block,0,stream>>>(dq,dk,dh,dqk);
    Softmax<<<grid(CTAS*64),block,0,stream>>>(dqk,dsoft);Attention<<<grid(VALUES),block,0,stream>>>(dsoft,dv,datt);
    FinalPack<<<grid(VALUES),block,0,stream>>>(datt,dq);FinalProject<<<grid(VALUES),block,0,stream>>>(dq,dfirst,dh,dproj);Tail<<<grid(RESIDUAL_VALUES),block,0,stream>>>(dproj,dh,dres);
    Surface<<<grid(CTAS*64),block,0,stream>>>(dres,deviceBase,deviceSurface);
  };
  HIP_CHECK(hipMemsetAsync(deviceSurface,0,textureCopyBytes,stream));run();HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<uint16_t>firstResidual(RESIDUAL_VALUES);HIP_CHECK(hipMemcpy(firstResidual.data(),dres,RESIDUAL_VALUES*2,hipMemcpyDeviceToHost));
  hipEvent_t start{},stop{};HIP_CHECK(hipEventCreate(&start));HIP_CHECK(hipEventCreate(&stop));HIP_CHECK(hipEventRecord(start,stream));
  for(uint32_t i=0;i<iterations;++i)run();HIP_CHECK(hipEventRecord(stop,stream));HIP_CHECK(hipEventSynchronize(stop));float totalMs=0;HIP_CHECK(hipEventElapsedTime(&totalMs,start,stop));
  std::vector<uint16_t>residual(RESIDUAL_VALUES);HIP_CHECK(hipMemcpy(residual.data(),dres,RESIDUAL_VALUES*2,hipMemcpyDeviceToHost));
  hipExternalSemaphoreSignalParams signalParams{};signalParams.params.fence.value=2;HIP_CHECK(hipSignalExternalSemaphoresAsync(&externalFence,&signalParams,1,stream));HIP_CHECK(hipStreamSynchronize(stream));

  HR_CHECK(queue->Wait(sharedFence.Get(),2));HR_CHECK(allocator->Reset());HR_CHECK(list->Reset(allocator.Get(),nullptr));
  bridge::Transition(list.Get(),sharedBuffer.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
  bridge::Transition(list.Get(),outputTexture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
  D3D12_TEXTURE_COPY_LOCATION outputDst{},sharedOutput{};outputDst.pResource=outputTexture.Get();outputDst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  sharedOutput.pResource=sharedBuffer.Get();sharedOutput.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;sharedOutput.PlacedFootprint=footprint;sharedOutput.PlacedFootprint.Offset=outputOffset;
  list->CopyTextureRegion(&outputDst,0,0,0,&sharedOutput,nullptr);
  bridge::Transition(list.Get(),outputTexture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
  D3D12_TEXTURE_COPY_LOCATION readbackDst{},outputSrc{};readbackDst.pResource=readback.Get();readbackDst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;readbackDst.PlacedFootprint=footprint;
  outputSrc.pResource=outputTexture.Get();outputSrc.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;list->CopyTextureRegion(&readbackDst,0,0,0,&outputSrc,nullptr);
  bridge::Transition(list.Get(),outputTexture.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
  bridge::Transition(list.Get(),sharedBuffer.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
  HR_CHECK(list->Close());executeAndWait();
  std::vector<uint16_t>surface(SURFACE_VALUES);HR_CHECK(readback->Map(0,nullptr,&mapped));
  for(uint32_t y=0;y<HEIGHT;++y)std::memcpy(surface.data()+size_t(y)*WIDTH*4,static_cast<const uint8_t*>(mapped)+size_t(footprint.Footprint.RowPitch)*y,WIDTH*8);
  readback->Unmap(0,nullptr);

  uint64_t referenceMismatches=0,surfaceMismatches=0,nonfinite=0,changed=0;
  if(hasReference){const auto*r=reinterpret_cast<const uint16_t*>(reference.data());for(size_t i=0;i<RESIDUAL_VALUES;++i)referenceMismatches+=residual[i]!=r[i];}
  std::vector<uint16_t>expected(SURFACE_VALUES,0);const auto*base=reinterpret_cast<const uint16_t*>(baseBytes.data());
  for(size_t cta=0;cta<CTAS;++cta)for(uint32_t token=0;token<TOKENS;++token){uint32_t g=token/16,w=token%16;int32_t x=int32_t((cta%GW)*8)-4+int32_t((g&1)*4+w%4),y=int32_t((cta/GW)*8)-4+int32_t((g>>1)*4+w/4);if(x<0||y<0||x>=int32_t(WIDTH)||y>=int32_t(HEIGHT))continue;size_t s=(cta*64+token)*4,d=(size_t(y)*WIDTH+uint32_t(x))*4;for(uint32_t c=0;c<3;++c){float bv=__half2float(__ushort_as_half(base[d+c])),rv=__half2float(__ushort_as_half(residual[s+c]));expected[d+c]=__half_as_ushort(__float2half_rn(std::min(1.0f,std::max(0.0f,bv+.25f*rv))));changed+=expected[d+c]!=base[d+c];}expected[d+3]=__half_as_ushort(__float2half_rn(1));}
  for(size_t i=0;i<SURFACE_VALUES;++i){surfaceMismatches+=surface[i]!=expected[i];nonfinite+=!std::isfinite(__half2float(__ushort_as_half(surface[i])));}
  const bool deterministic=firstResidual==residual;
  const bool pass=deterministic&&referenceMismatches==0&&surfaceMismatches==0&&nonfinite==0&&changed>0;
  Write(argv[5],residual.data(),RESIDUAL_VALUES*2);Write(argv[6],surface.data(),SURFACE_VALUES*2);
  char rh[32]{},sh[32]{};std::snprintf(rh,sizeof(rh),"%016llX",(unsigned long long)Hash(residual.data(),RESIDUAL_VALUES*2));std::snprintf(sh,sizeof(sh),"%016llX",(unsigned long long)Hash(surface.data(),SURFACE_VALUES*2));
  const std::string json="{\n  \"schema\": 1,\n  \"experiment\": \"amd_output_head_resident_d3d12\",\n  \"status\": \""+std::string(pass?"PASS":"FAIL")+"\",\n  \"classification\": \"AMD_NATIVE_RESIDENT_D3D12_OUTPUT_HEAD\",\n  \"hip_device\": \""+std::string(hipProp.name)+"\",\n  \"d3d12_vendor_id\": \"0x1002\",\n  \"single_process\": true,\n  \"resident_gpu_intermediates\": true,\n  \"runtime_rtx_trace_dependency\": false,\n  \"d3d12_texture_staging\": true,\n  \"external_fence_wait_signal\": true,\n  \"reference_half_mismatches\": "+std::to_string(referenceMismatches)+",\n  \"d3d12_surface_component_mismatches\": "+std::to_string(surfaceMismatches)+",\n  \"deterministic_repeat\": "+std::string(deterministic?"true":"false")+",\n  \"non_finite_surface_components\": "+std::to_string(nonfinite)+",\n  \"changed_rgb_components\": "+std::to_string(changed)+",\n  \"iterations\": "+std::to_string(iterations)+",\n  \"average_gpu_output_head_ms\": "+std::to_string(totalMs/float(iterations))+",\n  \"residual_fnv1a64\": \""+rh+"\",\n  \"surface_fnv1a64\": \""+sh+"\",\n  \"game_runtime_ready\": false,\n  \"next_gate\": \"Bind feature-18 game resources and execute at the evaluate boundary\"\n}\n";
  Write(argv[7],json.data(),json.size());
  std::printf("[%s] %s resident D3D12 output head: ref=%llu surface=%llu %.6f ms\n",pass?"PASS":"FAIL",hipProp.name,(unsigned long long)referenceMismatches,(unsigned long long)surfaceMismatches,totalMs/float(iterations));

  HIP_CHECK(hipEventDestroy(start));HIP_CHECK(hipEventDestroy(stop));HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dres));HIP_CHECK(hipFree(dproj));HIP_CHECK(hipFree(datt));HIP_CHECK(hipFree(dsoft));HIP_CHECK(hipFree(dqk));HIP_CHECK(hipFree(dv));HIP_CHECK(hipFree(dk));HIP_CHECK(hipFree(dq));HIP_CHECK(hipFree(dp176));HIP_CHECK(hipFree(dfirst));HIP_CHECK(hipFree(dhidden));HIP_CHECK(hipFree(dah));HIP_CHECK(hipFree(da));HIP_CHECK(hipFree(dh));HIP_CHECK(hipFree(ds));HIP_CHECK(hipFree(dm));
  HIP_CHECK(hipDestroyExternalSemaphore(externalFence));HIP_CHECK(hipDestroyExternalMemory(externalMemory));return pass?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}}
