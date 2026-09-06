#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace full_graph_dx12 {
using Microsoft::WRL::ComPtr;
inline void Check(HRESULT result) { if(FAILED(result))throw std::runtime_error("full-graph D3D12 operation failed"); }
constexpr uint64_t ActivationArenaBytes=29773824;
constexpr uint64_t Slot0Bytes=110592;

class PrefixSlot0 {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    static void Transition(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,
                           D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){
        if(before==after)return;D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};list->ResourceBarrier(1,&barrier);
    }
public:
    PrefixSlot0(ID3D12Device* device,const std::vector<uint8_t>& shader):device_(device){
        if(!device||shader.empty()||shader.size()>1024*1024)throw std::invalid_argument("slot0 shader/device");
        D3D12_ROOT_PARAMETER parameter{};parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameter.Descriptor.ShaderRegister=0;parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC desc{};desc.NumParameters=1;desc.pParameters=&parameter;
        ComPtr<ID3DBlob> blob,error;Check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error));
        Check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root_)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};pipeline.pRootSignature=root_.Get();pipeline.CS={shader.data(),shader.size()};
        Check(device->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(&pipeline_)));
    }
    void Record(ID3D12GraphicsCommandList* list,ID3D12Resource* activation,D3D12_RESOURCE_STATES state)const{
        if(!list||!activation||(list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT&&list->GetType()!=D3D12_COMMAND_LIST_TYPE_COMPUTE))
            throw std::invalid_argument("slot0 list/resource");
        ComPtr<ID3D12Device> listDevice,resourceDevice;Check(list->GetDevice(IID_PPV_ARGS(&listDevice)));Check(activation->GetDevice(IID_PPV_ARGS(&resourceDevice)));
        if(listDevice.Get()!=device_.Get()||resourceDevice.Get()!=device_.Get())throw std::invalid_argument("slot0 device mismatch");
        auto d=activation->GetDesc();D3D12_HEAP_PROPERTIES heap{};D3D12_HEAP_FLAGS flags{};Check(activation->GetHeapProperties(&heap,&flags));
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER||d.Width<ActivationArenaBytes||heap.Type!=D3D12_HEAP_TYPE_DEFAULT||
           !(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))throw std::invalid_argument("slot0 activation contract");
        Transition(list,activation,state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(root_.Get());list->SetPipelineState(pipeline_.Get());
        list->SetComputeRootUnorderedAccessView(0,activation->GetGPUVirtualAddress());list->Dispatch(108,1,1);
        D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;barrier.UAV.pResource=activation;list->ResourceBarrier(1,&barrier);
        Transition(list,activation,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,state);
    }
};
}
