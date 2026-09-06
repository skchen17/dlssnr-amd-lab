#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace output_head_dx12 {
using Microsoft::WRL::ComPtr;
inline void Check(HRESULT result) {
    if (FAILED(result)) throw std::runtime_error("output-head D3D12 operation failed");
}
constexpr uint64_t MainBytes = 1966080, SkipBytes = 7864320, HeadBytes = 21808;
constexpr uint64_t HalfBytes = 16257024, E4Bytes = 8128512;
constexpr uint64_t SurfaceBytes = 640ull * 360 * 8;
constexpr uint32_t WorkItems = 81 * 49 * 64;
inline constexpr std::array<const char*, 11> ShaderNames = {
    "output_head_full_activation_d3d12", "output_head_full_first128_hidden_d3d12",
    "output_head_full_first128_project_d3d12", "output_head_full_mma128_175_d3d12",
    "output_head_full_prepare_qk_d3d12", "output_head_full_prepare_v_d3d12",
    "output_head_full_qk_d3d12", "output_head_full_softmax_v_d3d12",
    "output_head_final_projection_d3d12", "output_head_tail_d3d12",
    "output_head_surface_d3d12"
};

struct BufferView {
    ID3D12Resource* resource = nullptr;
    uint64_t offset = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};
struct TextureView {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};
struct Inputs {
    BufferView main, skip, head;
    TextureView base, output;
};

// Fixed captured 640x360 contract. Record() only appends commands: no file IO,
// Map, queue submission or CPU/GPU wait. External resource states are restored.
// Compute root signature, root arguments and PSO are clobbered; callers rebind.
// Keep this object and all input resources alive until the caller's fence passes.
// One instance may be reused in submission order on ONE queue; recording is not
// thread safe. Independent/concurrent queues require independent instances.
class Executor {
public:
    using ShaderSet = std::array<std::vector<uint8_t>, 11>;
    explicit Executor(ID3D12Device* device, const ShaderSet& shaders) : device_(device) {
        if (!device) throw std::invalid_argument("null output-head device");
        D3D12_ROOT_PARAMETER params[5]{};
        for (unsigned i = 0; i < 5; ++i) {
            params[i].ParameterType = i < 3 ? D3D12_ROOT_PARAMETER_TYPE_SRV : D3D12_ROOT_PARAMETER_TYPE_UAV;
            params[i].Descriptor.ShaderRegister = i < 3 ? i : i - 3;
            params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.NumParameters = 5; desc.pParameters = params;
        ComPtr<ID3DBlob> blob, error;
        Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_)));
        for (size_t i = 0; i < shaders.size(); ++i) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC p{};
            p.pRootSignature = root_.Get(); p.CS = {shaders[i].data(), shaders[i].size()};
            Check(device->CreateComputePipelineState(&p, IID_PPV_ARGS(&pipelines_[i])));
        }
        const uint64_t sizes[] = {E4Bytes, HalfBytes, E4Bytes * 4, HalfBytes,
            HalfBytes * 3, E4Bytes, E4Bytes, E4Bytes, HalfBytes * 2,
            HalfBytes, HalfBytes, 2032128, SurfaceBytes};
        for (size_t i = 0; i < scratch_.size(); ++i)
            scratch_[i] = Buffer(sizes[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        base_ = Buffer(SurfaceBytes, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    // Validate all external bindings before appending any GPU command.
    void Validate(ID3D12GraphicsCommandList* list, const Inputs& input) const {
        if (!list || (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT &&
                      list->GetType() != D3D12_COMMAND_LIST_TYPE_COMPUTE))
            throw std::invalid_argument("output head requires direct/compute list");
        ComPtr<ID3D12Device> listDevice;
        Check(list->GetDevice(IID_PPV_ARGS(&listDevice)));
        if (listDevice.Get() != device_.Get()) throw std::invalid_argument("list device mismatch");
        ValidateBuffer(input.main, MainBytes); ValidateBuffer(input.skip, SkipBytes);
        ValidateBuffer(input.head, HeadBytes);
        ValidateTexture(input.base); ValidateTexture(input.output);
        if (input.base.resource == input.output.resource)
            throw std::invalid_argument("in-place base/output is not supported");
        const BufferView views[] = {input.main, input.skip, input.head};
        for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < i; ++j)
                if (views[i].resource == views[j].resource && views[i].state != views[j].state)
                    throw std::invalid_argument("aliased buffer state disagreement");
    }

    void Record(ID3D12GraphicsCommandList* list, const Inputs& input,
                ID3D12Resource* residualReadback = nullptr) {
        Validate(list, input);
        if (residualReadback) {
            ValidateResource(residualReadback);
            D3D12_HEAP_PROPERTIES heap{}; D3D12_HEAP_FLAGS flags{};
            Check(residualReadback->GetHeapProperties(&heap, &flags));
            const auto desc = residualReadback->GetDesc();
            if (heap.Type != D3D12_HEAP_TYPE_READBACK ||
                desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < 2032128)
                throw std::invalid_argument("invalid diagnostic readback");
        }
        const BufferView views[] = {input.main, input.skip, input.head};
        for (unsigned i = 0; i < 3; ++i) {
            bool first = true;
            for (unsigned j = 0; j < i; ++j) first &= views[i].resource != views[j].resource;
            if (first) Transition(list, views[i].resource, views[i].state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        Transition(list, input.base.resource, input.base.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto source = TextureLocation(input.base.resource), destination = LinearLocation(base_.Get());
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        Transition(list, input.base.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, input.base.state);
        Transition(list, base_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->SetComputeRootSignature(root_.Get());
        auto address = [](const BufferView& v) { return v.resource->GetGPUVirtualAddress() + v.offset; };
        auto gpu = [&](unsigned i) { return scratch_[i]->GetGPUVirtualAddress(); };
        auto read = [&](unsigned i) { Transition(list, scratch_[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); };
        auto dispatch = [&](unsigned p, D3D12_GPU_VIRTUAL_ADDRESS a, D3D12_GPU_VIRTUAL_ADDRESS b,
                            D3D12_GPU_VIRTUAL_ADDRESS c, unsigned out, uint32_t groups, int second = -1) {
            list->SetPipelineState(pipelines_[p].Get());
            // Bind valid fallback addresses even for unused shader parameters.
            list->SetComputeRootShaderResourceView(0, a);
            list->SetComputeRootShaderResourceView(1, b ? b : a);
            list->SetComputeRootShaderResourceView(2, c ? c : a);
            list->SetComputeRootUnorderedAccessView(3, gpu(out));
            list->SetComputeRootUnorderedAccessView(4, gpu(second < 0 ? out : unsigned(second)));
            list->Dispatch(groups, 1, 1);
        };
        const auto model = address(input.head);
        dispatch(0, address(input.main), address(input.skip), model, 0, Groups(E4Bytes / 4, 256), 1); read(0); read(1);
        dispatch(1, gpu(0), model, 0, 2, Groups(E4Bytes, 256)); read(2);
        dispatch(2, gpu(1), gpu(2), model, 3, Groups(HalfBytes / 4, 256)); read(3);
        dispatch(3, gpu(3), model, 0, 4, Groups(HalfBytes * 3 / 4, 256)); read(4);
        dispatch(4, gpu(4), model, 0, 5, Groups(WorkItems, 64), 6); read(5); read(6);
        dispatch(5, gpu(4), 0, 0, 7, Groups(E4Bytes / 4, 256)); read(7);
        dispatch(6, gpu(5), gpu(6), model, 8, Groups(HalfBytes * 2 / 4, 256)); read(8);
        dispatch(7, gpu(8), gpu(7), 0, 9, Groups(WorkItems, 64)); read(9);
        dispatch(8, gpu(9), gpu(3), model, 10, Groups(WorkItems * 32, 256)); read(10);
        dispatch(9, gpu(10), model, 0, 11, Groups(WorkItems, 256)); read(11);
        dispatch(10, gpu(11), base_->GetGPUVirtualAddress(), 0, 12, Groups(WorkItems, 256));
        Transition(list, scratch_[12].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(list, input.output.resource, input.output.state, D3D12_RESOURCE_STATE_COPY_DEST);
        source = LinearLocation(scratch_[12].Get()); destination = TextureLocation(input.output.resource);
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        Transition(list, input.output.resource, D3D12_RESOURCE_STATE_COPY_DEST, input.output.state);
        Transition(list, scratch_[12].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (residualReadback) {
            Transition(list, scratch_[11].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(residualReadback, 0, scratch_[11].Get(), 0, 2032128);
            Transition(list, scratch_[11].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        for (unsigned i = 0; i < 12; ++i)
            Transition(list, scratch_[i].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Transition(list, base_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        for (unsigned i = 0; i < 3; ++i) {
            bool first = true;
            for (unsigned j = 0; j < i; ++j) first &= views[i].resource != views[j].resource;
            if (first) Transition(list, views[i].resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, views[i].state);
        }
    }

    static D3D12_TEXTURE_COPY_LOCATION LinearLocation(ID3D12Resource* resource) {
        D3D12_TEXTURE_COPY_LOCATION result{};
        result.pResource = resource; result.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        result.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, 640, 360, 1, 640 * 8};
        return result;
    }
    static D3D12_TEXTURE_COPY_LOCATION TextureLocation(ID3D12Resource* resource) {
        D3D12_TEXTURE_COPY_LOCATION result{};
        result.pResource = resource; result.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        return result;
    }
    static void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        if (before == after) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &barrier);
    }
private:
    static uint32_t Groups(uint64_t values, uint32_t threads) { return uint32_t((values + threads - 1) / threads); }
    void ValidateResource(ID3D12Resource* resource) const {
        if (!resource) throw std::invalid_argument("null output-head resource");
        ComPtr<ID3D12Device> device;
        Check(resource->GetDevice(IID_PPV_ARGS(&device)));
        if (device.Get() != device_.Get()) throw std::invalid_argument("resource device mismatch");
    }
    void ValidateBuffer(const BufferView& view, uint64_t bytes) const {
        ValidateResource(view.resource);
        const auto d = view.resource->GetDesc();
        D3D12_HEAP_PROPERTIES heap{}; D3D12_HEAP_FLAGS flags{};
        Check(view.resource->GetHeapProperties(&heap, &flags));
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || heap.Type != D3D12_HEAP_TYPE_DEFAULT ||
            (view.offset & 3) || view.offset > d.Width || bytes > d.Width - view.offset)
            throw std::invalid_argument("invalid default-buffer slice");
    }
    void ValidateTexture(const TextureView& view) const {
        ValidateResource(view.resource);
        const auto d = view.resource->GetDesc();
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.Width != 640 || d.Height != 360 ||
            d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || d.DepthOrArraySize != 1 ||
            d.MipLevels != 1 || d.SampleDesc.Count != 1)
            throw std::invalid_argument("output head requires single-sample 640x360 RGBA16F textures");
    }
    ComPtr<ID3D12Resource> Buffer(uint64_t bytes, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = bytes;
        d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> result;
        Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&result)));
        return result;
    }
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    std::array<ComPtr<ID3D12PipelineState>, 11> pipelines_;
    std::array<ComPtr<ID3D12Resource>, 13> scratch_;
    ComPtr<ID3D12Resource> base_;
};
} // namespace output_head_dx12
