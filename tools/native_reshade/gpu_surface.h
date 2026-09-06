#pragma once
// SDR display-code-value preview, not scene-linear/HDR. All pixel conversions on GPU.
#include "../native_frame_bridge/native_frame_bridge.cpp"
#include <d3dcompiler.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
namespace native_preview {
struct Surface {
    Bridge bridge;
    ComPtr<ID3D12Resource> source;
    ComPtr<ID3D12DescriptorHeap> srvs,rtvs;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> decode,encode;
    DXGI_FORMAT format{};
    bool uncertain=false;
    static bool supported(DXGI_FORMAT f){return f==DXGI_FORMAT_R8G8B8A8_UNORM||f==DXGI_FORMAT_B8G8R8A8_UNORM||f==DXGI_FORMAT_R10G10B10A2_UNORM;}
    void init(ID3D12Device* d,ID3D12CommandQueue* q,UINT w,UINT h,DXGI_FORMAT f,UINT64 luid){
        if(!supported(f))throw std::runtime_error("SDR packed format only");format=f;
        bridge.init(w,h,false,luid,d,q);
        auto td=bridge.texture->GetDesc();td.Format=f;td.Flags=D3D12_RESOURCE_FLAG_NONE;
        D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
        hr(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&td,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&source)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=2;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(d->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&srvs)));hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        hr(d->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&rtvs)));
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,0,0,0};D3D12_ROOT_PARAMETER param{};
        param.ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;param.DescriptorTable={1,&range};param.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rd{1,&param,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};ComPtr<ID3DBlob> blob,errors;
        hr(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors));hr(d->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)));
        const char* shader="Texture2D<float4> t:register(t0); float4 VS(uint i:SV_VertexID):SV_Position {return float4(i==2?3:-1,i==1?3:-1,0,1);} float4 PS(float4 p:SV_Position):SV_Target{return t.Load(int3(int2(p.xy),0));}";
        ComPtr<ID3DBlob> vs,ps;hr(D3DCompile(shader,strlen(shader),nullptr,nullptr,nullptr,"VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&errors));
        hr(D3DCompile(shader,strlen(shader),nullptr,nullptr,nullptr,"PS","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&errors));
        for(int i=0;i<2;++i){D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};p.pRootSignature=root.Get();p.VS={vs->GetBufferPointer(),vs->GetBufferSize()};p.PS={ps->GetBufferPointer(),ps->GetBufferSize()};
            p.SampleMask=UINT_MAX;p.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;p.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;p.RasterizerState.DepthClipEnable=TRUE;
            p.BlendState.RenderTarget[0].RenderTargetWriteMask=i?7:15; // preserve actual game's alpha
            p.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;p.NumRenderTargets=1;p.RTVFormats[0]=i?f:DXGI_FORMAT_R16G16B16A16_FLOAT;p.SampleDesc.Count=1;
            hr(d->CreateGraphicsPipelineState(&p,IID_PPV_ARGS(i?&encode:&decode)));}
        D3D12_SHADER_RESOURCE_VIEW_DESC sv{};sv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;sv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;sv.Texture2D.MipLevels=1;
        auto handle=srvs->GetCPUDescriptorHandleForHeapStart();sv.Format=f;d->CreateShaderResourceView(source.Get(),&sv,handle);
        handle.ptr+=d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);sv.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d->CreateShaderResourceView(bridge.texture.Get(),&sv,handle);
        d->CreateRenderTargetView(bridge.texture.Get(),nullptr,rtvs->GetCPUDescriptorHandleForHeapStart());
    }
    void draw(ID3D12Resource* target,bool output){
        auto* d=bridge.device.Get();auto* c=bridge.cmd.Get();auto rtv=rtvs->GetCPUDescriptorHandleForHeapStart();
        if(output){rtv.ptr+=d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);d->CreateRenderTargetView(target,nullptr,rtv);}
        c->SetPipelineState(output?encode.Get():decode.Get());c->SetGraphicsRootSignature(root.Get());ID3D12DescriptorHeap* heaps[]={srvs.Get()};c->SetDescriptorHeaps(1,heaps);
        auto srv=srvs->GetGPUDescriptorHandleForHeapStart();if(output)srv.ptr+=d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        c->SetGraphicsRootDescriptorTable(0,srv);D3D12_VIEWPORT vp{0,0,float(bridge.width),float(bridge.height),0,1};D3D12_RECT sc{0,0,LONG(bridge.width),LONG(bridge.height)};
        c->RSSetViewports(1,&vp);c->RSSetScissorRects(1,&sc);c->OMSetRenderTargets(1,&rtv,FALSE,nullptr);c->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);c->DrawInstanced(3,1,0,0);
    }
    void validate(ID3D12Resource* target){
        auto td=target->GetDesc();ComPtr<ID3D12Device> d,reference;hr(target->GetDevice(IID_PPV_ARGS(&d)));
        // ReShade hooks Resource::GetDevice to return its proxy. Compare two
        // resource owners through the SAME public API, never proxy vs raw device.
        hr(bridge.texture->GetDevice(IID_PPV_ARGS(&reference)));
        ComPtr<IUnknown> a,b;hr(d.As(&a));hr(reference.As(&b));
        if(a!=b||td.Width!=bridge.width||td.Height!=bridge.height||td.Format!=format||td.SampleDesc.Count!=1||td.MipLevels!=1||td.DepthOrArraySize!=1)
            throw std::runtime_error("changed resource generation/geometry/device; stop, no stale output");
    }
    void input(ID3D12Resource* target){
        validate(target);bridge.require(0);bridge.begin();auto c=bridge.cmd.Get();
        transition(c,target,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(c,source.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);c->CopyResource(source.Get(),target);
        transition(c,target,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
        transition(c,source.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        transition(c,bridge.texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_RENDER_TARGET);draw(bridge.texture.Get(),false);
        transition(c,bridge.texture.Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COMMON);
        transition(c,source.Get(),D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        uncertain=true;bridge.submit();bridge.waitHost();uncertain=false;bridge.submitTexture();
    }
    void output(ID3D12Resource* target){
        validate(target);bridge.receiveTexture();bridge.begin();auto c=bridge.cmd.Get();
        transition(c,bridge.texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);draw(target,true);
        transition(c,bridge.texture.Get(),D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        uncertain=true;bridge.submit();bridge.waitHost();uncertain=false;
    }
};
}
