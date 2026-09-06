#pragma once
#include <d3d12.h>
namespace ffx_boundary {
// Entire callback analysis, never a claim that GPU work has completed.
struct Batch {
    UINT transitions=0,aliases=0,unknown=0;
    D3D12_RESOURCE_BARRIER target{};
    bool relevant=false,accepted=false;
};
inline Batch InspectBatch(ID3D12Resource* output,UINT count,const D3D12_RESOURCE_BARRIER* barriers) {
    Batch r;
    if(!output||!count||count>4096||!barriers){r.unknown=1;return r;}
    for(UINT i=0;i<count;++i) {
        const auto& b=barriers[i];
        if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING)++r.aliases;
        else if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION){
            if(b.Transition.pResource==output){++r.transitions;r.target=b;}
        }else if(b.Type!=D3D12_RESOURCE_BARRIER_TYPE_UAV)++r.unknown;
    }
    r.relevant=r.transitions||r.aliases||r.unknown;
    auto& t=r.target.Transition;
    r.accepted=r.transitions==1&&!r.aliases&&!r.unknown&&r.target.Flags==D3D12_RESOURCE_BARRIER_FLAG_NONE&&
        (t.Subresource==0||t.Subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)&&
        (t.StateBefore==D3D12_RESOURCE_STATE_UNORDERED_ACCESS||t.StateBefore==D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)&&
        t.StateAfter==(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return r;
}
}
