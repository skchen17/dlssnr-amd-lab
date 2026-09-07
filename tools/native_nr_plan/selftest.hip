#include "nr_plan.h"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>

__global__ void plan_marker(const NRFrameBindingsV3* bindings,const float* weights,
    float* workspace) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=16)return;
    const float* input=static_cast<const float*>(bindings->current_color);
    float* output=static_cast<float*>(bindings->output_residual);
    workspace[i]=input[i]+weights[i]+float(bindings->frame_id);
    output[i]=workspace[i];
}

static hipError_t record(hipStream_t stream,void* workspace,const void* weights,
    const NRFrameBindingsV3* bindings,void*) {
    hipLaunchKernelGGL(plan_marker,dim3(1),dim3(32),0,stream,bindings,
        static_cast<const float*>(weights),static_cast<float*>(workspace));
    return hipGetLastError();
}

int main() {
    std::array<float,16> input{},input2{},weights{},got{},got2{};
    for(int i=0;i<16;++i){input[i]=float(i);input2[i]=float(i)*2.f;weights[i]=float(i)*0.25f;}
    float *device_input{},*device_input2{},*device_output{},*device_output2{};
    if(hipMalloc(&device_input,sizeof(input))!=hipSuccess||
       hipMalloc(&device_input2,sizeof(input2))!=hipSuccess||
       hipMalloc(&device_output,sizeof(got))!=hipSuccess||
       hipMalloc(&device_output2,sizeof(got2))!=hipSuccess)return 2;
    if(hipMemcpy(device_input,input.data(),sizeof(input),hipMemcpyHostToDevice)!=hipSuccess)return 3;
    if(hipMemcpy(device_input2,input2.data(),sizeof(input2),hipMemcpyHostToDevice)!=hipSuccess)return 3;
    NRPlanDesc desc{sizeof(got),sizeof(weights),3840,2160};NRPlan* plan{};
    NRPlanArenaRegion arena{0,sizeof(got),16,1,"selftest_workspace"};
    NRShapeDesc shape{};shape.struct_size=sizeof(shape);shape.render_width=1920;
    shape.render_height=1080;shape.output_width=1920;shape.output_height=1080;
    shape.valid_width=1920;shape.valid_height=1080;shape.color_mode=NR_COLOR_SDR_LINEAR;
    shape.color_format=NR_FORMAT_RGBA16_FLOAT;shape.motion_format=NR_FORMAT_RG16_FLOAT;
    shape.depth_format=NR_FORMAT_R32_FLOAT;
    if(nrPlanCreate(&desc,&plan)!=hipSuccess||nrPlanUploadWeights(plan,weights.data(),sizeof(weights))!=hipSuccess||
       nrPlanConfigureArena(plan,&arena,1)!=hipSuccess||
       nrPlanPrepareShape(plan,&shape)!=hipSuccess||
       nrPlanSetRecorderV3(plan,record,nullptr)!=hipSuccess||nrPlanFinalize(plan)!=hipSuccess)return 4;
    NRPlanGraphStats graph_stats{};NRPlanResourceStats resource_stats{};NRPlanKernelNodeInfo kernel_info{};
    if(nrPlanGetGraphStats(plan,&graph_stats)!=hipSuccess||
       nrPlanGetResourceStats(plan,&resource_stats)!=hipSuccess||
       nrPlanGetKernelNodeInfo(plan,0,&kernel_info)!=hipSuccess)return 10;
    hipEvent_t done{};hipEventCreate(&done);
    NRFrameBindingsV3 bindings{};bindings.struct_size=sizeof(bindings);
    bindings.abi_version=NR_PLAN_FRAME_BINDINGS_ABI_VERSION_V3;
    bindings.current_color=device_input;bindings.output_residual=device_output;
    bindings.render_width=1920;bindings.render_height=1080;
    bindings.output_width=1920;bindings.output_height=1080;
    bindings.valid_width=1920;bindings.valid_height=1080;
    bindings.pre_exposure=1.f;bindings.exposure_scale=1.f;
    bindings.reset=1;bindings.frame_id=7;bindings.resource_generation=1;
    if(nrPlanSubmitV3(plan,&bindings,nullptr,done)!=hipSuccess||hipEventSynchronize(done)!=hipSuccess)return 5;
    if(hipMemcpy(got.data(),device_output,sizeof(got),hipMemcpyDeviceToHost)!=hipSuccess)return 6;
    bindings.current_color=device_input2;bindings.output_residual=device_output2;
    bindings.frame_id=11;bindings.resource_generation=2;
    if(nrPlanSubmitV3(plan,&bindings,nullptr,done)!=hipSuccess||hipEventSynchronize(done)!=hipSuccess)return 7;
    if(hipMemcpy(got2.data(),device_output2,sizeof(got2),hipMemcpyDeviceToHost)!=hipSuccess)return 8;
    bool pass=true;for(int i=0;i<16;++i){
        pass&=got[i]==input[i]+weights[i]+7.f;
        pass&=got2[i]==input2[i]+weights[i]+11.f;
    }
    NRPlanPerformanceStats performance{};
    if(nrPlanGetPerformanceStats(plan,&performance)!=hipSuccess)return 11;
    hipEventDestroy(done);nrPlanDestroy(plan);hipFree(device_input);hipFree(device_input2);
    hipFree(device_output);hipFree(device_output2);
    pass&=graph_stats.kernel_nodes==1&&resource_stats.owns_workspace&&resource_stats.owns_weights&&
          resource_stats.owns_graph_source&&resource_stats.owns_graph_executable&&
          !resource_stats.graph_references_external_allocations&&resource_stats.arena_region_count==1&&
          resource_stats.deployment_ready&&
          resource_stats.frame_bindings_abi_version==NR_PLAN_FRAME_BINDINGS_ABI_VERSION_V3&&
          performance.submitted_frames==2&&performance.completed_frames>=1&&
          kernel_info.grid_x==1&&kernel_info.block_x==32&&kernel_info.name[0];
    std::printf("{\"status\":\"%s\",\"graph_replays\":2,\"pointer_updates\":true,"
                "\"dynamic_frame_allocation\":false,\"kernel_name\":\"%s\","
                "\"cpp_owned_resources\":true}\n",pass?"PASS":"FAIL",kernel_info.name);
    return pass?0:9;
}
