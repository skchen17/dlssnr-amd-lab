#include "nr_plan.h"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>

__global__ void plan_marker(const NRFrameBindings* bindings,const float* weights,
    float* workspace) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=16)return;
    const float* input=static_cast<const float*>(bindings->input);
    float* output=static_cast<float*>(bindings->output);
    workspace[i]=input[i]+weights[i]+float(bindings->frame_id);
    output[i]=workspace[i];
}

static hipError_t record(hipStream_t stream,void* workspace,const void* weights,
    const NRFrameBindings* bindings,void*) {
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
    if(nrPlanCreate(&desc,&plan)!=hipSuccess||nrPlanUploadWeights(plan,weights.data(),sizeof(weights))!=hipSuccess||
       nrPlanSetRecorder(plan,record,nullptr)!=hipSuccess||nrPlanFinalize(plan)!=hipSuccess)return 4;
    hipEvent_t done{};hipEventCreate(&done);
    NRFrameBindings bindings{device_input,device_output,1920,1080,7,1};
    if(nrPlanSubmit(plan,&bindings,nullptr,done)!=hipSuccess||hipEventSynchronize(done)!=hipSuccess)return 5;
    if(hipMemcpy(got.data(),device_output,sizeof(got),hipMemcpyDeviceToHost)!=hipSuccess)return 6;
    bindings={device_input2,device_output2,2560,1440,11,2};
    if(nrPlanSubmit(plan,&bindings,nullptr,done)!=hipSuccess||hipEventSynchronize(done)!=hipSuccess)return 7;
    if(hipMemcpy(got2.data(),device_output2,sizeof(got2),hipMemcpyDeviceToHost)!=hipSuccess)return 8;
    bool pass=true;for(int i=0;i<16;++i){
        pass&=got[i]==input[i]+weights[i]+7.f;
        pass&=got2[i]==input2[i]+weights[i]+11.f;
    }
    hipEventDestroy(done);nrPlanDestroy(plan);hipFree(device_input);hipFree(device_input2);
    hipFree(device_output);hipFree(device_output2);
    std::printf("{\"status\":\"%s\",\"graph_replays\":2,\"pointer_updates\":true,\"dynamic_frame_allocation\":false}\n",pass?"PASS":"FAIL");
    return pass?0:9;
}
