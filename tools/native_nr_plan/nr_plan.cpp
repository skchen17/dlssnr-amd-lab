#include "nr_plan.h"
#include <new>

struct NRPlan {
    NRPlanDesc desc{};
    hipStream_t stream{};
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    void* workspace{};
    void* weights{};
    NRFrameBindings* device_bindings{};
    NRFrameBindings* pinned_bindings{};
    hipEvent_t binding_consumed{};
    NRPlanRecordFn record{};
    void* record_user{};
    bool finalized{};
    bool submitted{};
};

static hipError_t cleanup(NRPlan* plan) {
    if(!plan)return hipSuccess;
    hipError_t first=hipSuccess;
    auto keep=[&](hipError_t error){if(first==hipSuccess&&error!=hipSuccess)first=error;};
    if(plan->stream)keep(hipStreamSynchronize(plan->stream));
    if(plan->executable)keep(hipGraphExecDestroy(plan->executable));
    if(plan->graph)keep(hipGraphDestroy(plan->graph));
    if(plan->binding_consumed)keep(hipEventDestroy(plan->binding_consumed));
    if(plan->device_bindings)keep(hipFree(plan->device_bindings));
    if(plan->pinned_bindings)keep(hipHostFree(plan->pinned_bindings));
    if(plan->weights)keep(hipFree(plan->weights));
    if(plan->workspace)keep(hipFree(plan->workspace));
    if(plan->stream)keep(hipStreamDestroy(plan->stream));
    delete plan;
    return first;
}

NRPLAN_API hipError_t nrPlanCreate(const NRPlanDesc* desc, NRPlan** out_plan) {
    if(!desc||!out_plan||!desc->workspace_bytes||!desc->weight_bytes||
       !desc->max_width||!desc->max_height)return hipErrorInvalidValue;
    *out_plan=nullptr;
    auto* plan=new(std::nothrow) NRPlan;
    if(!plan)return hipErrorOutOfMemory;
    plan->desc=*desc;
    hipError_t error=hipStreamCreateWithFlags(&plan->stream,hipStreamNonBlocking);
    if(error==hipSuccess)error=hipMalloc(&plan->workspace,size_t(desc->workspace_bytes));
    if(error==hipSuccess)error=hipMalloc(&plan->weights,size_t(desc->weight_bytes));
    if(error==hipSuccess)error=hipMalloc(&plan->device_bindings,sizeof(NRFrameBindings));
    if(error==hipSuccess)error=hipHostMalloc(&plan->pinned_bindings,sizeof(NRFrameBindings));
    if(error==hipSuccess)error=hipEventCreateWithFlags(&plan->binding_consumed,hipEventDisableTiming);
    if(error!=hipSuccess){cleanup(plan);return error;}
    *out_plan=plan;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanUploadWeights(NRPlan* plan,const void* host_weights,uint64_t bytes) {
    if(!plan||!host_weights||bytes!=plan->desc.weight_bytes||plan->finalized)return hipErrorInvalidValue;
    hipError_t error=hipMemcpyAsync(plan->weights,host_weights,size_t(bytes),hipMemcpyHostToDevice,plan->stream);
    return error==hipSuccess?hipStreamSynchronize(plan->stream):error;
}

NRPLAN_API hipError_t nrPlanSetRecorder(NRPlan* plan,NRPlanRecordFn record,void* user) {
    if(!plan||!record||plan->finalized)return hipErrorInvalidValue;
    plan->record=record;plan->record_user=user;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanFinalize(NRPlan* plan) {
    if(!plan||!plan->record||plan->finalized)return hipErrorInvalidValue;
    hipError_t error=hipStreamBeginCapture(plan->stream,hipStreamCaptureModeGlobal);
    if(error!=hipSuccess)return error;
    error=plan->record(plan->stream,plan->workspace,plan->weights,
        plan->device_bindings,plan->record_user);
    hipGraph_t graph{};
    hipError_t end_error=hipStreamEndCapture(plan->stream,&graph);
    if(error!=hipSuccess){if(graph)hipGraphDestroy(graph);return error;}
    if(end_error!=hipSuccess)return end_error;
    hipGraphExec_t executable{};
    error=hipGraphInstantiate(&executable,graph,nullptr,nullptr,0);
    if(error!=hipSuccess){hipGraphDestroy(graph);return error;}
    plan->graph=graph;plan->executable=executable;plan->finalized=true;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanSubmit(NRPlan* plan,const NRFrameBindings* bindings,
    hipEvent_t input_ready,hipEvent_t output_done) {
    if(!plan||!bindings||!plan->finalized||!bindings->input||!bindings->output||
       !bindings->width||!bindings->height||bindings->width>plan->desc.max_width||
       bindings->height>plan->desc.max_height)return hipErrorInvalidValue;
    // One in-flight frame per plan keeps the pinned parameter block immutable
    // until its H2D copy has been consumed. Callers may create multiple plans
    // for deeper pipelining; there is no hidden synchronization or allocation.
    if(plan->submitted&&hipEventQuery(plan->binding_consumed)!=hipSuccess)return hipErrorNotReady;
    *plan->pinned_bindings=*bindings;
    hipError_t error=hipSuccess;
    if(input_ready)error=hipStreamWaitEvent(plan->stream,input_ready,0);
    if(error==hipSuccess)error=hipMemcpyAsync(plan->device_bindings,plan->pinned_bindings,
        sizeof(NRFrameBindings),hipMemcpyHostToDevice,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->binding_consumed,plan->stream);
    if(error==hipSuccess)error=hipGraphLaunch(plan->executable,plan->stream);
    if(error==hipSuccess&&output_done)error=hipEventRecord(output_done,plan->stream);
    if(error==hipSuccess)plan->submitted=true;
    return error;
}

NRPLAN_API hipError_t nrPlanReset(NRPlan* plan) {
    if(!plan)return hipErrorInvalidValue;
    hipError_t error=hipStreamSynchronize(plan->stream);
    if(error==hipSuccess)error=hipMemsetAsync(plan->workspace,0,size_t(plan->desc.workspace_bytes),plan->stream);
    if(error==hipSuccess)error=hipStreamSynchronize(plan->stream);
    plan->submitted=false;
    return error;
}

NRPLAN_API hipError_t nrPlanDestroy(NRPlan* plan) {return cleanup(plan);}

