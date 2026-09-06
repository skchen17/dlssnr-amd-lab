#include "nr_plan.h"
#include <new>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

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
    void* static_input{};
    void* static_output{};
    uint64_t input_bytes{};
    uint64_t output_bytes{};
    uint32_t static_width{};
    uint32_t static_height{};
    bool owns_graph{};
    bool finalized{};
    bool submitted{};
};

struct InstantiateTask {
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    hipError_t error{hipErrorUnknown};
    int device{};
};

#ifdef _WIN32
static DWORD WINAPI instantiate_worker(void* raw) {
    auto* task=static_cast<InstantiateTask*>(raw);
    task->error=hipSetDevice(task->device);
    if(task->error==hipSuccess)
        task->error=hipGraphInstantiate(&task->executable,task->graph,nullptr,nullptr,0);
    return 0;
}
#endif

static hipError_t instantiate_graph(hipGraph_t graph,hipGraphExec_t* executable) {
    if(!graph||!executable)return hipErrorInvalidValue;
#ifdef _WIN32
    InstantiateTask task{};task.graph=graph;
    hipError_t error=hipGetDevice(&task.device);
    if(error!=hipSuccess)return error;
    // ROCm's Windows graph instantiator recursively walks long dependency
    // chains. The default process stack overflows on the 640-wide 71-block
    // graph, so reserve an initialization-only 64 MiB host stack.
    HANDLE thread=CreateThread(nullptr,64ull*1024*1024,instantiate_worker,&task,
                               STACK_SIZE_PARAM_IS_A_RESERVATION,nullptr);
    if(!thread)return hipErrorUnknown;
    DWORD wait=WaitForSingleObject(thread,INFINITE);CloseHandle(thread);
    if(wait!=WAIT_OBJECT_0)return hipErrorUnknown;
    if(task.error==hipSuccess)*executable=task.executable;
    return task.error;
#else
    return hipGraphInstantiate(executable,graph,nullptr,nullptr,0);
#endif
}

static hipError_t cleanup(NRPlan* plan) {
    if(!plan)return hipSuccess;
    hipError_t first=hipSuccess;
    auto keep=[&](hipError_t error){if(first==hipSuccess&&error!=hipSuccess)first=error;};
    if(plan->stream)keep(hipStreamSynchronize(plan->stream));
    if(plan->executable)keep(hipGraphExecDestroy(plan->executable));
    if(plan->graph&&plan->owns_graph)keep(hipGraphDestroy(plan->graph));
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

NRPLAN_API hipError_t nrPlanGetStream(NRPlan* plan,hipStream_t* out_stream) {
    if(!plan||!out_stream)return hipErrorInvalidValue;
    *out_stream=plan->stream;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanAdoptGraph(NRPlan* plan,hipGraph_t source_graph) {
    if(!plan||!source_graph||plan->finalized||plan->record)return hipErrorInvalidValue;
    hipGraphExec_t executable{};
    hipError_t error=instantiate_graph(source_graph,&executable);
    if(error!=hipSuccess)return error;
    plan->graph=source_graph;plan->owns_graph=false;
    plan->executable=executable;plan->finalized=true;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanGetGraphStats(NRPlan* plan,NRPlanGraphStats* out_stats) {
    if(!plan||!out_stats||!plan->graph||!plan->finalized)return hipErrorInvalidValue;
    size_t count=0;
    hipError_t error=hipGraphGetNodes(plan->graph,nullptr,&count);
    if(error!=hipSuccess)return error;
    std::vector<hipGraphNode_t> nodes(count);
    if(count){
        error=hipGraphGetNodes(plan->graph,nodes.data(),&count);
        if(error!=hipSuccess)return error;
    }
    NRPlanGraphStats stats{};stats.total_nodes=count;
    for(size_t i=0;i<count;++i){
        hipGraphNodeType type{};
        error=hipGraphNodeGetType(nodes[i],&type);
        if(error!=hipSuccess)return error;
        if(type==hipGraphNodeTypeKernel)++stats.kernel_nodes;
        else if(type==hipGraphNodeTypeMemcpy)++stats.memcpy_nodes;
    }
    *out_stats=stats;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanSetStaticIO(NRPlan* plan,void* static_input,
    uint64_t input_bytes,void* static_output,uint64_t output_bytes,
    uint32_t width,uint32_t height) {
    if(!plan||!plan->finalized||!static_input||!static_output||!input_bytes||!output_bytes||
       !width||!height||width>plan->desc.max_width||height>plan->desc.max_height)
        return hipErrorInvalidValue;
    plan->static_input=static_input;plan->input_bytes=input_bytes;
    plan->static_output=static_output;plan->output_bytes=output_bytes;
    plan->static_width=width;plan->static_height=height;
    return hipSuccess;
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
    error=instantiate_graph(graph,&executable);
    if(error!=hipSuccess){hipGraphDestroy(graph);return error;}
    plan->graph=graph;plan->owns_graph=true;
    plan->executable=executable;plan->finalized=true;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanSubmit(NRPlan* plan,const NRFrameBindings* bindings,
    hipEvent_t input_ready,hipEvent_t output_done) {
    if(!plan||!bindings||!plan->finalized||!bindings->input||!bindings->output||
       !bindings->width||!bindings->height||bindings->width>plan->desc.max_width||
       bindings->height>plan->desc.max_height||
       (plan->static_input&&(bindings->width!=plan->static_width||bindings->height!=plan->static_height)))
        return hipErrorInvalidValue;
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
    if(error==hipSuccess&&plan->static_input)
        error=hipMemcpyAsync(plan->static_input,bindings->input,size_t(plan->input_bytes),
            hipMemcpyDeviceToDevice,plan->stream);
    if(error==hipSuccess)error=hipGraphLaunch(plan->executable,plan->stream);
    if(error==hipSuccess&&plan->static_output)
        error=hipMemcpyAsync(bindings->output,plan->static_output,size_t(plan->output_bytes),
            hipMemcpyDeviceToDevice,plan->stream);
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
