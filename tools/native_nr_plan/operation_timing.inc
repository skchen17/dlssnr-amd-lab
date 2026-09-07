// Diagnostic only: enabled explicitly before capture. No counter collection,
// no kernel changes, no per-node synchronization. Only serial graphs qualify.
static bool operation_timing_enabled() {
    const char* value=std::getenv("NRPLAN_OPERATION_TIMING");
    return value&&std::strcmp(value,"1")==0;
}

static std::vector<hipGraphNode_t> capture_nodes(NRPlan* plan) {
    hipStreamCaptureStatus status{};unsigned long long id{};hipGraph_t graph{};
    const hipGraphNode_t* dependencies{};size_t count{};
    if(hipStreamGetCaptureInfo_v2(plan->stream,&status,&id,&graph,&dependencies,
        &count)!=hipSuccess||!graph)return {};
    count=0;if(hipGraphGetNodes(graph,nullptr,&count)!=hipSuccess)return {};
    std::vector<hipGraphNode_t> nodes(count);
    if(count&&hipGraphGetNodes(graph,nodes.data(),&count)!=hipSuccess)return {};
    return nodes;
}

struct DiagnosticScope {
    NRPlan* plan;std::string label;std::set<hipGraphNode_t> before;
    bool enabled;
    DiagnosticScope(NRPlan* p,std::string name):plan(p),label(std::move(name)),
        enabled(p->recording_graph&&operation_timing_enabled()) {
        if(enabled){auto nodes=capture_nodes(plan);before.insert(nodes.begin(),nodes.end());}
    }
    ~DiagnosticScope() {
        if(enabled)for(auto node:capture_nodes(plan))
            if(!before.count(node))plan->diagnostic_labels[node]=label;
    }
};

static hipError_t instrument_operation_timing(NRPlan* plan,hipGraph_t graph) {
    if(!operation_timing_enabled())return hipSuccess;
    size_t count=0;hipError_t error=hipGraphGetNodes(graph,nullptr,&count);
    if(error!=hipSuccess||!count)return hipErrorInvalidValue;
    std::vector<hipGraphNode_t> nodes(count);
    error=hipGraphGetNodes(graph,nodes.data(),&count);if(error!=hipSuccess)return error;
    // Validate the existing graph is one chain. Never serialize a parallel graph
    // and misrepresent that result as the production topology.
    std::map<hipGraphNode_t,hipGraphNode_t> next;hipGraphNode_t root{};
    for(auto node:nodes){
        size_t n=0;error=hipGraphNodeGetDependencies(node,nullptr,&n);
        if(error!=hipSuccess)return error;
        if(n>1)return hipErrorInvalidValue;
        if(!n){if(root)return hipErrorInvalidValue;root=node;}
        else {hipGraphNode_t parent{};error=hipGraphNodeGetDependencies(node,&parent,&n);
            if(error!=hipSuccess)return error;if(next.count(parent))return hipErrorInvalidValue;
            next[parent]=node;}
    }
    for(auto node=root;node;node=next.count(node)?next[node]:nullptr)
        plan->diagnostic_nodes.push_back(node);
    if(plan->diagnostic_nodes.size()!=count)return hipErrorInvalidValue;
    const char* dot_path=std::getenv("NRPLAN_OPERATION_DOT");
    if(!dot_path)return hipErrorInvalidValue;
    error=hipGraphDebugDotPrint(graph,dot_path,0);if(error!=hipSuccess)return error;
    plan->diagnostic_events.resize(count+1);
    for(auto& event:plan->diagnostic_events){error=hipEventCreate(&event);if(error!=hipSuccess)return error;}
    for(size_t i=0;i<count;++i){
        auto node=plan->diagnostic_nodes[i];hipGraphNode_t event_node{};
        hipGraphNode_t previous=i?plan->diagnostic_nodes[i-1]:nullptr;
        if(i){error=hipGraphRemoveDependencies(graph,&previous,&node,1);if(error!=hipSuccess)return error;}
        error=hipGraphAddEventRecordNode(&event_node,graph,i?&previous:nullptr,i?1:0,
            plan->diagnostic_events[i]);if(error!=hipSuccess)return error;
        error=hipGraphAddDependencies(graph,&event_node,&node,1);if(error!=hipSuccess)return error;
    }
    hipGraphNode_t stop{};auto last=plan->diagnostic_nodes.back();
    return hipGraphAddEventRecordNode(&stop,graph,&last,1,plan->diagnostic_events.back());
}

static void timing_json_string(std::ostream& out,const std::string& value) {
    out<<'"';for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<c;
        else if(c<32)out<<' ';else out<<c;}out<<'"';
}

NRPLAN_API hipError_t nrPlanDebugWriteOperationTimings(NRPlan* plan,const char* path) {
    if(!plan||!path||plan->diagnostic_nodes.empty())return hipErrorInvalidValue;
    hipError_t error=hipEventQuery(plan->diagnostic_events.back());if(error!=hipSuccess)return error;
    std::ofstream out(std::filesystem::u8path(path));if(!out)return hipErrorInvalidValue;
    out<<std::setprecision(10)<<"{\"measurement\":\"instrumented HIP event intervals; not hardware busy counters\",\"nodes\":[";
    for(size_t i=0;i<plan->diagnostic_nodes.size();++i){
        auto node=plan->diagnostic_nodes[i];float ms{};
        error=hipEventElapsedTime(&ms,plan->diagnostic_events[i],plan->diagnostic_events[i+1]);
        if(error!=hipSuccess)return error;
        hipGraphNodeType type{};error=hipGraphNodeGetType(node,&type);if(error!=hipSuccess)return error;
        if(i)out<<',';out<<"{\"index\":"<<i<<",\"scope\":";
        timing_json_string(out,plan->diagnostic_labels.count(node)?plan->diagnostic_labels[node]:"Unassigned");
        out<<",\"event_ms\":"<<ms<<",\"kind\":";
        timing_json_string(out,type==hipGraphNodeTypeKernel?"kernel":type==hipGraphNodeTypeMemcpy?"memcpy":"other");
        if(type==hipGraphNodeTypeKernel){
            hipKernelNodeParams params{};error=hipGraphKernelNodeGetParams(node,&params);if(error!=hipSuccess)return error;
            // Windows graph params.func is not a hipKernel_t. Querying it as
            // one sets a sticky invalid-handle error in the host thread. Names
            // come from the original DOT's dependency order instead.
            out<<",\"name\":\"resolved_offline_from_original_graph_dot\"";
            out<<",\"grid\":["<<params.gridDim.x<<','<<params.gridDim.y<<','<<params.gridDim.z
               <<"],\"block\":["<<params.blockDim.x<<','<<params.blockDim.y<<','<<params.blockDim.z
               <<"],\"dynamic_lds_bytes\":"<<params.sharedMemBytes;
            hipFuncAttributes attr{};
            if(hipFuncGetAttributes(&attr,params.func)==hipSuccess)
                out<<",\"registers_per_thread\":"<<attr.numRegs
                   <<",\"static_lds_bytes\":"<<attr.sharedSizeBytes
                   <<",\"local_bytes_per_thread\":"<<attr.localSizeBytes;
        }out<<'}';
    }out<<"]}";return out?hipSuccess:hipErrorInvalidValue;
}
