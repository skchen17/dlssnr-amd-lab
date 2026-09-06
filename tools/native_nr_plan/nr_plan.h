#pragma once
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

#ifdef _WIN32
#define NRPLAN_API extern "C" __declspec(dllexport)
#else
#define NRPLAN_API extern "C" __attribute__((visibility("default")))
#endif

struct NRPlan;

struct NRPlanDesc {
    uint64_t workspace_bytes;
    uint64_t weight_bytes;
    uint32_t max_width;
    uint32_t max_height;
};

// Kernels recorded into the graph consume this device-resident table. This
// indirection lets each frame change only input/output pointers and metadata.
struct NRFrameBindings {
    const void* input;
    void* output;
    uint32_t width;
    uint32_t height;
    uint64_t frame_id;
    uint64_t resource_generation;
};

struct NRPlanGraphStats {
    uint64_t total_nodes;
    uint64_t kernel_nodes;
    uint64_t memcpy_nodes;
};

// The callback is called exactly once during Finalize, while stream capture is
// active. It must only enqueue HIP work; it must not allocate or synchronize.
using NRPlanRecordFn = hipError_t (*)(hipStream_t stream, void* workspace,
    const void* weights, const NRFrameBindings* device_bindings, void* user);

NRPLAN_API hipError_t nrPlanCreate(const NRPlanDesc* desc, NRPlan** out_plan);
NRPLAN_API hipError_t nrPlanUploadWeights(NRPlan* plan, const void* host_weights,
    uint64_t bytes);
NRPLAN_API hipError_t nrPlanSetRecorder(NRPlan* plan, NRPlanRecordFn record,
    void* user);
NRPLAN_API hipError_t nrPlanGetStream(NRPlan* plan, hipStream_t* out_stream);
// Instantiate a graph captured by an allocator-aware owner (for the
// transitional PyTorch oracle bridge). NRPlan owns the executable, while the
// capture owner keeps the source graph and its allocation pool alive until
// NRPlan is destroyed. This avoids the ROCm/Windows large-graph clone overflow.
NRPLAN_API hipError_t nrPlanAdoptGraph(NRPlan* plan, hipGraph_t source_graph);
NRPLAN_API hipError_t nrPlanGetGraphStats(NRPlan* plan,
    NRPlanGraphStats* out_stats);
NRPLAN_API hipError_t nrPlanSetStaticIO(NRPlan* plan, void* static_input,
    uint64_t input_bytes, void* static_output, uint64_t output_bytes,
    uint32_t width, uint32_t height);
NRPLAN_API hipError_t nrPlanFinalize(NRPlan* plan);
NRPLAN_API hipError_t nrPlanSubmit(NRPlan* plan, const NRFrameBindings* bindings,
    hipEvent_t input_ready, hipEvent_t output_done);
NRPLAN_API hipError_t nrPlanReset(NRPlan* plan);
NRPLAN_API hipError_t nrPlanDestroy(NRPlan* plan);
