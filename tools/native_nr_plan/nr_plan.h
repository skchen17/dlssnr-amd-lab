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

constexpr uint32_t NR_PLAN_FRAME_BINDINGS_ABI_VERSION = 2;
constexpr uint32_t NR_PLAN_FRAME_BINDINGS_ABI_VERSION_V3 = 3;

enum NRPrecisionProfile : uint32_t {
    NR_PRECISION_STRICT_FP16 = 0,
    NR_PRECISION_APPROX_FP8 = 1,
};

enum NRColorMode : uint32_t {
    NR_COLOR_SDR_LINEAR = 0,
    NR_COLOR_HDR_LINEAR = 1,
};

enum NRResourceFormat : uint32_t {
    NR_FORMAT_UNKNOWN = 0,
    NR_FORMAT_RGBA16_FLOAT = 1,
    NR_FORMAT_RG16_FLOAT = 2,
    NR_FORMAT_R32_FLOAT = 3,
};

enum NRFrameFlags : uint32_t {
    NR_FRAME_HAS_HISTORY = 1u << 0,
    NR_FRAME_HAS_MOTION = 1u << 1,
    NR_FRAME_HAS_DEPTH = 1u << 2,
    NR_FRAME_HAS_EXPOSURE = 1u << 3,
};

struct NRPlanDesc {
    uint64_t workspace_bytes;
    uint64_t weight_bytes;
    uint32_t max_width;
    uint32_t max_height;
};

// A plan is prepared for exactly one padded network shape. A caller creates a
// second plan (or destroys/re-prepares an idle plan) for a different shape;
// frame submission never silently changes geometry or reallocates the arena.
struct NRShapeDesc {
    uint32_t struct_size;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t valid_left;
    uint32_t valid_top;
    uint32_t valid_width;
    uint32_t valid_height;
    uint32_t color_mode;
    uint32_t color_format;
    uint32_t motion_format;
    uint32_t depth_format;
};

// Kernels recorded into the graph consume this device-resident table. This
// indirection lets each frame change only input/output pointers and metadata.
struct NRFrameBindings {
    uint32_t struct_size;
    uint32_t abi_version;
    const void* input;
    void* output;
    const void* history;
    void* next_history;
    const void* motion;
    const void* depth;
    const void* exposure;
    const void* controls;
    uint32_t width;
    uint32_t height;
    uint64_t frame_id;
    uint64_t resource_generation;
    uint32_t reset;
    uint32_t flags;
};

// ABI v3 makes every temporal/color value explicit. Kernels read this table
// through a stable device pointer recorded in the graph; no graph-node patching
// or host allocation is permitted in the per-frame path.
struct NRFrameBindingsV3 {
    uint32_t struct_size;
    uint32_t abi_version;
    const void* current_color;
    void* output_residual;
    const void* history;
    void* next_history;
    const void* motion;
    const void* depth;
    const void* exposure;
    const void* controls;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t valid_left;
    uint32_t valid_top;
    uint32_t valid_width;
    uint32_t valid_height;
    float jitter_x;
    float jitter_y;
    float motion_scale_x;
    float motion_scale_y;
    float pre_exposure;
    float exposure_scale;
    uint64_t frame_id;
    uint64_t resource_generation;
    uint32_t reset;
    uint32_t flags;
};

// Imported shared D3D12 fences remain owned by the caller. NRPlan only queues
// a GPU wait before the graph and a GPU signal after it; it never CPU-waits.
struct NRExternalSyncV3 {
    uint32_t struct_size;
    uint32_t reserved;
    hipExternalSemaphore_t input_ready;
    uint64_t input_value;
    hipExternalSemaphore_t output_done;
    uint64_t output_value;
};

struct NRPlanGraphStats {
    uint64_t total_nodes;
    uint64_t kernel_nodes;
    uint64_t memcpy_nodes;
};

// Fixed-size ABI used by the census tool. `kernel_ordinal` indexes only
// kernel nodes; `graph_node_ordinal` is the position returned by
// hipGraphGetNodes and is not an execution-order promise.
struct NRPlanKernelNodeInfo {
    uint64_t graph_node_ordinal;
    uint64_t kernel_ordinal;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t dynamic_shared_bytes;
    uint32_t registers_per_thread;
    uint64_t static_shared_bytes;
    uint64_t local_bytes_per_thread;
    uint32_t max_threads_per_block;
    char name[512];
};

struct NRPlanResourceStats {
    uint64_t workspace_bytes;
    uint64_t weight_bytes;
    uint64_t arena_region_count;
    uint8_t owns_workspace;
    uint8_t owns_weights;
    uint8_t owns_graph_source;
    uint8_t owns_graph_executable;
    uint8_t graph_references_external_allocations;
    uint8_t deployment_ready;
    uint8_t temporal_contract_verified;
    uint8_t precision_profile;
    uint32_t frame_bindings_abi_version;
    uint32_t native_stage_block_count;
    uint8_t complete_native_topology;
    uint8_t reserved[3];
};

// One attention suffix in a C64/C128/C256 Swin block. All offsets are relative
// to NRPlan's C++-owned workspace or selected model-weight allocation. Q/K/V,
// value and next_resident are one-byte E4M3; post/output keep the FP16 residual
// boundary. The descriptor is copied at initialization and is never referenced
// through caller memory by a captured graph.
struct NRApproxStageBlockDesc {
    uint32_t struct_size;
    uint32_t channels;
    uint32_t windows;
    uint32_t record_number;
    uint32_t feature_width;
    uint32_t feature_height;
    int32_t origin_x;
    int32_t origin_y;
    uint64_t raw_resident_offset;
    uint64_t grouped_offset;
    uint64_t seed_fp16_offset;
    uint64_t post_fp16_offset;
    uint64_t post_resident_offset;
    uint64_t qkv_projection_fp16_offset;
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t value_offset;
    uint64_t output_fp16_offset;
    uint64_t next_resident_offset;
    uint64_t ffn_expand_weight_offset;
    uint64_t ffn_contract_weight_offset;
    uint64_t ffn_mix_weight_offset;
    uint64_t ffn_scale_weight_offset;
    uint64_t a_index_weight_offset;
    uint64_t residual_index_weight_offset;
    uint64_t ffn_permutation_weight_offset;
    uint64_t ffn_inverse_permutation_weight_offset;
    uint64_t qkv_weight_offset;
    uint64_t qscale_weight_offset;
    uint64_t permutation_weight_offset;
    uint64_t position_bias_weight_offset;
    uint64_t project_weight_offset;
    uint64_t residual_scale_weight_offset;
};

// C512 has four source records and a different FFN split. It shares the same
// resident Q/K/V ABI but keeps a dedicated descriptor so a standard wide block
// can never be accidentally bound to split-block weights.
struct NRApproxSplit512BlockDesc {
    uint32_t struct_size;
    uint32_t windows;
    uint32_t feature_width;
    uint32_t feature_height;
    int32_t origin_x;
    int32_t origin_y;
    uint32_t record_number;
    uint32_t reserved;
    uint64_t raw_resident_offset;
    uint64_t projected_offset;
    uint64_t grouped_offset;
    uint64_t post_fp16_offset;
    uint64_t post_resident_offset;
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t value_offset;
    uint64_t output_fp16_offset;
    uint64_t next_resident_offset;
    uint64_t preproject_weight_offset;
    uint64_t expand_weight_offset;
    uint64_t contract_weight_offset;
    uint64_t ffn_project_weight_offset;
    uint64_t ffn_scale_weight_offset;
    uint64_t a_index_weight_offset;
    uint64_t residual_index_weight_offset;
    uint64_t perm64_weight_offset;
    uint64_t perm256_weight_offset;
    uint64_t ffn_permutation_weight_offset;
    uint64_t qkv_weight_offset;
    uint64_t qscale_weight_offset;
    uint64_t attention_permutation_weight_offset;
    uint64_t position_bias_weight_offset;
    uint64_t attention_project_weight_offset;
    uint64_t attention_scale_weight_offset;
};

// The 1024-channel bottleneck is global attention, not a Swin window block.
// Its five-stage recorder streams 16-key tiles and keeps only resident E4M3
// activations between launches. Eight descriptors alternate vit.ping/pong.
struct NRApproxVitBlockDesc {
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t record_number;
    uint32_t reserved;
    uint64_t input_offset;
    uint64_t hidden_offset;
    uint64_t post_offset;
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t value_offset;
    uint64_t next_offset;
    uint64_t expand_weight_offset;
    uint64_t contract_weight_offset;
    uint64_t qkv_weight_offset;
    uint64_t q_scale_weight_offset;
    uint64_t project_weight_offset;
    uint64_t ffn_scale_weight_offset;
    uint64_t attention_scale_weight_offset;
    uint64_t perm1024_weight_offset;
    uint64_t perm4096_weight_offset;
};

struct NRApproxBottleneckDesc {
    uint32_t struct_size;
    uint32_t tokens;
    uint32_t feature_width;
    uint32_t feature_height;
    uint32_t low_width;
    uint32_t low_height;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t c512_input_offset;
    uint64_t c512_skip_offset;
    uint64_t vit_input_offset;
    uint64_t vit_output_offset;
    uint64_t c512_output_offset;
    uint64_t encoder_weight_offset;
    uint64_t encoder_permutation_offset;
    uint64_t decoder_weight_offset;
    uint64_t decoder_scale_offset;
    uint64_t decoder_permutation_offset;
};

enum NRApproxTransitionDirection : uint32_t {
    NR_TRANSITION_ENCODER_DOWNSAMPLE = 1,
    NR_TRANSITION_DECODER_UPSAMPLE = 2,
};

// A fixed resident-layout scale boundary. Encoder descriptors are recorded
// immediately after anchor_record and decoder descriptors immediately before
// it. Resident activations/skips are E4M3 bytes; FP16 rounding is retained
// inside the kernels at pooling and residual-add semantic boundaries.
struct NRApproxScaleTransitionDesc {
    uint32_t struct_size;
    uint32_t direction;
    uint32_t anchor_record;
    uint32_t channels;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t target_width;
    uint32_t target_height;
    int32_t source_origin_x;
    int32_t source_origin_y;
    uint64_t source_offset;
    uint64_t source_resident_offset;
    uint64_t skip_offset;
    uint64_t target_offset;
    uint64_t project_weight_offset;
    uint64_t permutation_weight_offset;
    uint64_t skip_scale_weight_offset;
};

struct NRPlanPerformanceStats {
    uint64_t submitted_frames;
    uint64_t completed_frames;
    double last_gpu_ms;
    double total_gpu_ms;
    double min_gpu_ms;
    double max_gpu_ms;
};

struct NRModelPackageStats {
    uint32_t package_version;
    uint32_t precision_profile;
    uint64_t selected_weight_bytes;
    uint32_t selected_weight_segments;
    uint32_t temporal_contract_verified;
    char architecture[32];
    char temporal_contract_id[64];
    char target_arch[16];
    char original_model_sha256[65];
    char selected_weights_sha256[65];
};

struct NRPlanArenaRegion {
    uint64_t offset;
    uint64_t bytes;
    uint32_t alignment;
    uint32_t kind;
    char name[64];
};

// The callback is called exactly once during Finalize, while stream capture is
// active. It must only enqueue HIP work; it must not allocate or synchronize.
using NRPlanRecordFn = hipError_t (*)(hipStream_t stream, void* workspace,
    const void* weights, const NRFrameBindings* device_bindings, void* user);
using NRPlanRecordFnV3 = hipError_t (*)(hipStream_t stream, void* workspace,
    const void* weights, const NRFrameBindingsV3* device_bindings, void* user);

NRPLAN_API hipError_t nrPlanCreate(const NRPlanDesc* desc, NRPlan** out_plan);
NRPLAN_API hipError_t nrPlanUploadWeights(NRPlan* plan, const void* host_weights,
    uint64_t bytes);
// Loads the deterministic private model-package-v2 container. Create the plan
// with weight_bytes=0; the selected strict/FP8 section determines allocation.
NRPLAN_API hipError_t nrPlanLoadModelPackage(NRPlan* plan,
    const char* utf8_path, NRModelPackageStats* out_stats);
NRPLAN_API hipError_t nrPlanSetRecorder(NRPlan* plan, NRPlanRecordFn record,
    void* user);
NRPLAN_API hipError_t nrPlanSetRecorderV3(NRPlan* plan, NRPlanRecordFnV3 record,
    void* user);
NRPLAN_API hipError_t nrPlanSetPrecisionProfile(NRPlan* plan,
    NRPrecisionProfile profile);
NRPLAN_API hipError_t nrPlanPrepareShape(NRPlan* plan,
    const NRShapeDesc* shape);
NRPLAN_API hipError_t nrPlanGetStream(NRPlan* plan, hipStream_t* out_stream);
// Instantiate a graph captured by an allocator-aware owner (for the
// transitional PyTorch oracle bridge). NRPlan owns the executable, while the
// capture owner keeps the source graph and its allocation pool alive until
// NRPlan is destroyed. This avoids the ROCm/Windows large-graph clone overflow.
NRPLAN_API hipError_t nrPlanAdoptGraph(NRPlan* plan, hipGraph_t source_graph);
NRPLAN_API hipError_t nrPlanGetGraphStats(NRPlan* plan,
    NRPlanGraphStats* out_stats);
NRPLAN_API hipError_t nrPlanGetKernelNodeInfo(NRPlan* plan,
    uint64_t kernel_ordinal, NRPlanKernelNodeInfo* out_info);
NRPLAN_API hipError_t nrPlanGetKernelNodeInfos(NRPlan* plan,
    NRPlanKernelNodeInfo* out_infos, uint64_t capacity, uint64_t* out_count);
NRPLAN_API hipError_t nrPlanDebugDotPrint(NRPlan* plan, const char* path);
// Correctness-bring-up helpers for kernels whose leading arguments are all
// pointers/uint64 values. They inspect captured host graph parameters only and
// never enqueue device work.
NRPLAN_API hipError_t nrPlanDebugGetKernelU64Arguments(NRPlan* plan,
    uint64_t kernel_ordinal, uint32_t argument_count, uint64_t* out_arguments);
NRPLAN_API hipError_t nrPlanDebugGetOwnedAddresses(NRPlan* plan,
    uint64_t* out_workspace, uint64_t* out_weights, uint64_t* out_stage_e4_lut);
// Census-only marker. It is never present in the deploy/performance graph.
NRPLAN_API hipError_t nrPlanRecordStageMarker(hipStream_t stream,
    uint32_t stage_boundary, uint32_t* device_scratch);
NRPLAN_API hipError_t nrPlanGetResourceStats(NRPlan* plan,
    NRPlanResourceStats* out_stats);
NRPLAN_API hipError_t nrPlanConfigureArena(NRPlan* plan,
    const NRPlanArenaRegion* regions, uint64_t count);
// Selects the C++-owned approximate-stage recorder. It may be used for partial
// stage bring-up, but deployment_ready remains false until the complete 71-block
// topology is native. This API never adopts a PyTorch graph or allocation.
NRPLAN_API hipError_t nrPlanConfigureApproxStageBlocks(NRPlan* plan,
    const NRApproxStageBlockDesc* blocks, uint32_t count,
    uint8_t complete_native_topology);
NRPLAN_API hipError_t nrPlanConfigureApproxSplit512Blocks(NRPlan* plan,
    const NRApproxSplit512BlockDesc* blocks, uint32_t count);
NRPLAN_API hipError_t nrPlanConfigureApproxVitBlocks(NRPlan* plan,
    const NRApproxVitBlockDesc* blocks, uint32_t count);
NRPLAN_API hipError_t nrPlanConfigureApproxBottleneck(NRPlan* plan,
    const NRApproxBottleneckDesc* descriptor);
NRPLAN_API hipError_t nrPlanConfigureApproxScaleTransitions(NRPlan* plan,
    const NRApproxScaleTransitionDesc* descriptors, uint32_t count);
// Isolated correctness-gate helpers. They are synchronous by design and must
// never appear in the per-frame game path or performance measurements.
NRPLAN_API hipError_t nrPlanInitializeArenaFromDevice(NRPlan* plan,
    uint64_t offset, const void* device_source, uint64_t bytes);
NRPLAN_API hipError_t nrPlanDebugCopyArenaToDevice(NRPlan* plan,
    uint64_t offset, void* device_target, uint64_t bytes);
NRPLAN_API hipError_t nrPlanGetArenaRegion(NRPlan* plan,
    uint64_t index, NRPlanArenaRegion* out_region);
NRPLAN_API hipError_t nrPlanSetStaticIO(NRPlan* plan, void* static_input,
    uint64_t input_bytes, void* static_output, uint64_t output_bytes,
    uint32_t width, uint32_t height);
NRPLAN_API hipError_t nrPlanFinalize(NRPlan* plan);
NRPLAN_API hipError_t nrPlanSubmit(NRPlan* plan, const NRFrameBindings* bindings,
    hipEvent_t input_ready, hipEvent_t output_done);
NRPLAN_API hipError_t nrPlanSubmitV3(NRPlan* plan,
    const NRFrameBindingsV3* bindings, hipEvent_t input_ready,
    hipEvent_t output_done);
NRPLAN_API hipError_t nrPlanSubmitExternalV3(NRPlan* plan,
    const NRFrameBindingsV3* bindings, const NRExternalSyncV3* sync);
NRPLAN_API hipError_t nrPlanGetPerformanceStats(NRPlan* plan,
    NRPlanPerformanceStats* out_stats);
NRPLAN_API hipError_t nrPlanReset(NRPlan* plan);
NRPLAN_API hipError_t nrPlanDestroy(NRPlan* plan);
