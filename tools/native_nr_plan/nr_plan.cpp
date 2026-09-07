#include "nr_plan.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

extern "C" int nr_stage_initialize_e4_lut(void**);
extern "C" int nr_stage_c64_qkv_norm_fp8(const void*,const void*,const void*,
    const void*,void*,void*,void*,void*,const void*,int,void*);
extern "C" int nr_stage_c64_attention_fp8(const void*,const void*,const void*,
    const void*,void*,int,void*);
extern "C" int nr_stage_c64_project_fp8(const void*,const void*,const void*,
    const void*,const void*,void*,void*,int,void*);
extern "C" int nr_stage_c64_ffn_fp8(const void*,const void*,const void*,const void*,
    const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,
    int,int,int,int,int,void*);
extern "C" int nr_stage_c64_scatter_fp8(const void*,void*,int,int,int,int,void*);
extern "C" int nr_stage_c32_qkv_norm_fp8(const void*,const void*,const void*,
    const void*,void*,void*,void*,void*,const void*,int,void*);
extern "C" int nr_stage_c32_attention_fp8(const void*,const void*,const void*,
    const void*,void*,int,void*);
extern "C" int nr_stage_c32_project_fp8(const void*,const void*,const void*,
    const void*,const void*,void*,void*,int,void*);
extern "C" int nr_stage_c32_ffn_fp8(const void*,const void*,const void*,const void*,
    const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,
    int,int,int,int,int,void*);
extern "C" int nr_stage_c32_scatter_fp8(const void*,void*,int,int,int,int,void*);
extern "C" int nr_stage_c128_qkv_norm_fp8(const void*,const void*,const void*,
    const void*,void*,void*,void*,void*,const void*,int,void*);
extern "C" int nr_stage_c128_attention_fp8(const void*,const void*,const void*,
    const void*,void*,int,void*);
extern "C" int nr_stage_c128_project_fp8(const void*,const void*,const void*,
    const void*,const void*,void*,void*,int,void*);
extern "C" int nr_stage_c128_ffn_fp8(const void*,const void*,const void*,const void*,
    const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,
    int,int,int,int,int,void*);
extern "C" int nr_stage_c128_scatter_fp8(const void*,void*,int,int,int,int,void*);
extern "C" int nr_stage_c256_qkv_norm_fp8(const void*,const void*,const void*,
    const void*,void*,void*,void*,void*,const void*,int,void*);
extern "C" int nr_stage_c256_attention_fp8(const void*,const void*,const void*,
    const void*,void*,int,void*);
extern "C" int nr_stage_c256_project_fp8(const void*,const void*,const void*,
    const void*,const void*,void*,void*,int,void*);
extern "C" int nr_stage_c256_ffn_fp8(const void*,const void*,const void*,const void*,
    const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,
    int,int,int,int,int,void*);
extern "C" int nr_stage_c256_scatter_fp8(const void*,void*,int,int,int,int,void*);
extern "C" int nr_stage_c512_ffn_fp8(const void*,const void*,const void*,const void*,
    const void*,const void*,const void*,const void*,const void*,const void*,const void*,
    void*,void*,void*,void*,int,int,int,int,int,void*);
extern "C" int nr_stage_c512_qkv_norm_fp8(const void*,const void*,const void*,
    const void*,void*,void*,void*,int,void*);
extern "C" int nr_stage_c512_attention_fp8(const void*,const void*,const void*,
    const void*,void*,int,void*);
extern "C" int nr_stage_c512_project_fp8(const void*,const void*,const void*,
    const void*,const void*,void*,void*,int,void*);
extern "C" int nr_stage_c512_scatter_fp8(const void*,void*,int,int,int,int,void*);
extern "C" int nr_vit_block_fp8(const void*,const void*,const void*,const void*,
    const void*,const void*,const void*,const void*,const void*,const void*,void*,
    void*,void*,void*,void*,void*,void*,int,void*);
extern "C" int nr_encoder_final_to_vit_fp8(const void*,void*,const void*,
    const void*,void*,int,int,int,int,void*);
extern "C" int nr_decoder_input_from_vit_fp8(const void*,const void*,const void*,
    const void*,const void*,void*,int,int,int,int,void*);
extern "C" int nr_transition_encoder_fp8(const void*,const void*,void*,const void*,
    const void*,void*,int,int,int,int,int,void*);
extern "C" int nr_transition_decoder_fp8(const void*,const void*,const void*,const void*,
    const void*,void*,int,int,int,void*);

struct NRPlan {
    NRPlanDesc desc{};
    hipStream_t stream{};
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    void* workspace{};
    void* weights{};
    NRFrameBindings* device_bindings{};
    NRFrameBindings* pinned_bindings{};
    NRFrameBindingsV3* device_bindings_v3{};
    NRFrameBindingsV3* pinned_bindings_v3{};
    hipEvent_t binding_consumed{};
    hipEvent_t timing_start{};
    hipEvent_t timing_stop{};
    NRPlanRecordFn record{};
    NRPlanRecordFnV3 record_v3{};
    void* record_user{};
    void* static_input{};
    void* static_output{};
    uint64_t input_bytes{};
    uint64_t output_bytes{};
    void* stage_e4_lut{};
    bool stage_e4_lut_initialized{};
    uint32_t static_width{};
    uint32_t static_height{};
    bool owns_graph{};
    bool external_graph_allocations{};
    bool finalized{};
    bool submitted{};
    bool prepared{};
    bool timing_pending{};
    NRPrecisionProfile precision{NR_PRECISION_STRICT_FP16};
    NRShapeDesc shape{};
    NRPlanPerformanceStats performance{};
    bool weights_uploaded{};
    bool temporal_contract_verified{};
    std::vector<NRPlanArenaRegion> arena_regions;
    std::vector<NRApproxStageBlockDesc> approximate_stage_blocks;
    std::vector<NRApproxSplit512BlockDesc> approximate_split512_blocks;
    std::vector<NRApproxVitBlockDesc> approximate_vit_blocks;
    NRApproxBottleneckDesc approximate_bottleneck{};
    bool has_approximate_bottleneck{};
    std::vector<NRApproxScaleTransitionDesc> approximate_scale_transitions;
    bool complete_native_topology{};
};

static hipError_t ensure_stage_e4_lut(NRPlan* plan) {
    if (plan->stage_e4_lut_initialized) return hipSuccess;
    hipError_t error = static_cast<hipError_t>(nr_stage_initialize_e4_lut(
        &plan->stage_e4_lut));
    if (error == hipSuccess) plan->stage_e4_lut_initialized = true;
    return error;
}

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
    if(plan->timing_start)keep(hipEventDestroy(plan->timing_start));
    if(plan->timing_stop)keep(hipEventDestroy(plan->timing_stop));
    if(plan->device_bindings)keep(hipFree(plan->device_bindings));
    if(plan->pinned_bindings)keep(hipHostFree(plan->pinned_bindings));
    if(plan->device_bindings_v3)keep(hipFree(plan->device_bindings_v3));
    if(plan->pinned_bindings_v3)keep(hipHostFree(plan->pinned_bindings_v3));
    if(plan->stage_e4_lut)keep(hipFree(plan->stage_e4_lut));
    if(plan->weights)keep(hipFree(plan->weights));
    if(plan->workspace)keep(hipFree(plan->workspace));
    if(plan->stream)keep(hipStreamDestroy(plan->stream));
    delete plan;
    return first;
}

NRPLAN_API hipError_t nrPlanCreate(const NRPlanDesc* desc, NRPlan** out_plan) {
    if(!desc||!out_plan||!desc->workspace_bytes||
       !desc->max_width||!desc->max_height)return hipErrorInvalidValue;
    *out_plan=nullptr;
    auto* plan=new(std::nothrow) NRPlan;
    if(!plan)return hipErrorOutOfMemory;
    plan->desc=*desc;
    hipError_t error=hipStreamCreateWithFlags(&plan->stream,hipStreamNonBlocking);
    if(error==hipSuccess)error=hipMalloc(&plan->workspace,size_t(desc->workspace_bytes));
    if(error==hipSuccess&&desc->weight_bytes)
        error=hipMalloc(&plan->weights,size_t(desc->weight_bytes));
    if(error==hipSuccess)error=hipMalloc(&plan->device_bindings,sizeof(NRFrameBindings));
    if(error==hipSuccess)error=hipHostMalloc(&plan->pinned_bindings,sizeof(NRFrameBindings));
    if(error==hipSuccess)error=hipMalloc(&plan->device_bindings_v3,sizeof(NRFrameBindingsV3));
    if(error==hipSuccess)error=hipHostMalloc(&plan->pinned_bindings_v3,sizeof(NRFrameBindingsV3));
    if(error==hipSuccess)error=hipEventCreateWithFlags(&plan->binding_consumed,hipEventDisableTiming);
    if(error==hipSuccess)error=hipEventCreate(&plan->timing_start);
    if(error==hipSuccess)error=hipEventCreate(&plan->timing_stop);
    if(error!=hipSuccess){cleanup(plan);return error;}
    *out_plan=plan;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanUploadWeights(NRPlan* plan,const void* host_weights,uint64_t bytes) {
    if(!plan||!host_weights||!plan->weights||bytes!=plan->desc.weight_bytes||
       plan->finalized||plan->weights_uploaded)return hipErrorInvalidValue;
    hipError_t error=hipMemcpyAsync(plan->weights,host_weights,size_t(bytes),hipMemcpyHostToDevice,plan->stream);
    if(error==hipSuccess)error=hipStreamSynchronize(plan->stream);
    if(error==hipSuccess)plan->weights_uploaded=true;
    return error;
}

namespace {
constexpr size_t kPackageHeaderBytes=512;
constexpr char kPackageMagic[8]={'N','R','M','P','K','G','2','\0'};
constexpr char kWeightIndexMagic[8]={'N','R','W','I','D','X','2','\0'};
constexpr size_t kWeightIndexHeaderBytes=16;
constexpr size_t kWeightIndexRecordBytes=128;
constexpr uint8_t kOriginalModelSha[32]={
    0xA5,0x51,0x3B,0x18,0x45,0xC9,0x8A,0x48,0x69,0x85,0xED,0x04,0xF3,0x8E,0x66,0xA1,
    0x85,0x4C,0xCE,0x33,0xC2,0xAB,0xA3,0xA5,0x05,0x86,0x60,0x28,0xBD,0x4E,0xE3,0xE5};

template<class T>T package_value(const std::vector<uint8_t>& raw,size_t offset) {
    T value{};std::memcpy(&value,raw.data()+offset,sizeof(value));return value;
}

bool package_text(const std::vector<uint8_t>& raw,size_t offset,size_t bytes,
    char* output,size_t capacity) {
    size_t length=0;while(length<bytes&&raw[offset+length])++length;
    if(!length||length==bytes||length>=capacity)return false;
    std::memcpy(output,raw.data()+offset,length);output[length]=0;return true;
}

void hex_sha(const uint8_t* value,char* output) {
    static constexpr char hex[]="0123456789ABCDEF";
    for(size_t i=0;i<32;++i){output[2*i]=hex[value[i]>>4];output[2*i+1]=hex[value[i]&15];}
    output[64]=0;
}

bool sha256(const uint8_t* data,size_t bytes,std::array<uint8_t,32>* output) {
#ifdef _WIN32
    if(!output||bytes>(std::numeric_limits<ULONG>::max)())return false;
    BCRYPT_ALG_HANDLE algorithm{};BCRYPT_HASH_HANDLE hash{};
    DWORD object_bytes=0,result_bytes=0;std::vector<uint8_t> object;
    NTSTATUS status=BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0);
    if(status>=0)status=BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,
        reinterpret_cast<PUCHAR>(&object_bytes),sizeof(object_bytes),&result_bytes,0);
    if(status>=0){object.resize(object_bytes);status=BCryptCreateHash(algorithm,&hash,
        object.data(),object_bytes,nullptr,0,0);}
    if(status>=0)status=BCryptHashData(hash,const_cast<PUCHAR>(data),ULONG(bytes),0);
    if(status>=0)status=BCryptFinishHash(hash,output->data(),ULONG(output->size()),0);
    if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);
    return status>=0;
#else
    (void)data;(void)bytes;(void)output;return false;
#endif
}
}

NRPLAN_API hipError_t nrPlanLoadModelPackage(NRPlan* plan,const char* path,
    NRModelPackageStats* out_stats) {
    if(!plan||!path||!path[0]||!out_stats||plan->finalized||plan->weights||
       plan->desc.weight_bytes||plan->weights_uploaded)return hipErrorInvalidValue;
    std::ifstream input(std::filesystem::u8path(path),std::ios::binary|std::ios::ate);
    if(!input)return hipErrorInvalidValue;
    auto end=input.tellg();
    if(end<std::streamoff(kPackageHeaderBytes)||end>std::streamoff(1ull<<31))return hipErrorInvalidValue;
    std::vector<uint8_t> raw;
    raw.resize(static_cast<size_t>(end));
    input.seekg(0);
    if(!input.read(reinterpret_cast<char*>(raw.data()),
        static_cast<std::streamsize>(raw.size())))return hipErrorInvalidValue;
    if(std::memcmp(raw.data(),kPackageMagic,8)||package_value<uint32_t>(raw,8)!=2||
       package_value<uint32_t>(raw,12)!=kPackageHeaderBytes||
       std::memcmp(raw.data()+56,kOriginalModelSha,32))return hipErrorInvalidValue;
    const uint32_t flags=package_value<uint32_t>(raw,16);
    const uint32_t section_count=package_value<uint32_t>(raw,20);
    const bool approximate=plan->precision==NR_PRECISION_APPROX_FP8;
    if(!(flags&1)||(flags&~3)||(approximate&&!(flags&2))||
       section_count!=((flags&2)?2u:1u))return hipErrorInvalidValue;
    const size_t strict_offset=size_t(package_value<uint64_t>(raw,24));
    const size_t strict_bytes=size_t(package_value<uint64_t>(raw,32));
    const size_t fp8_offset=size_t(package_value<uint64_t>(raw,40));
    const size_t fp8_bytes=size_t(package_value<uint64_t>(raw,48));
    const size_t index_offset=size_t(package_value<uint64_t>(raw,264));
    const size_t index_bytes=size_t(package_value<uint64_t>(raw,272));
    if(strict_offset!=kPackageHeaderBytes||!strict_bytes||strict_offset>raw.size()||
       strict_bytes>raw.size()-strict_offset||index_offset%256||
       index_offset<strict_offset+strict_bytes||index_offset>raw.size()||
       !index_bytes||index_bytes>raw.size()-index_offset||
       index_offset+index_bytes!=raw.size())return hipErrorInvalidValue;
    if(flags&2){
        const size_t expected=(strict_offset+strict_bytes+255)&~size_t(255);
        if(fp8_offset!=expected||!fp8_bytes||fp8_offset>index_offset||
           fp8_bytes>index_offset-fp8_offset)return hipErrorInvalidValue;
    }else if(fp8_offset||fp8_bytes)return hipErrorInvalidValue;
    std::array<uint8_t,32> index_digest{};
    if(!sha256(raw.data()+index_offset,index_bytes,&index_digest)||
       std::memcmp(index_digest.data(),raw.data()+280,32)||
       index_bytes<kWeightIndexHeaderBytes||
       std::memcmp(raw.data()+index_offset,kWeightIndexMagic,8))return hipErrorInvalidValue;
    const uint32_t record_count=package_value<uint32_t>(raw,index_offset+8);
    const uint32_t record_bytes=package_value<uint32_t>(raw,index_offset+12);
    if(record_bytes!=kWeightIndexRecordBytes||
       record_count>(index_bytes-kWeightIndexHeaderBytes)/record_bytes||
       kWeightIndexHeaderBytes+size_t(record_count)*record_bytes!=index_bytes)
        return hipErrorInvalidValue;
    std::array<size_t,2> cursors{};std::array<uint32_t,2> segment_counts{};
    std::array<std::set<std::string>,2> names;
    for(uint32_t number=0;number<record_count;++number){
        const size_t first=index_offset+kWeightIndexHeaderBytes+size_t(number)*record_bytes;
        const uint32_t profile=package_value<uint32_t>(raw,first);
        const uint32_t reserved=package_value<uint32_t>(raw,first+4);
        const size_t segment_offset=size_t(package_value<uint64_t>(raw,first+8));
        const size_t segment_bytes=size_t(package_value<uint64_t>(raw,first+16));
        const size_t section_bytes=profile==0?strict_bytes:fp8_bytes;
        const size_t section_offset=profile==0?strict_offset:fp8_offset;
        char name[64]{};
        if(profile>1||reserved||(profile==1&&!(flags&2))||!segment_bytes||
           segment_offset!=cursors[profile]||segment_offset>section_bytes||
           segment_bytes>section_bytes-segment_offset||
           !package_text(raw,first+56,64,name,sizeof(name)))return hipErrorInvalidValue;
        for(size_t i=first+120;i<first+128;++i)if(raw[i])return hipErrorInvalidValue;
        std::array<uint8_t,32> segment_digest{};
        if(!sha256(raw.data()+section_offset+segment_offset,segment_bytes,&segment_digest)||
           std::memcmp(segment_digest.data(),raw.data()+first+24,32)||
           !names[profile].insert(name).second)return hipErrorInvalidValue;
        cursors[profile]+=segment_bytes;++segment_counts[profile];
    }
    if(cursors[0]!=strict_bytes||!segment_counts[0]||
       ((flags&2)&&((cursors[1]!=fp8_bytes)||!segment_counts[1])))return hipErrorInvalidValue;
    const size_t offset=size_t(package_value<uint64_t>(raw,approximate?40:24));
    const size_t bytes=size_t(package_value<uint64_t>(raw,approximate?48:32));
    const size_t sha_offset=approximate?120:88;
    if(!bytes||offset<kPackageHeaderBytes||offset>raw.size()||bytes>raw.size()-offset)
        return hipErrorInvalidValue;
    std::array<uint8_t,32> digest{};
    if(!sha256(raw.data()+offset,bytes,&digest)||
       std::memcmp(digest.data(),raw.data()+sha_offset,32))return hipErrorInvalidValue;
    void* weights{};hipError_t error=hipMalloc(&weights,bytes);
    if(error==hipSuccess)error=hipMemcpyAsync(weights,raw.data()+offset,bytes,
        hipMemcpyHostToDevice,plan->stream);
    if(error==hipSuccess)error=hipStreamSynchronize(plan->stream);
    if(error!=hipSuccess){if(weights)(void)hipFree(weights);return error;}
    NRModelPackageStats stats{};stats.package_version=2;
    stats.precision_profile=uint32_t(plan->precision);stats.selected_weight_bytes=bytes;
    stats.selected_weight_segments=segment_counts[approximate?1:0];
    if(!package_text(raw,152,32,stats.architecture,sizeof(stats.architecture))||
       !package_text(raw,184,64,stats.temporal_contract_id,sizeof(stats.temporal_contract_id))||
       !package_text(raw,248,16,stats.target_arch,sizeof(stats.target_arch))){
        (void)hipFree(weights);return hipErrorInvalidValue;
    }
    hex_sha(raw.data()+56,stats.original_model_sha256);
    hex_sha(digest.data(),stats.selected_weights_sha256);
    // No temporal contract has passed the RTX sequence/difference gate yet.
    // Merely choosing a non-empty string in a package must never enable it.
    stats.temporal_contract_verified=0;plan->temporal_contract_verified=false;
    plan->weights=weights;plan->desc.weight_bytes=bytes;plan->weights_uploaded=true;
    *out_stats=stats;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanSetRecorder(NRPlan* plan,NRPlanRecordFn record,void* user) {
    if(!plan||!record||plan->finalized||plan->record_v3||
       !plan->approximate_stage_blocks.empty()||!plan->approximate_split512_blocks.empty()||
       !plan->approximate_vit_blocks.empty()||plan->has_approximate_bottleneck)
        return hipErrorInvalidValue;
    plan->record=record;plan->record_user=user;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanSetRecorderV3(NRPlan* plan,NRPlanRecordFnV3 record,void* user) {
    if(!plan||!record||plan->finalized||plan->record||
       !plan->approximate_stage_blocks.empty()||!plan->approximate_split512_blocks.empty()||
       !plan->approximate_vit_blocks.empty()||plan->has_approximate_bottleneck)
        return hipErrorInvalidValue;
    plan->record_v3=record;plan->record_user=user;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanSetPrecisionProfile(NRPlan* plan,NRPrecisionProfile profile) {
    if(!plan||plan->finalized||profile>NR_PRECISION_APPROX_FP8)return hipErrorInvalidValue;
    plan->precision=profile;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanPrepareShape(NRPlan* plan,const NRShapeDesc* shape) {
    if(!plan||!shape||shape->struct_size!=sizeof(NRShapeDesc)||plan->finalized||
       !shape->render_width||!shape->render_height||!shape->output_width||
       !shape->output_height||shape->output_width>plan->desc.max_width||
       shape->output_height>plan->desc.max_height||!shape->valid_width||
       !shape->valid_height||shape->valid_left>shape->output_width||
       shape->valid_top>shape->output_height||
       shape->valid_width>shape->output_width-shape->valid_left||
       shape->valid_height>shape->output_height-shape->valid_top||
       shape->color_mode>NR_COLOR_HDR_LINEAR||
       shape->color_format!=NR_FORMAT_RGBA16_FLOAT||
       shape->motion_format!=NR_FORMAT_RG16_FLOAT||
       shape->depth_format!=NR_FORMAT_R32_FLOAT)return hipErrorInvalidValue;
    plan->shape=*shape;plan->prepared=true;plan->submitted=false;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanGetStream(NRPlan* plan,hipStream_t* out_stream) {
    if(!plan||!out_stream)return hipErrorInvalidValue;
    *out_stream=plan->stream;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanAdoptGraph(NRPlan* plan,hipGraph_t source_graph) {
    if(!plan||!source_graph||plan->finalized||plan->record||plan->record_v3||
       !plan->approximate_stage_blocks.empty()||!plan->approximate_split512_blocks.empty()||
       !plan->approximate_vit_blocks.empty()||plan->has_approximate_bottleneck)
        return hipErrorInvalidValue;
    hipGraphExec_t executable{};
    hipError_t error=instantiate_graph(source_graph,&executable);
    if(error!=hipSuccess)return error;
    plan->graph=source_graph;plan->owns_graph=false;
    plan->external_graph_allocations=true;
    plan->executable=executable;plan->finalized=true;
    return hipSuccess;
}

static hipError_t graph_nodes(NRPlan* plan,std::vector<hipGraphNode_t>* nodes) {
    if(!plan||!nodes||!plan->graph||!plan->finalized)return hipErrorInvalidValue;
    size_t count=0;hipError_t error=hipGraphGetNodes(plan->graph,nullptr,&count);
    if(error!=hipSuccess)return error;
    nodes->resize(count);
    return count?hipGraphGetNodes(plan->graph,nodes->data(),&count):hipSuccess;
}

NRPLAN_API hipError_t nrPlanGetGraphStats(NRPlan* plan,NRPlanGraphStats* out_stats) {
    if(!plan||!out_stats||!plan->graph||!plan->finalized)return hipErrorInvalidValue;
    std::vector<hipGraphNode_t> nodes;hipError_t error=graph_nodes(plan,&nodes);
    if(error!=hipSuccess)return error;
    NRPlanGraphStats stats{};stats.total_nodes=nodes.size();
    for(size_t i=0;i<nodes.size();++i){
        hipGraphNodeType type{};
        error=hipGraphNodeGetType(nodes[i],&type);
        if(error!=hipSuccess)return error;
        if(type==hipGraphNodeTypeKernel)++stats.kernel_nodes;
        else if(type==hipGraphNodeTypeMemcpy)++stats.memcpy_nodes;
    }
    *out_stats=stats;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanGetKernelNodeInfo(NRPlan* plan,
    uint64_t wanted,NRPlanKernelNodeInfo* out_info) {
    if(!out_info)return hipErrorInvalidValue;
    std::vector<hipGraphNode_t> nodes;hipError_t error=graph_nodes(plan,&nodes);
    if(error!=hipSuccess)return error;
    uint64_t ordinal=0;
    for(uint64_t graph_index=0;graph_index<nodes.size();++graph_index){
        hipGraphNodeType type{};error=hipGraphNodeGetType(nodes[graph_index],&type);
        if(error!=hipSuccess)return error;
        if(type!=hipGraphNodeTypeKernel)continue;
        if(ordinal++!=wanted)continue;
        hipKernelNodeParams params{};
        error=hipGraphKernelNodeGetParams(nodes[graph_index],&params);
        if(error!=hipSuccess)return error;
        NRPlanKernelNodeInfo info{};
        info.graph_node_ordinal=graph_index;info.kernel_ordinal=wanted;
        info.grid_x=params.gridDim.x;info.grid_y=params.gridDim.y;info.grid_z=params.gridDim.z;
        info.block_x=params.blockDim.x;info.block_y=params.blockDim.y;info.block_z=params.blockDim.z;
        info.dynamic_shared_bytes=params.sharedMemBytes;
        hipFuncAttributes attr{};
        if(hipFuncGetAttributes(&attr,params.func)==hipSuccess){
            info.registers_per_thread=attr.numRegs;
            info.static_shared_bytes=attr.sharedSizeBytes;
            info.local_bytes_per_thread=attr.localSizeBytes;
            info.max_threads_per_block=attr.maxThreadsPerBlock;
        }else (void)hipGetLastError();
        const char* name=nullptr;
        if(hipKernelGetName(&name,reinterpret_cast<hipKernel_t>(params.func))==hipSuccess&&name)
            std::strncpy(info.name,name,sizeof(info.name)-1);
        else{
            (void)hipGetLastError();
            std::strncpy(info.name,"<unresolved>",sizeof(info.name)-1);
        }
        *out_info=info;return hipSuccess;
    }
    return hipErrorInvalidValue;
}

NRPLAN_API hipError_t nrPlanGetKernelNodeInfos(NRPlan* plan,
    NRPlanKernelNodeInfo* out_infos,uint64_t capacity,uint64_t* out_count) {
    if(!out_count)return hipErrorInvalidValue;
    std::vector<hipGraphNode_t> nodes;hipError_t error=graph_nodes(plan,&nodes);
    if(error!=hipSuccess)return error;
    uint64_t count=0;
    for(auto node:nodes){
        hipGraphNodeType type{};error=hipGraphNodeGetType(node,&type);
        if(error!=hipSuccess)return error;
        if(type==hipGraphNodeTypeKernel)++count;
    }
    *out_count=count;
    if(!out_infos)return capacity?hipErrorInvalidValue:hipSuccess;
    if(capacity<count)return hipErrorInvalidValue;
    uint64_t ordinal=0;
    for(uint64_t graph_index=0;graph_index<nodes.size();++graph_index){
        hipGraphNodeType type{};error=hipGraphNodeGetType(nodes[graph_index],&type);
        if(error!=hipSuccess)return error;
        if(type!=hipGraphNodeTypeKernel)continue;
        auto& info=out_infos[ordinal];info={};
        info.graph_node_ordinal=graph_index;info.kernel_ordinal=ordinal++;
        hipKernelNodeParams params{};error=hipGraphKernelNodeGetParams(nodes[graph_index],&params);
        if(error!=hipSuccess)return error;
        info.grid_x=params.gridDim.x;info.grid_y=params.gridDim.y;info.grid_z=params.gridDim.z;
        info.block_x=params.blockDim.x;info.block_y=params.blockDim.y;info.block_z=params.blockDim.z;
        info.dynamic_shared_bytes=params.sharedMemBytes;
        hipFuncAttributes attr{};
        if(hipFuncGetAttributes(&attr,params.func)==hipSuccess){
            info.registers_per_thread=attr.numRegs;info.static_shared_bytes=attr.sharedSizeBytes;
            info.local_bytes_per_thread=attr.localSizeBytes;
            info.max_threads_per_block=attr.maxThreadsPerBlock;
        }else (void)hipGetLastError();
        const char* name=nullptr;
        if(hipKernelGetName(&name,reinterpret_cast<hipKernel_t>(params.func))==hipSuccess&&name)
            std::strncpy(info.name,name,sizeof(info.name)-1);
        else{
            (void)hipGetLastError();
            std::strncpy(info.name,"<unresolved>",sizeof(info.name)-1);
        }
    }
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanDebugDotPrint(NRPlan* plan,const char* path) {
    if(!plan||!path||!path[0]||!plan->graph||!plan->finalized)return hipErrorInvalidValue;
    return hipGraphDebugDotPrint(plan->graph,path,
        hipGraphDebugDotFlagsVerbose|hipGraphDebugDotFlagsKernelNodeParams|
        hipGraphDebugDotFlagsMemcpyNodeParams|hipGraphDebugDotFlagsHandles);
}

NRPLAN_API hipError_t nrPlanDebugGetKernelU64Arguments(NRPlan* plan,
    uint64_t wanted,uint32_t argument_count,uint64_t* out_arguments) {
    if(!argument_count||argument_count>16||!out_arguments)return hipErrorInvalidValue;
    std::vector<hipGraphNode_t> nodes;hipError_t error=graph_nodes(plan,&nodes);
    if(error!=hipSuccess)return error;
    uint64_t ordinal=0;
    for(auto node:nodes){
        hipGraphNodeType type{};error=hipGraphNodeGetType(node,&type);
        if(error!=hipSuccess)return error;
        if(type!=hipGraphNodeTypeKernel)continue;
        if(ordinal++!=wanted)continue;
        hipKernelNodeParams params{};error=hipGraphKernelNodeGetParams(node,&params);
        if(error!=hipSuccess||!params.kernelParams)
            return error==hipSuccess?hipErrorInvalidValue:error;
        for(uint32_t i=0;i<argument_count;++i){
            if(!params.kernelParams[i])return hipErrorInvalidValue;
            std::memcpy(&out_arguments[i],params.kernelParams[i],sizeof(uint64_t));
        }
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

NRPLAN_API hipError_t nrPlanDebugGetOwnedAddresses(NRPlan* plan,
    uint64_t* out_workspace,uint64_t* out_weights,uint64_t* out_stage_e4_lut) {
    if(!plan||!out_workspace||!out_weights||!out_stage_e4_lut)
        return hipErrorInvalidValue;
    *out_workspace=reinterpret_cast<uint64_t>(plan->workspace);
    *out_weights=reinterpret_cast<uint64_t>(plan->weights);
    *out_stage_e4_lut=reinterpret_cast<uint64_t>(plan->stage_e4_lut);
    return hipSuccess;
}

__global__ static void nrplan_stage_marker(uint32_t* scratch,uint32_t boundary) {
    if(blockIdx.x==0&&threadIdx.x==0)*scratch=boundary;
}

NRPLAN_API hipError_t nrPlanRecordStageMarker(hipStream_t stream,
    uint32_t stage_boundary,uint32_t* device_scratch) {
    if(!stream||!device_scratch||stage_boundary>71)return hipErrorInvalidValue;
    hipLaunchKernelGGL(nrplan_stage_marker,dim3(1),dim3(1),0,stream,device_scratch,stage_boundary);
    return hipGetLastError();
}

NRPLAN_API hipError_t nrPlanGetResourceStats(NRPlan* plan,NRPlanResourceStats* out_stats) {
    if(!plan||!out_stats)return hipErrorInvalidValue;
    NRPlanResourceStats stats{};
    stats.workspace_bytes=plan->desc.workspace_bytes;stats.weight_bytes=plan->desc.weight_bytes;
    stats.arena_region_count=plan->arena_regions.size();stats.owns_workspace=plan->workspace!=nullptr;
    stats.owns_weights=plan->weights!=nullptr;stats.owns_graph_source=plan->graph&&plan->owns_graph;
    stats.owns_graph_executable=plan->executable!=nullptr;
    stats.graph_references_external_allocations=plan->external_graph_allocations;
    // A legacy recorder or an adopted PyTorch graph may still own its executable,
    // but it is not the ABI-v3 deployment topology.  Do not report v3 merely
    // because this version of the DLL happens to know that structure.
    const bool builtin=!plan->approximate_stage_blocks.empty()||
                       !plan->approximate_split512_blocks.empty()||
                       !plan->approximate_vit_blocks.empty()||plan->has_approximate_bottleneck;
    stats.frame_bindings_abi_version=(plan->record_v3||builtin)?
        NR_PLAN_FRAME_BINDINGS_ABI_VERSION_V3:NR_PLAN_FRAME_BINDINGS_ABI_VERSION;
    stats.deployment_ready=stats.owns_workspace&&stats.owns_weights&&stats.owns_graph_source&&
        stats.owns_graph_executable&&!stats.graph_references_external_allocations&&
        stats.arena_region_count!=0&&(plan->record_v3||builtin)&&plan->prepared&&
        plan->weights_uploaded&&plan->finalized&&!plan->record_user;
    stats.native_stage_block_count=uint32_t(plan->approximate_stage_blocks.size()+
                                             plan->approximate_split512_blocks.size()+
                                             plan->approximate_vit_blocks.size()+
                                             (plan->has_approximate_bottleneck?1:0));
    stats.complete_native_topology=plan->complete_native_topology;
    stats.deployment_ready=stats.deployment_ready&&stats.complete_native_topology;
    stats.temporal_contract_verified=plan->temporal_contract_verified;
    stats.precision_profile=uint8_t(plan->precision);
    *out_stats=stats;return hipSuccess;
}

NRPLAN_API hipError_t nrPlanConfigureArena(NRPlan* plan,
    const NRPlanArenaRegion* regions,uint64_t count) {
    if(!plan||plan->finalized||(count&&!regions))return hipErrorInvalidValue;
    std::vector<NRPlanArenaRegion> copy;copy.reserve(size_t(count));
    for(uint64_t i=0;i<count;++i){
        NRPlanArenaRegion region=regions[i];region.name[sizeof(region.name)-1]=0;
        if(!region.bytes||!region.alignment||(region.alignment&(region.alignment-1))||
           region.offset%region.alignment||region.offset>plan->desc.workspace_bytes||
           region.bytes>plan->desc.workspace_bytes-region.offset||!region.name[0])
            return hipErrorInvalidValue;
        copy.push_back(region);
    }
    std::sort(copy.begin(),copy.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
    for(size_t i=1;i<copy.size();++i)
        if(copy[i-1].offset+copy[i-1].bytes>copy[i].offset)return hipErrorInvalidValue;
    plan->arena_regions=std::move(copy);return hipSuccess;
}

static bool range_fits(uint64_t offset,uint64_t bytes,uint64_t capacity) {
    return bytes&&offset<=capacity&&bytes<=capacity-offset;
}

static bool range_in_arena(const NRPlan* plan,uint64_t offset,uint64_t bytes) {
    if(!range_fits(offset,bytes,plan->desc.workspace_bytes))return false;
    for(const auto& region:plan->arena_regions){
        if(offset<region.offset)continue;
        uint64_t relative=offset-region.offset;
        if(relative<=region.bytes&&bytes<=region.bytes-relative)return true;
    }
    return false;
}

NRPLAN_API hipError_t nrPlanConfigureApproxStageBlocks(NRPlan* plan,
    const NRApproxStageBlockDesc* blocks,uint32_t count,uint8_t complete_native_topology) {
    // The current built-in recorder covers standard Swin blocks, but not
    // Pre/Head/transitions. Refuse a caller attempting to label
    // this partial topology as deployable.
    if(!plan||!blocks||!count||count>256||complete_native_topology||plan->finalized||
       plan->record||plan->record_v3||plan->precision!=NR_PRECISION_APPROX_FP8||
       !plan->weights_uploaded||plan->arena_regions.empty())return hipErrorInvalidValue;
    hipError_t lut_error=ensure_stage_e4_lut(plan);
    if(lut_error!=hipSuccess)return lut_error;
    std::vector<NRApproxStageBlockDesc> copy;copy.reserve(count);
    for(uint32_t i=0;i<count;++i){
        NRApproxStageBlockDesc block=blocks[i];
        if(block.struct_size!=sizeof(NRApproxStageBlockDesc)||
           block.record_number>70||(i&&block.record_number<=blocks[i-1].record_number)||
           (block.channels!=32&&block.channels!=64&&block.channels!=128&&
            block.channels!=256)||
           !block.windows||block.windows>262144||!block.feature_width||!block.feature_height||
           block.feature_width%4||block.feature_height%4||
           (block.origin_x!=0&&block.origin_x!=-4)||(block.origin_y!=0&&block.origin_y!=-4))
            return hipErrorInvalidValue;
        uint64_t expected_windows=uint64_t((block.feature_width-block.origin_x+7)/8)*
            uint64_t((block.feature_height-block.origin_y+7)/8);
        if(block.windows!=expected_windows)return hipErrorInvalidValue;
        uint64_t elements=uint64_t(block.windows)*64*block.channels;
        uint64_t image_elements=uint64_t(block.feature_width)*block.feature_height*block.channels;
        uint64_t heads=block.channels/32;
        if(elements>std::numeric_limits<uint64_t>::max()/2)return hipErrorInvalidValue;
        const uint64_t fp16_bytes=elements*2;
        const std::array<std::pair<uint64_t,uint64_t>,12> arena_ranges{{
            {block.raw_resident_offset,image_elements},{block.grouped_offset,elements},
            {block.seed_fp16_offset,fp16_bytes},{block.post_fp16_offset,fp16_bytes},
            {block.post_resident_offset,elements},
            {block.qkv_projection_fp16_offset,fp16_bytes*3},
            {block.q_offset,elements},{block.k_offset,elements},{block.v_offset,elements},
            {block.value_offset,elements},{block.output_fp16_offset,fp16_bytes},
            {block.next_resident_offset,image_elements}}};
        for(const auto& range:arena_ranges)
            if(!range_in_arena(plan,range.first,range.second))return hipErrorInvalidValue;
        const std::array<std::pair<uint64_t,uint64_t>,14> weight_ranges{{
            {block.ffn_expand_weight_offset,uint64_t(block.channels)*block.channels*4},
            {block.ffn_contract_weight_offset,uint64_t(block.channels)*128},
            {block.ffn_mix_weight_offset,uint64_t(block.channels)*block.channels},
            {block.ffn_scale_weight_offset,uint64_t(block.channels)*2},
            {block.a_index_weight_offset,uint64_t(block.channels)*64*4},
            {block.residual_index_weight_offset,uint64_t(block.channels)*64*4},
            {block.ffn_permutation_weight_offset,32*4},
            {block.ffn_inverse_permutation_weight_offset,32*4},
            {block.qkv_weight_offset,uint64_t(block.channels)*block.channels*3},
            {block.qscale_weight_offset,heads*2},
            {block.permutation_weight_offset,uint64_t(block.channels)*4},
            {block.position_bias_weight_offset,heads*64*64*2},
            {block.project_weight_offset,uint64_t(block.channels)*block.channels},
            {block.residual_scale_weight_offset,uint64_t(block.channels)*2}}};
        for(size_t range_number=0;range_number<weight_ranges.size();++range_number){
            // C32 has one FFN group and therefore no cross-head mix matrix.
            if(block.channels==32&&range_number==2){
                if(block.ffn_mix_weight_offset)return hipErrorInvalidValue;
                continue;
            }
            const auto& range=weight_ranges[range_number];
            if(!range_fits(range.first,range.second,plan->desc.weight_bytes))
                return hipErrorInvalidValue;
        }
        copy.push_back(block);
    }
    for(const auto& split:plan->approximate_split512_blocks)
        for(const auto& block:copy)
            if(split.record_number==block.record_number)return hipErrorInvalidValue;
    for(const auto& vit:plan->approximate_vit_blocks)
        for(const auto& block:copy)
            if(vit.record_number==block.record_number)return hipErrorInvalidValue;
    plan->approximate_stage_blocks=std::move(copy);
    plan->complete_native_topology=false;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanConfigureApproxSplit512Blocks(NRPlan* plan,
    const NRApproxSplit512BlockDesc* blocks,uint32_t count) {
    if(!plan||!blocks||!count||count>16||plan->finalized||plan->record||plan->record_v3||
       plan->precision!=NR_PRECISION_APPROX_FP8||!plan->weights_uploaded||
       plan->arena_regions.empty())return hipErrorInvalidValue;
    hipError_t lut_error=ensure_stage_e4_lut(plan);
    if(lut_error!=hipSuccess)return lut_error;
    std::vector<NRApproxSplit512BlockDesc> copy;copy.reserve(count);
    for(uint32_t i=0;i<count;++i){
        NRApproxSplit512BlockDesc block=blocks[i];
        const bool record_ok=(block.record_number>=23&&block.record_number<=30)||
                             (block.record_number>=40&&block.record_number<=47);
        if(block.struct_size!=sizeof(NRApproxSplit512BlockDesc)||block.reserved||!record_ok||
           (i&&block.record_number<=blocks[i-1].record_number)||!block.windows||
           block.windows>262144||!block.feature_width||!block.feature_height||
           block.feature_width%4||block.feature_height%4||
           (block.origin_x!=0&&block.origin_x!=-4)||
           (block.origin_y!=0&&block.origin_y!=-4))return hipErrorInvalidValue;
        uint64_t expected_windows=uint64_t((block.feature_width-block.origin_x+7)/8)*
            uint64_t((block.feature_height-block.origin_y+7)/8);
        if(block.windows!=expected_windows)return hipErrorInvalidValue;
        uint64_t elements=uint64_t(block.windows)*64*512;
        uint64_t image_elements=uint64_t(block.feature_width)*block.feature_height*512;
        if(elements>std::numeric_limits<uint64_t>::max()/2)return hipErrorInvalidValue;
        uint64_t fp16_bytes=elements*2;
        const std::array<std::pair<uint64_t,uint64_t>,11> arena_ranges{{
            {block.raw_resident_offset,image_elements},{block.projected_offset,elements},
            {block.grouped_offset,elements},{block.post_fp16_offset,fp16_bytes},
            {block.post_resident_offset,elements},{block.q_offset,elements},
            {block.k_offset,elements},{block.v_offset,elements},
            {block.value_offset,elements},{block.output_fp16_offset,fp16_bytes},
            {block.next_resident_offset,image_elements}}};
        for(const auto& range:arena_ranges)
            if(!range_in_arena(plan,range.first,range.second))return hipErrorInvalidValue;
        const std::array<std::pair<uint64_t,uint64_t>,16> weight_ranges{{
            {block.preproject_weight_offset,512ull*512},
            {block.expand_weight_offset,8ull*64*256},
            {block.contract_weight_offset,8ull*256*64},
            {block.ffn_project_weight_offset,512ull*512},
            {block.ffn_scale_weight_offset,512ull*2},
            {block.a_index_weight_offset,64ull*512*4},
            {block.residual_index_weight_offset,64ull*512*4},
            {block.perm64_weight_offset,64ull*4},
            {block.perm256_weight_offset,256ull*4},
            {block.ffn_permutation_weight_offset,512ull*4},
            {block.qkv_weight_offset,512ull*1536},
            {block.qscale_weight_offset,16ull*2},
            {block.attention_permutation_weight_offset,512ull*4},
            {block.position_bias_weight_offset,16ull*64*64*2},
            {block.attention_project_weight_offset,512ull*512},
            {block.attention_scale_weight_offset,512ull*2}}};
        for(const auto& range:weight_ranges)
            if(!range_fits(range.first,range.second,plan->desc.weight_bytes))
                return hipErrorInvalidValue;
        for(const auto& standard:plan->approximate_stage_blocks)
            if(standard.record_number==block.record_number)return hipErrorInvalidValue;
        for(const auto& vit:plan->approximate_vit_blocks)
            if(vit.record_number==block.record_number)return hipErrorInvalidValue;
        copy.push_back(block);
    }
    plan->approximate_split512_blocks=std::move(copy);
    plan->complete_native_topology=false;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanConfigureApproxVitBlocks(NRPlan* plan,
    const NRApproxVitBlockDesc* blocks,uint32_t count) {
    if(!plan||!blocks||count!=8||plan->finalized||plan->record||plan->record_v3||
       plan->precision!=NR_PRECISION_APPROX_FP8||!plan->weights_uploaded||
       plan->arena_regions.empty())return hipErrorInvalidValue;
    std::vector<NRApproxVitBlockDesc> copy;copy.reserve(count);
    for(uint32_t i=0;i<count;++i){
        NRApproxVitBlockDesc block=blocks[i];
        if(block.struct_size!=sizeof(NRApproxVitBlockDesc)||block.reserved||
           block.record_number!=31+i||block.tokens!=640)return hipErrorInvalidValue;
        const uint64_t activation=uint64_t(block.tokens)*1024;
        const uint64_t hidden=uint64_t(block.tokens)*4096;
        const std::array<std::pair<uint64_t,uint64_t>,8> arena_ranges{{
            {block.input_offset,activation},{block.hidden_offset,hidden},
            {block.post_offset,activation},{block.q_offset,activation},
            {block.k_offset,activation},{block.v_offset,activation},
            {block.value_offset,activation},{block.next_offset,activation}}};
        for(const auto& range:arena_ranges)
            if(!range_in_arena(plan,range.first,range.second))return hipErrorInvalidValue;
        const std::array<std::pair<uint64_t,uint64_t>,9> weight_ranges{{
            {block.expand_weight_offset,1024ull*4096},
            {block.contract_weight_offset,4096ull*1024},
            {block.qkv_weight_offset,1024ull*3072},
            {block.q_scale_weight_offset,32ull*2},
            {block.project_weight_offset,1024ull*1024},
            {block.ffn_scale_weight_offset,1024ull*2},
            {block.attention_scale_weight_offset,1024ull*2},
            {block.perm1024_weight_offset,1024ull*4},
            {block.perm4096_weight_offset,4096ull*4}}};
        for(const auto& range:weight_ranges)
            if(!range_fits(range.first,range.second,plan->desc.weight_bytes))
                return hipErrorInvalidValue;
        for(const auto& standard:plan->approximate_stage_blocks)
            if(standard.record_number==block.record_number)return hipErrorInvalidValue;
        for(const auto& split:plan->approximate_split512_blocks)
            if(split.record_number==block.record_number)return hipErrorInvalidValue;
        copy.push_back(block);
    }
    plan->approximate_vit_blocks=std::move(copy);
    plan->complete_native_topology=false;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanConfigureApproxBottleneck(NRPlan* plan,
    const NRApproxBottleneckDesc* descriptor) {
    if(!plan||!descriptor||plan->has_approximate_bottleneck||plan->finalized||
       plan->record||plan->record_v3||plan->precision!=NR_PRECISION_APPROX_FP8||
       !plan->weights_uploaded||plan->arena_regions.empty()||
       plan->approximate_split512_blocks.size()!=16||
       plan->approximate_vit_blocks.size()!=8)return hipErrorInvalidValue;
    NRApproxBottleneckDesc value=*descriptor;
    if(value.struct_size!=sizeof(NRApproxBottleneckDesc)||value.tokens!=640||
       value.feature_width!=60||value.feature_height!=36||value.low_width!=32||
       value.low_height!=20||value.reserved0||value.reserved1)
        return hipErrorInvalidValue;
    const uint64_t c512=uint64_t(value.feature_width)*value.feature_height*512;
    const uint64_t vit=uint64_t(value.tokens)*1024;
    const std::array<std::pair<uint64_t,uint64_t>,5> arena_ranges{{
        {value.c512_input_offset,c512},{value.c512_skip_offset,c512},
        {value.vit_input_offset,vit},{value.vit_output_offset,vit},
        {value.c512_output_offset,c512}}};
    for(const auto& range:arena_ranges)
        if(!range_in_arena(plan,range.first,range.second))return hipErrorInvalidValue;
    const std::array<std::pair<uint64_t,uint64_t>,5> weight_ranges{{
        {value.encoder_weight_offset,512ull*1024},
        {value.encoder_permutation_offset,512ull*4},
        {value.decoder_weight_offset,1024ull*512},
        {value.decoder_scale_offset,512ull*2},
        {value.decoder_permutation_offset,1024ull*4}}};
    for(const auto& range:weight_ranges)
        if(!range_fits(range.first,range.second,plan->desc.weight_bytes))
            return hipErrorInvalidValue;
    plan->approximate_bottleneck=value;plan->has_approximate_bottleneck=true;
    plan->complete_native_topology=false;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanConfigureApproxScaleTransitions(NRPlan* plan,
    const NRApproxScaleTransitionDesc* descriptors,uint32_t count) {
    if(!plan||!descriptors||count!=8||plan->finalized||plan->record||plan->record_v3||
       plan->precision!=NR_PRECISION_APPROX_FP8||!plan->weights_uploaded||
       plan->arena_regions.empty()||plan->approximate_stage_blocks.size()!=44||
       plan->approximate_split512_blocks.size()!=16)return hipErrorInvalidValue;
    constexpr std::array<uint32_t,4> encoder_records{{4,8,14,22}};
    constexpr std::array<uint32_t,4> decoder_records{{48,56,62,66}};
    constexpr std::array<uint32_t,4> encoder_channels{{32,64,128,256}};
    constexpr std::array<uint32_t,4> decoder_channels{{256,128,64,32}};
    std::vector<NRApproxScaleTransitionDesc> copy;copy.reserve(count);
    uint32_t encoder_number=0,decoder_number=0;
    for(uint32_t i=0;i<count;++i){
        NRApproxScaleTransitionDesc value=descriptors[i];
        if(value.struct_size!=sizeof(NRApproxScaleTransitionDesc)||
           (i&&value.anchor_record<=descriptors[i-1].anchor_record)||
           (value.channels!=32&&value.channels!=64&&value.channels!=128&&
           value.channels!=256)||!value.source_width||!value.source_height||
           !value.target_width||!value.target_height||
           (value.source_origin_x!=0&&value.source_origin_x!=-4)||
           (value.source_origin_y!=0&&value.source_origin_y!=-4))return hipErrorInvalidValue;
        uint64_t source_channels=0,target_channels=0;
        if(value.direction==NR_TRANSITION_ENCODER_DOWNSAMPLE){
            if(encoder_number>=4||value.anchor_record!=encoder_records[encoder_number]||
               value.channels!=encoder_channels[encoder_number]||
               value.source_width!=value.target_width*2||
               value.source_height!=value.target_height*2||value.skip_scale_weight_offset)
                return hipErrorInvalidValue;
            source_channels=value.channels;target_channels=value.channels*2;++encoder_number;
        }else if(value.direction==NR_TRANSITION_DECODER_UPSAMPLE){
            if(decoder_number>=4||value.anchor_record!=decoder_records[decoder_number]||
               value.channels!=decoder_channels[decoder_number]||
               value.target_width!=value.source_width*2||
               value.target_height!=value.source_height*2||value.source_origin_x||
               value.source_origin_y||value.source_offset!=value.source_resident_offset)
                return hipErrorInvalidValue;
            source_channels=value.channels*2;target_channels=value.channels;++decoder_number;
        }else return hipErrorInvalidValue;
        if((value.source_width&3)||(value.source_height&3)||(value.target_width&3)||
           (value.target_height&3))return hipErrorInvalidValue;
        const uint64_t resident_source_bytes=uint64_t(value.source_width)*value.source_height*source_channels;
        const uint64_t source_windows=uint64_t((value.source_width-value.source_origin_x+7)/8)*
            uint64_t((value.source_height-value.source_origin_y+7)/8);
        const uint64_t source_bytes=value.direction==NR_TRANSITION_ENCODER_DOWNSAMPLE?
            source_windows*64*value.channels*2:resident_source_bytes;
        const uint64_t target_bytes=uint64_t(value.target_width)*value.target_height*target_channels;
        const uint64_t skip_bytes=value.direction==NR_TRANSITION_ENCODER_DOWNSAMPLE?
            resident_source_bytes:uint64_t(value.target_width)*value.target_height*value.channels;
        if(!range_in_arena(plan,value.source_offset,source_bytes)||
           !range_in_arena(plan,value.source_resident_offset,resident_source_bytes)||
           !range_in_arena(plan,value.skip_offset,skip_bytes)||
           !range_in_arena(plan,value.target_offset,target_bytes)||
           !range_fits(value.project_weight_offset,
               uint64_t(value.channels)*value.channels*2,plan->desc.weight_bytes)||
           !range_fits(value.permutation_weight_offset,source_channels*4,
                       plan->desc.weight_bytes)||
           (value.direction==NR_TRANSITION_DECODER_UPSAMPLE&&
            !range_fits(value.skip_scale_weight_offset,uint64_t(value.channels)*2,
                        plan->desc.weight_bytes)))return hipErrorInvalidValue;
        copy.push_back(value);
    }
    if(encoder_number!=4||decoder_number!=4)return hipErrorInvalidValue;
    plan->approximate_scale_transitions=std::move(copy);
    plan->complete_native_topology=false;
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanInitializeArenaFromDevice(NRPlan* plan,uint64_t offset,
    const void* source,uint64_t bytes) {
    if(!plan||!source||plan->finalized||!range_in_arena(plan,offset,bytes))
        return hipErrorInvalidValue;
    hipError_t error=hipMemcpyAsync(static_cast<uint8_t*>(plan->workspace)+offset,
        source,size_t(bytes),hipMemcpyDeviceToDevice,plan->stream);
    return error==hipSuccess?hipStreamSynchronize(plan->stream):error;
}

NRPLAN_API hipError_t nrPlanDebugCopyArenaToDevice(NRPlan* plan,uint64_t offset,
    void* target,uint64_t bytes) {
    if(!plan||!target||!plan->finalized||!range_in_arena(plan,offset,bytes))
        return hipErrorInvalidValue;
    hipError_t error=hipStreamSynchronize(plan->stream);
    if(error==hipSuccess)error=hipMemcpyAsync(target,
        static_cast<uint8_t*>(plan->workspace)+offset,size_t(bytes),
        hipMemcpyDeviceToDevice,plan->stream);
    return error==hipSuccess?hipStreamSynchronize(plan->stream):error;
}

NRPLAN_API hipError_t nrPlanGetArenaRegion(NRPlan* plan,uint64_t index,
    NRPlanArenaRegion* out_region) {
    if(!plan||!out_region||index>=plan->arena_regions.size())return hipErrorInvalidValue;
    *out_region=plan->arena_regions[size_t(index)];return hipSuccess;
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

static hipError_t record_approximate_stage_block(NRPlan* plan,
    const NRApproxStageBlockDesc& block) {
    auto* workspace=static_cast<uint8_t*>(plan->workspace);
    auto* weights=static_cast<uint8_t*>(plan->weights);
    auto error_from=[](int value){return static_cast<hipError_t>(value);};
    const void* raw=workspace+block.raw_resident_offset;
    void* grouped=workspace+block.grouped_offset;
    void* seed=workspace+block.seed_fp16_offset;
    void* post=workspace+block.post_fp16_offset;
    void* resident=workspace+block.post_resident_offset;
    void* qkv_projection=workspace+block.qkv_projection_fp16_offset;
    const void* expand=weights+block.ffn_expand_weight_offset;
    const void* contract=weights+block.ffn_contract_weight_offset;
    const void* mix=weights+block.ffn_mix_weight_offset;
    const void* ffn_scale=weights+block.ffn_scale_weight_offset;
    const void* ai=weights+block.a_index_weight_offset;
    const void* ri=weights+block.residual_index_weight_offset;
    const void* ffn_permutation=weights+block.ffn_permutation_weight_offset;
    const void* inverse=weights+block.ffn_inverse_permutation_weight_offset;
    const void* qkv=weights+block.qkv_weight_offset;
    const void* qscale=weights+block.qscale_weight_offset;
    const void* permutation=weights+block.permutation_weight_offset;
    void* q=workspace+block.q_offset;void* k=workspace+block.k_offset;
    void* v=workspace+block.v_offset;void* value=workspace+block.value_offset;
    const void* bias=weights+block.position_bias_weight_offset;
    const void* project=weights+block.project_weight_offset;
    const void* scale=weights+block.residual_scale_weight_offset;
    void* output=workspace+block.output_fp16_offset;
    void* next=workspace+block.next_resident_offset;
    hipError_t error=hipErrorInvalidValue;
#define RECORD_STAGE(C) \
    error=error_from(nr_stage_c##C##_ffn_fp8(raw,expand,contract,mix,ffn_scale,ai,ri,ffn_permutation,inverse,grouped,seed,post,resident,int(block.windows),int(block.feature_width),int(block.feature_height),block.origin_x,block.origin_y,plan->stream)); \
    if(error==hipSuccess)error=error_from(nr_stage_c##C##_qkv_norm_fp8(resident,qkv,qscale,permutation,qkv_projection,q,k,v,plan->stage_e4_lut,int(block.windows),plan->stream)); \
    if(error==hipSuccess)error=error_from(nr_stage_c##C##_attention_fp8(q,k,v,bias,value,int(block.windows),plan->stream)); \
    if(error==hipSuccess)error=error_from(nr_stage_c##C##_project_fp8(post,value,project,scale,permutation,output,resident,int(block.windows),plan->stream)); \
    if(error==hipSuccess)error=error_from(nr_stage_c##C##_scatter_fp8(resident,next,int(block.feature_width),int(block.feature_height),block.origin_x,block.origin_y,plan->stream))
    if(block.channels==32){RECORD_STAGE(32);}
    else if(block.channels==64){RECORD_STAGE(64);}
    else if(block.channels==128){RECORD_STAGE(128);}
    else if(block.channels==256){RECORD_STAGE(256);}
#undef RECORD_STAGE
    return error;
}

static hipError_t record_approximate_split512_block(NRPlan* plan,
    const NRApproxSplit512BlockDesc& block) {
    auto* workspace=static_cast<uint8_t*>(plan->workspace);
    auto* weights=static_cast<uint8_t*>(plan->weights);
    auto error_from=[](int value){return static_cast<hipError_t>(value);};
    void* projected=workspace+block.projected_offset;
    void* grouped=workspace+block.grouped_offset;
    void* post=workspace+block.post_fp16_offset;
    void* resident=workspace+block.post_resident_offset;
    hipError_t error=error_from(nr_stage_c512_ffn_fp8(
        workspace+block.raw_resident_offset,weights+block.preproject_weight_offset,
        weights+block.expand_weight_offset,weights+block.contract_weight_offset,
        weights+block.ffn_project_weight_offset,weights+block.ffn_scale_weight_offset,
        weights+block.a_index_weight_offset,weights+block.residual_index_weight_offset,
        weights+block.perm64_weight_offset,weights+block.perm256_weight_offset,
        weights+block.ffn_permutation_weight_offset,projected,grouped,post,resident,
        int(block.windows),int(block.feature_width),int(block.feature_height),
        block.origin_x,block.origin_y,plan->stream));
    void* q=workspace+block.q_offset;void* k=workspace+block.k_offset;
    void* v=workspace+block.v_offset;void* value=workspace+block.value_offset;
    if(error==hipSuccess)error=error_from(nr_stage_c512_qkv_norm_fp8(resident,
        weights+block.qkv_weight_offset,weights+block.qscale_weight_offset,
        weights+block.attention_permutation_weight_offset,q,k,v,int(block.windows),plan->stream));
    if(error==hipSuccess)error=error_from(nr_stage_c512_attention_fp8(q,k,v,
        weights+block.position_bias_weight_offset,value,int(block.windows),plan->stream));
    if(error==hipSuccess)error=error_from(nr_stage_c512_project_fp8(post,value,
        weights+block.attention_project_weight_offset,
        weights+block.attention_scale_weight_offset,
        weights+block.attention_permutation_weight_offset,
        workspace+block.output_fp16_offset,resident,int(block.windows),plan->stream));
    if(error==hipSuccess)error=error_from(nr_stage_c512_scatter_fp8(resident,
        workspace+block.next_resident_offset,int(block.feature_width),int(block.feature_height),
        block.origin_x,block.origin_y,plan->stream));
    return error;
}

static hipError_t record_approximate_vit_block(NRPlan* plan,
    const NRApproxVitBlockDesc& block) {
    auto* workspace=static_cast<uint8_t*>(plan->workspace);
    auto* weights=static_cast<uint8_t*>(plan->weights);
    return static_cast<hipError_t>(nr_vit_block_fp8(
        workspace+block.input_offset,weights+block.expand_weight_offset,
        weights+block.contract_weight_offset,weights+block.qkv_weight_offset,
        weights+block.q_scale_weight_offset,weights+block.project_weight_offset,
        weights+block.ffn_scale_weight_offset,
        weights+block.attention_scale_weight_offset,
        weights+block.perm1024_weight_offset,weights+block.perm4096_weight_offset,
        workspace+block.hidden_offset,workspace+block.post_offset,
        workspace+block.q_offset,workspace+block.k_offset,workspace+block.v_offset,
        workspace+block.value_offset,workspace+block.next_offset,int(block.tokens),
        plan->stream));
}

static hipError_t record_encoder_bottleneck(NRPlan* plan) {
    const auto& b=plan->approximate_bottleneck;
    auto* workspace=static_cast<uint8_t*>(plan->workspace);
    auto* weights=static_cast<uint8_t*>(plan->weights);
    return static_cast<hipError_t>(nr_encoder_final_to_vit_fp8(
        workspace+b.c512_input_offset,workspace+b.c512_skip_offset,
        weights+b.encoder_weight_offset,weights+b.encoder_permutation_offset,
        workspace+b.vit_input_offset,int(b.feature_width),int(b.feature_height),
        int(b.low_width),int(b.low_height),plan->stream));
}

static hipError_t record_decoder_bottleneck(NRPlan* plan) {
    const auto& b=plan->approximate_bottleneck;
    auto* workspace=static_cast<uint8_t*>(plan->workspace);
    auto* weights=static_cast<uint8_t*>(plan->weights);
    return static_cast<hipError_t>(nr_decoder_input_from_vit_fp8(
        workspace+b.vit_output_offset,workspace+b.c512_skip_offset,
        weights+b.decoder_weight_offset,weights+b.decoder_scale_offset,
        weights+b.decoder_permutation_offset,workspace+b.c512_output_offset,
        int(b.feature_width),int(b.feature_height),int(b.low_width),
        int(b.low_height),plan->stream));
}

static hipError_t record_scale_transition(NRPlan* plan,
    const NRApproxScaleTransitionDesc& transition) {
    auto* workspace=static_cast<uint8_t*>(plan->workspace);
    auto* weights=static_cast<uint8_t*>(plan->weights);
    if(transition.direction==NR_TRANSITION_ENCODER_DOWNSAMPLE)
        return static_cast<hipError_t>(nr_transition_encoder_fp8(
            workspace+transition.source_offset,workspace+transition.source_resident_offset,
            workspace+transition.skip_offset,
            weights+transition.project_weight_offset,
            weights+transition.permutation_weight_offset,
            workspace+transition.target_offset,int(transition.source_width),
            int(transition.source_height),transition.source_origin_x,
            transition.source_origin_y,int(transition.channels),plan->stream));
    if(transition.direction==NR_TRANSITION_DECODER_UPSAMPLE)
        return static_cast<hipError_t>(nr_transition_decoder_fp8(
            workspace+transition.source_offset,workspace+transition.skip_offset,
            weights+transition.project_weight_offset,
            weights+transition.skip_scale_weight_offset,
            weights+transition.permutation_weight_offset,
            workspace+transition.target_offset,int(transition.target_width),
            int(transition.target_height),int(transition.channels),plan->stream));
    return hipErrorInvalidValue;
}

static hipError_t record_scale_transition_at(NRPlan* plan,uint32_t record,
    NRApproxTransitionDirection direction) {
    for(const auto& transition:plan->approximate_scale_transitions)
        if(transition.anchor_record==record&&transition.direction==uint32_t(direction))
            return record_scale_transition(plan,transition);
    return hipSuccess;
}

static hipError_t record_approximate_stage_blocks(NRPlan* plan) {
    size_t wide=0,split=0,vit=0;
    while(wide<plan->approximate_stage_blocks.size()||
          split<plan->approximate_split512_blocks.size()||
          vit<plan->approximate_vit_blocks.size()){
        uint32_t wide_record=wide<plan->approximate_stage_blocks.size()?
            plan->approximate_stage_blocks[wide].record_number:UINT32_MAX;
        uint32_t split_record=split<plan->approximate_split512_blocks.size()?
            plan->approximate_split512_blocks[split].record_number:UINT32_MAX;
        uint32_t vit_record=vit<plan->approximate_vit_blocks.size()?
            plan->approximate_vit_blocks[vit].record_number:UINT32_MAX;
        uint32_t recorded=UINT32_MAX;
        hipError_t error=hipSuccess;
        uint32_t kind=UINT32_MAX;
        if(wide_record<=split_record&&wide_record<=vit_record){
            recorded=wide_record;kind=0;
        }else if(split_record<=vit_record){
            recorded=split_record;kind=1;
        }else{
            recorded=vit_record;kind=2;
        }
        if(recorded==UINT32_MAX)return hipErrorInvalidValue;
        error=record_scale_transition_at(plan,recorded,NR_TRANSITION_DECODER_UPSAMPLE);
        if(error!=hipSuccess)return error;
        if(kind==0)error=record_approximate_stage_block(
            plan,plan->approximate_stage_blocks[wide++]);
        else if(kind==1)error=record_approximate_split512_block(
            plan,plan->approximate_split512_blocks[split++]);
        else error=record_approximate_vit_block(plan,plan->approximate_vit_blocks[vit++]);
        if(error!=hipSuccess)return error;
        error=record_scale_transition_at(plan,recorded,NR_TRANSITION_ENCODER_DOWNSAMPLE);
        if(error!=hipSuccess)return error;
        if(plan->has_approximate_bottleneck&&recorded==30){
            error=record_encoder_bottleneck(plan);if(error!=hipSuccess)return error;
        }
        if(plan->has_approximate_bottleneck&&recorded==38){
            error=record_decoder_bottleneck(plan);if(error!=hipSuccess)return error;
        }
    }
    return hipSuccess;
}

NRPLAN_API hipError_t nrPlanFinalize(NRPlan* plan) {
    const bool builtin=plan&&(!plan->approximate_stage_blocks.empty()||
                             !plan->approximate_split512_blocks.empty()||
                             !plan->approximate_vit_blocks.empty()||
                             plan->has_approximate_bottleneck);
    if(!plan||(!plan->record&&!plan->record_v3&&!builtin)||plan->finalized||
       !plan->weights_uploaded||((plan->record_v3||builtin)&&!plan->prepared))return hipErrorInvalidValue;
    hipError_t error=hipStreamBeginCapture(plan->stream,hipStreamCaptureModeGlobal);
    if(error!=hipSuccess)return error;
    error=builtin?record_approximate_stage_blocks(plan):plan->record_v3?
        plan->record_v3(plan->stream,plan->workspace,plan->weights,
            plan->device_bindings_v3,plan->record_user):
        plan->record(plan->stream,plan->workspace,plan->weights,
            plan->device_bindings,plan->record_user);
    hipGraph_t graph{};
    hipError_t end_error=hipStreamEndCapture(plan->stream,&graph);
    if(error!=hipSuccess){if(graph)hipGraphDestroy(graph);return error;}
    if(end_error!=hipSuccess)return end_error;
    hipGraphExec_t executable{};
    error=instantiate_graph(graph,&executable);
    if(error!=hipSuccess){hipGraphDestroy(graph);return error;}
    plan->graph=graph;plan->owns_graph=true;
    plan->external_graph_allocations=false;
    plan->executable=executable;plan->finalized=true;
    return hipSuccess;
}

static hipError_t harvest_timing(NRPlan* plan);

NRPLAN_API hipError_t nrPlanSubmit(NRPlan* plan,const NRFrameBindings* bindings,
    hipEvent_t input_ready,hipEvent_t output_done) {
    if(!plan||!bindings||bindings->struct_size!=sizeof(NRFrameBindings)||
       bindings->abi_version!=NR_PLAN_FRAME_BINDINGS_ABI_VERSION||
       !plan->finalized||!bindings->input||!bindings->output||
       !bindings->width||!bindings->height||bindings->width>plan->desc.max_width||
       bindings->height>plan->desc.max_height||
       (plan->static_input&&(bindings->width!=plan->static_width||bindings->height!=plan->static_height)))
        return hipErrorInvalidValue;
    hipError_t error=harvest_timing(plan);
    if(error!=hipSuccess)return error;
    // One in-flight frame per plan keeps the pinned parameter block immutable
    // until its H2D copy has been consumed. Callers may create multiple plans
    // for deeper pipelining; there is no hidden synchronization or allocation.
    if(plan->submitted&&hipEventQuery(plan->binding_consumed)!=hipSuccess)return hipErrorNotReady;
    *plan->pinned_bindings=*bindings;
    error=hipSuccess;
    if(input_ready)error=hipStreamWaitEvent(plan->stream,input_ready,0);
    if(error==hipSuccess)error=hipMemcpyAsync(plan->device_bindings,plan->pinned_bindings,
        sizeof(NRFrameBindings),hipMemcpyHostToDevice,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->binding_consumed,plan->stream);
    if(error==hipSuccess&&plan->static_input)
        error=hipMemcpyAsync(plan->static_input,bindings->input,size_t(plan->input_bytes),
            hipMemcpyDeviceToDevice,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->timing_start,plan->stream);
    if(error==hipSuccess)error=hipGraphLaunch(plan->executable,plan->stream);
    if(error==hipSuccess&&plan->static_output)
        error=hipMemcpyAsync(bindings->output,plan->static_output,size_t(plan->output_bytes),
            hipMemcpyDeviceToDevice,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->timing_stop,plan->stream);
    if(error==hipSuccess&&output_done)error=hipEventRecord(output_done,plan->stream);
    if(error==hipSuccess){plan->submitted=true;plan->timing_pending=true;++plan->performance.submitted_frames;}
    return error;
}

static hipError_t harvest_timing(NRPlan* plan) {
    if(!plan||!plan->timing_pending)return hipSuccess;
    hipError_t query=hipEventQuery(plan->timing_stop);
    if(query==hipErrorNotReady)return hipSuccess;
    if(query!=hipSuccess)return query;
    float milliseconds=0.f;
    hipError_t error=hipEventElapsedTime(&milliseconds,plan->timing_start,plan->timing_stop);
    if(error!=hipSuccess)return error;
    const double value=milliseconds;
    plan->performance.last_gpu_ms=value;plan->performance.total_gpu_ms+=value;
    if(!plan->performance.completed_frames||value<plan->performance.min_gpu_ms)
        plan->performance.min_gpu_ms=value;
    if(!plan->performance.completed_frames||value>plan->performance.max_gpu_ms)
        plan->performance.max_gpu_ms=value;
    ++plan->performance.completed_frames;plan->timing_pending=false;return hipSuccess;
}

static bool bindings_v3_are_valid(NRPlan* plan,const NRFrameBindingsV3* bindings) {
    if(!plan||!bindings||bindings->struct_size!=sizeof(NRFrameBindingsV3)||
       bindings->abi_version!=NR_PLAN_FRAME_BINDINGS_ABI_VERSION_V3||
       !plan->finalized||!plan->prepared||!bindings->current_color||
       !bindings->output_residual||bindings->render_width!=plan->shape.render_width||
       bindings->render_height!=plan->shape.render_height||
       bindings->output_width!=plan->shape.output_width||
       bindings->output_height!=plan->shape.output_height||
       bindings->valid_left!=plan->shape.valid_left||
       bindings->valid_top!=plan->shape.valid_top||
       bindings->valid_width!=plan->shape.valid_width||
       bindings->valid_height!=plan->shape.valid_height||
       !std::isfinite(bindings->jitter_x)||!std::isfinite(bindings->jitter_y)||
       !std::isfinite(bindings->motion_scale_x)||!std::isfinite(bindings->motion_scale_y)||
       !std::isfinite(bindings->pre_exposure)||bindings->pre_exposure<=0||
       !std::isfinite(bindings->exposure_scale)||bindings->exposure_scale<=0||
       ((bindings->flags&NR_FRAME_HAS_HISTORY)&&(!bindings->history||!bindings->next_history))||
       ((bindings->flags&NR_FRAME_HAS_MOTION)&&!bindings->motion)||
       ((bindings->flags&NR_FRAME_HAS_DEPTH)&&!bindings->depth)||
       ((bindings->flags&NR_FRAME_HAS_EXPOSURE)&&!bindings->exposure)||
       (bindings->flags&~uint32_t(NR_FRAME_HAS_HISTORY|NR_FRAME_HAS_MOTION|
                                  NR_FRAME_HAS_DEPTH|NR_FRAME_HAS_EXPOSURE))||
       (!bindings->reset&&!(bindings->flags&NR_FRAME_HAS_HISTORY))||
       (plan->temporal_contract_verified&&!bindings->next_history)||
       (!plan->temporal_contract_verified&&(!bindings->reset||bindings->flags)))return false;
    return true;
}

static hipError_t submit_v3(NRPlan* plan,const NRFrameBindingsV3* bindings,
    hipEvent_t input_ready,hipEvent_t output_done,const NRExternalSyncV3* sync) {
    if(!bindings_v3_are_valid(plan,bindings))return hipErrorInvalidValue;
    if(sync&&((input_ready||output_done)||sync->struct_size!=sizeof(NRExternalSyncV3)||
       sync->reserved||!sync->input_ready||!sync->output_done||
       !sync->input_value||!sync->output_value))return hipErrorInvalidValue;
    hipError_t error=harvest_timing(plan);
    if(error!=hipSuccess)return error;
    if(plan->submitted&&hipEventQuery(plan->binding_consumed)!=hipSuccess)return hipErrorNotReady;
    *plan->pinned_bindings_v3=*bindings;
    if(sync){
        hipExternalSemaphoreWaitParams wait{};wait.params.fence.value=sync->input_value;
        hipExternalSemaphore_t semaphore=sync->input_ready;
        error=hipWaitExternalSemaphoresAsync(&semaphore,&wait,1,plan->stream);
    }else if(input_ready)error=hipStreamWaitEvent(plan->stream,input_ready,0);
    if(error==hipSuccess)error=hipMemcpyAsync(plan->device_bindings_v3,
        plan->pinned_bindings_v3,sizeof(NRFrameBindingsV3),hipMemcpyHostToDevice,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->binding_consumed,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->timing_start,plan->stream);
    if(error==hipSuccess)error=hipGraphLaunch(plan->executable,plan->stream);
    if(error==hipSuccess)error=hipEventRecord(plan->timing_stop,plan->stream);
    if(error==hipSuccess&&sync){
        hipExternalSemaphoreSignalParams signal{};signal.params.fence.value=sync->output_value;
        hipExternalSemaphore_t semaphore=sync->output_done;
        error=hipSignalExternalSemaphoresAsync(&semaphore,&signal,1,plan->stream);
    }else if(error==hipSuccess&&output_done)error=hipEventRecord(output_done,plan->stream);
    if(error==hipSuccess){plan->submitted=true;plan->timing_pending=true;++plan->performance.submitted_frames;}
    return error;
}

NRPLAN_API hipError_t nrPlanSubmitV3(NRPlan* plan,const NRFrameBindingsV3* bindings,
    hipEvent_t input_ready,hipEvent_t output_done) {
    return submit_v3(plan,bindings,input_ready,output_done,nullptr);
}

NRPLAN_API hipError_t nrPlanSubmitExternalV3(NRPlan* plan,
    const NRFrameBindingsV3* bindings,const NRExternalSyncV3* sync) {
    if(!sync)return hipErrorInvalidValue;
    return submit_v3(plan,bindings,nullptr,nullptr,sync);
}

NRPLAN_API hipError_t nrPlanGetPerformanceStats(NRPlan* plan,
    NRPlanPerformanceStats* out_stats) {
    if(!plan||!out_stats)return hipErrorInvalidValue;
    hipError_t error=harvest_timing(plan);if(error!=hipSuccess)return error;
    *out_stats=plan->performance;return hipSuccess;
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
