#pragma once
#include <cstdint>

struct NRStandardStageLaunch {
    const void* raw;
    const void* expand;
    const void* contract;
    const void* mix;
    const void* ffn_scale;
    const void* a_index;
    const void* residual_index;
    const void* ffn_permutation;
    const void* inverse_permutation;
    void* grouped;
    void* seed;
    void* post;
    void* post_resident;
    const void* qkv;
    const void* qscale;
    const void* permutation;
    void* qkv_projection;
    void* q;
    void* k;
    void* v;
    const void* lut;
    const void* bias;
    void* value;
    const void* project;
    const void* residual_scale;
    void* output;
    void* next_resident;
    int windows;
    int width;
    int height;
    int origin_x;
    int origin_y;
    int stop_after_qkv;
    int force_publish_sync;
    void* stream;
};

#ifdef __cplusplus
extern "C" {
#endif
int nr_stage_c64_block_launch(const NRStandardStageLaunch* launch);
int nr_stage_c128_block_launch(const NRStandardStageLaunch* launch);
int nr_stage_c256_block_launch(const NRStandardStageLaunch* launch);
#ifdef __cplusplus
}
#endif
