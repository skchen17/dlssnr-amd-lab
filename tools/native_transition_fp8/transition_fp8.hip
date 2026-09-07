#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

using F8 = float __attribute__((ext_vector_type(8)));
using I2 = int __attribute__((ext_vector_type(2)));

__device__ static _Float16 hround_transition(float value) {
    return (_Float16)__float2half_rn(value);
}

__device__ static uint8_t encode_e4_transition(float value) {
    return __hip_cvt_float_to_fp8(value, __HIP_SATFINITE, __HIP_E4M3);
}

__device__ static _Float16 decode_e4_transition(uint8_t value) {
    return (_Float16)__half(__hip_cvt_fp8_to_halfraw(value, __HIP_E4M3));
}

__device__ static F8 transition_mma(I2 a, I2 b, F8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

template<int C> __device__ static uint64_t packed_offset(
    int y, int x, int channel, int width) {
    const int row = (y & 3) * 4 + (x & 3);
    const int col = channel & 31;
    const int mma = (col >> 4) * 2 + ((col & 15) >> 3);
    const int pairs[8] = {0, 2, 1, 3, 4, 6, 5, 7};
    const int element = (row >= 8) * 2 + (col & 1);
    const int operation = pairs[mma * 2 + element / 2];
    const int local = (channel / 32) * 512 +
        ((row & 7) * 4 + (col & 7) / 2) * 16 + operation * 2 + (element & 1);
    return (uint64_t(y / 4) * (width / 4) + x / 4) * (16 * C) + local;
}

template<int C> __device__ static uint64_t logical_window_offset(
    int y, int x, int channel, int width, int origin_x, int origin_y) {
    const uint64_t physical = packed_offset<C>(y, x, channel, width);
    const int cell_elements = 16 * C;
    const int cell_x = int((physical / cell_elements) % (width / 4));
    const int cell_y = int((physical / cell_elements) / (width / 4));
    const int local = int(physical % cell_elements);
    const int head = local / 512;
    const int within = local % 512;
    const int lane = within / 16;
    const int element = within % 16;
    const int translated_x = cell_x - origin_x / 4;
    const int translated_y = cell_y - origin_y / 4;
    const int windows_x = (width - origin_x + 7) / 8;
    const int window = translated_y / 2 * windows_x + translated_x / 2;
    const int cell = (translated_y % 2) * 2 + translated_x % 2;
    const int row = lane / 4 + (element % 8) / 4 * 8;
    const int col = head * 32 + lane % 4 * 2 + element % 4 / 2 * 8 +
        element / 8 * 16 + (element & 1);
    return (uint64_t(window) * 64 + cell * 16 + row) * C + col;
}

// One wave computes sixteen destination pixels by sixteen output channels.
// Pooling is evaluated at the original explicit FP16 rounding boundaries, then
// quantized exactly once at the recovered E4M3 projection boundary. The result
// is written directly to the packed layout consumed by the next Swin stage.
template<int C> __global__ void encoder_transition_fp8(
    const _Float16* source, const uint8_t* project, const int32_t* permutation,
    uint8_t* target, int source_width, int source_height, int origin_x, int origin_y) {
    constexpr int OUT = C * 2;
    const int lane = threadIdx.x;
    const int pixel_in_tile = lane & 15;
    const int half = lane >> 4;
    const int output_tiles = OUT / 16;
    const int task = blockIdx.x;
    const int output_tile = task % output_tiles;
    const int pixel_tile = task / output_tiles;
    const int target_width = source_width / 2;
    const int target_height = source_height / 2;
    const int pixel = pixel_tile * 16 + pixel_in_tile;
    if (pixel >= target_width * target_height) return;
    const int x = pixel % target_width;
    const int y = pixel / target_width;
    F8 accumulator = {};
    #pragma clang loop unroll(disable)
    for (int chunk = 0; chunk < C / 16; ++chunk) {
        I2 weight = {}, activation = {};
        for (int element = 0; element < 8; ++element) {
            const int k = chunk * 16 + half * 8 + element;
            const int input_channel = permutation[k];
            const _Float16 p00 = source[logical_window_offset<C>(
                y * 2, x * 2, input_channel, source_width, origin_x, origin_y)];
            const _Float16 p01 = source[logical_window_offset<C>(
                y * 2, x * 2 + 1, input_channel, source_width, origin_x, origin_y)];
            const _Float16 p10 = source[logical_window_offset<C>(
                y * 2 + 1, x * 2, input_channel, source_width, origin_x, origin_y)];
            const _Float16 p11 = source[logical_window_offset<C>(
                y * 2 + 1, x * 2 + 1, input_channel, source_width, origin_x, origin_y)];
            const _Float16 upper = hround_transition(float(p00) + float(p01));
            const _Float16 lower = hround_transition(float(p10) + float(p11));
            const _Float16 sum = hround_transition(float(upper) + float(lower));
            const uint8_t pooled = encode_e4_transition(
                float(hround_transition(float(sum) * .25f)));
            const int output_channel = output_tile * 16 + pixel_in_tile;
            weight[element / 4] |= unsigned(project[k * OUT + output_channel]) <<
                ((element & 3) * 8);
            activation[element / 4] |= unsigned(pooled) << ((element & 3) * 8);
        }
        accumulator = transition_mma(weight, activation, accumulator);
    }
    for (int element = 0; element < 8; ++element) {
        const int output_channel = output_tile * 16 + half * 8 + element;
        target[packed_offset<OUT>(y, x, output_channel, target_width)] =
            encode_e4_transition(float(hround_transition(accumulator[element])));
    }
}

// Decoder projection, 2x expansion and skip fusion remain in one bounded wave.
// A low-resolution result is never materialized: each accumulator element is
// written to its four high-resolution consumers with the original FP16 scale
// and add rounding, then converted at the next-stage E4M3 resident boundary.
template<int C> __global__ void decoder_transition_fp8(
    const uint8_t* source, const uint8_t* skip, const uint8_t* project,
    const _Float16* skip_scale, const int32_t* permutation, uint8_t* target,
    int target_width, int target_height) {
    constexpr int IN = C * 2;
    const int lane = threadIdx.x;
    const int pixel_in_tile = lane & 15;
    const int half = lane >> 4;
    const int output_tiles = C / 16;
    const int task = blockIdx.x;
    const int output_tile = task % output_tiles;
    const int pixel_tile = task / output_tiles;
    const int source_width = target_width / 2;
    const int source_height = target_height / 2;
    const int pixel = pixel_tile * 16 + pixel_in_tile;
    if (pixel >= source_width * source_height) return;
    const int source_x = pixel % source_width;
    const int source_y = pixel / source_width;
    F8 accumulator = {};
    #pragma clang loop unroll(disable)
    for (int chunk = 0; chunk < IN / 16; ++chunk) {
        I2 weight = {}, activation = {};
        for (int element = 0; element < 8; ++element) {
            const int k = chunk * 16 + half * 8 + element;
            const int input_channel = permutation[k];
            const int output_channel = output_tile * 16 + pixel_in_tile;
            weight[element / 4] |= unsigned(project[k * C + output_channel]) <<
                ((element & 3) * 8);
            activation[element / 4] |= unsigned(source[packed_offset<IN>(
                source_y, source_x, input_channel, source_width)]) << ((element & 3) * 8);
        }
        accumulator = transition_mma(weight, activation, accumulator);
    }
    for (int element = 0; element < 8; ++element) {
        const int output_channel = output_tile * 16 + half * 8 + element;
        const _Float16 projected = hround_transition(accumulator[element]);
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const int x = source_x * 2 + dx;
            const int y = source_y * 2 + dy;
            const _Float16 skip_value = decode_e4_transition(
                skip[packed_offset<C>(y, x, output_channel, target_width)]);
            const _Float16 scaled = hround_transition(
                float(skip_value) * float(skip_scale[output_channel]));
            const _Float16 result = hround_transition(float(projected) + float(scaled));
            target[packed_offset<C>(y, x, output_channel, target_width)] =
                encode_e4_transition(float(result));
        }
    }
}

extern "C" __declspec(dllexport) int nr_transition_encoder_fp8(
    const void* source, const void* source_resident, void* skip, const void* project,
    const void* permutation, void* target, int width, int height, int origin_x,
    int origin_y, int channels, void* stream) {
    if (!source || !source_resident || !skip || !project || !permutation || !target ||
        width <= 0 || height <= 0 || (width & 7) || (height & 7) ||
        (origin_x != 0 && origin_x != -4) || (origin_y != 0 && origin_y != -4) ||
        (channels != 32 && channels != 64 && channels != 128 && channels != 256))
        return int(hipErrorInvalidValue);
    const uint64_t source_bytes = uint64_t(width) * height * channels;
    hipError_t error = hipMemcpyAsync(skip, source_resident, source_bytes,
        hipMemcpyDeviceToDevice, reinterpret_cast<hipStream_t>(stream));
    if (error != hipSuccess) return int(error);
    const uint64_t pixels = uint64_t(width / 2) * (height / 2);
    const uint64_t blocks = ((pixels + 15) / 16) * (channels * 2 / 16);
#define LAUNCH_ENCODER(C) hipLaunchKernelGGL(encoder_transition_fp8<C>, \
    dim3(unsigned(blocks)), dim3(32), 0, reinterpret_cast<hipStream_t>(stream), \
    static_cast<const _Float16*>(source), static_cast<const uint8_t*>(project), \
    static_cast<const int32_t*>(permutation), static_cast<uint8_t*>(target), width, height, \
    origin_x, origin_y)
    if (channels == 32) { LAUNCH_ENCODER(32); }
    else if (channels == 64) { LAUNCH_ENCODER(64); }
    else if (channels == 128) { LAUNCH_ENCODER(128); }
    else if (channels == 256) { LAUNCH_ENCODER(256); }
    else return int(hipErrorInvalidValue);
#undef LAUNCH_ENCODER
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_transition_decoder_fp8(
    const void* source, const void* skip, const void* project, const void* skip_scale,
    const void* permutation, void* target, int width, int height, int channels,
    void* stream) {
    if (!source || !skip || !project || !skip_scale || !permutation || !target ||
        width <= 0 || height <= 0 || (width & 7) || (height & 7) ||
        (channels != 32 && channels != 64 && channels != 128 && channels != 256))
        return int(hipErrorInvalidValue);
    const uint64_t pixels = uint64_t(width / 2) * (height / 2);
    const uint64_t blocks = ((pixels + 15) / 16) * (channels / 16);
#define LAUNCH_DECODER(C) hipLaunchKernelGGL(decoder_transition_fp8<C>, \
    dim3(unsigned(blocks)), dim3(32), 0, reinterpret_cast<hipStream_t>(stream), \
    static_cast<const uint8_t*>(source), static_cast<const uint8_t*>(skip), \
    static_cast<const uint8_t*>(project), static_cast<const _Float16*>(skip_scale), \
    static_cast<const int32_t*>(permutation), static_cast<uint8_t*>(target), width, height)
    if (channels == 32) { LAUNCH_DECODER(32); }
    else if (channels == 64) { LAUNCH_DECODER(64); }
    else if (channels == 128) { LAUNCH_DECODER(128); }
    else if (channels == 256) { LAUNCH_DECODER(256); }
    else return int(hipErrorInvalidValue);
#undef LAUNCH_DECODER
    return int(hipGetLastError());
}
