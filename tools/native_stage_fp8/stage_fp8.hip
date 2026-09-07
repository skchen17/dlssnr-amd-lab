#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include "stage_fp8_api.h"

using F8 = float __attribute__((ext_vector_type(8)));
using I2 = int __attribute__((ext_vector_type(2)));

__device__ static _Float16 hround(float x) { return (_Float16)__float2half_rn(x); }

// gfx1201 currently executes the ROCm OCP conversion helper as packed FP16,
// while an inlined branch-heavy replacement is also miscompiled under a
// divergent wave mask. Build the exhaustive, RTX-validated FP16 -> E4M3 table
// once on the host and publish it before graph capture. The hot kernels then
// perform one uniform byte lookup from already-rounded FP16 bits.
__device__ uint8_t nr_stage_e4_from_fp16_lut[65536];

static uint32_t e4_host_round_shift(uint32_t value, uint32_t shift) {
    uint32_t base = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1u);
    return base + uint32_t(remainder > halfway ||
                           (remainder == halfway && (base & 1u)));
}

static uint8_t e4_host_from_half_bits(uint16_t bits) {
    uint8_t sign = uint8_t((bits >> 8) & 0x80);
    uint32_t exponent = (bits >> 10) & 31u;
    uint32_t mantissa = bits & 1023u;
    if (exponent == 31u && mantissa != 0u) return 0x7f;
    uint32_t code;
    if (exponent < 9u) {
        code = e4_host_round_shift(1024u + mantissa, 16u - exponent);
        if (code > 8u) code = 8u;
    } else {
        uint32_t rounded = e4_host_round_shift(mantissa, 7u);
        bool carry = rounded >= 8u;
        code = ((exponent - 8u + uint32_t(carry)) << 3) |
               (carry ? 0u : rounded);
        if (code > 0x7eu) code = 0x7eu;
    }
    return sign | uint8_t(code);
}

extern "C" __declspec(dllexport) int nr_stage_initialize_e4_lut(void** explicit_lut) {
    if (!explicit_lut || *explicit_lut) return int(hipErrorInvalidValue);
    std::array<uint8_t, 65536> table{};
    for (uint32_t bits = 0; bits < table.size(); ++bits)
        table[bits] = e4_host_from_half_bits(uint16_t(bits));
    hipError_t error = hipMalloc(explicit_lut, table.size());
    if (error == hipSuccess)
        error = hipMemcpy(*explicit_lut, table.data(), table.size(),
                          hipMemcpyHostToDevice);
    // Keep the legacy symbol initialized for kernels that have not yet been
    // migrated. The QKV publisher below deliberately uses the explicit plan-
    // owned pointer so its correctness no longer depends on symbol relocation.
    if (error == hipSuccess)
        error = hipMemcpyToSymbol(nr_stage_e4_from_fp16_lut, table.data(),
                                  table.size(), 0, hipMemcpyHostToDevice);
    if (error != hipSuccess && *explicit_lut) {
        hipFree(*explicit_lut);
        *explicit_lut = nullptr;
    }
    return int(error);
}

__device__ static uint8_t e4(float x) {
    __half rounded = __float2half_rn(x);
    return nr_stage_e4_from_fp16_lut[__half_as_ushort(rounded)];
}
__device__ static uint8_t e4_explicit(float x, const uint8_t* lut) {
    __half rounded = __float2half_rn(x);
    return lut[__half_as_ushort(rounded)];
}
__device__ static _Float16 e4h(uint8_t x) {
    return (_Float16)__half(__hip_cvt_fp8_to_halfraw(x,__HIP_E4M3));
}
__device__ static _Float16 cubic(_Float16 hx) {
    float x=float(hx),v=fminf(4.f,fmaxf(-4.f,x));
    _Float16 a=hround(-0.055908203125f*fabsf(v)+0.447265625f);
    _Float16 b=hround(v*float(a)+0.89453125f);
    return e4h(e4(float(hround(x*float(b)))));
}

__device__ static F8 fp8_mma(I2 a, I2 b, F8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

template<int C> __device__ static int64_t window_source_index(
    int window,int physical,int width,int height,int ox,int oy) {
    int cell_elements=16*C,cell=physical/cell_elements,local=physical%cell_elements;
    int gx=(width-ox+7)/8;
    int x=(window%gx*8+ox)/4+cell%2;
    int y=(window/gx*8+oy)/4+cell/2;
    if(x<0||x>=width/4||y<0||y>=height/4)return -1;
    return (int64_t(y)*(width/4)+x)*cell_elements+local;
}

// Bounded grouped FFN. A workgroup owns (window, head, 16-token tile), keeps
// only the current 128-channel group in LDS, and publishes an E4M3 grouped
// operand plus an FP16 residual seed for the independent cross-head mix.
template<int C> __global__ void stage_ffn_group_fp8(
    const uint8_t* raw,const uint8_t* expand,const uint8_t* contract,
    const _Float16* scale,const int32_t* ai,const int32_t* ri,
    const int32_t* permutation,const int32_t* inverse_permutation,
    uint8_t* grouped,_Float16* seed,int windows,int width,int height,int ox,int oy) {
    __shared__ _Float16 hidden[128][16];
    constexpr int HEADS=C/32;
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,head=task%HEADS,tile=task/HEADS;
    int window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    #pragma clang loop unroll(disable)
    for(int m=0;m<8;++m){
        F8 acc={};
        #pragma clang loop unroll(disable)
        for(int chunk=0;chunk<C/16;++chunk){
            I2 weight={},activation={};
            for(int e=0;e<8;++e){
                int kk=chunk*16+half*8+e,hidden_col=head*128+m*16+token;
                weight[e/4]|=unsigned(expand[kk*(4*C)+hidden_col])<<((e%4)*8);
                int64_t source=window_source_index<C>(window,ai[row*C+kk],width,height,ox,oy);
                activation[e/4]|=unsigned(source<0?0:raw[source])<<((e%4)*8);
            }
            acc=fp8_mma(weight,activation,acc);
        }
        for(int e=0;e<8;++e)hidden[m*16+half*8+e][token]=cubic(hround(acc[e]));
    }
    __syncthreads();
    #pragma clang loop unroll(disable)
    for(int m=0;m<2;++m){
        F8 acc={};
        for(int part=0;part<4;++part)for(int chunk=0;chunk<2;++chunk){
            I2 weight={},activation={};
            for(int e=0;e<8;++e){
                int kk=chunk*16+half*8+e,k=part*32+kk;
                weight[e/4]|=unsigned(contract[(head*128+k)*32+m*16+token])<<((e%4)*8);
                activation[e/4]|=unsigned(e4(float(hidden[part*32+permutation[kk]][token])))<<((e%4)*8);
            }
            acc=fp8_mma(weight,activation,acc);
        }
        for(int e=0;e<8;++e){
            int local=m*16+half*8+e,channel=head*32+local;
            int mix_input=head*32+inverse_permutation[local];
            size_t logical=(size_t(window)*64+row)*C;
            grouped[logical+mix_input]=e4(float(hround(acc[e])));
            int64_t source=window_source_index<C>(window,ri[row*C+channel],width,height,ox,oy);
            _Float16 residual=source<0?_Float16(0):e4h(raw[source]);
            seed[logical+channel]=hround(float(residual)*float(scale[channel]));
        }
    }
}

// Convert the current origin's logical window output directly back to the
// persistent packed-image resident layout. No logical FP16 image is created.
template<int C> __global__ void stage_scatter_fp8(
    const uint8_t* logical,uint8_t* packed,int width,int height,int ox,int oy,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    size_t cell_elements=16*C,cell_index=i/cell_elements;
    int x=int(cell_index%(width/4)),y=int(cell_index/(width/4));
    int local=int(i%cell_elements),head=local/512,within=local%512;
    int lane=within/16,e=within%16,tx=x-ox/4,ty=y-oy/4,gx=(width-ox+7)/8;
    int window=ty/2*gx+tx/2,cell=(ty%2)*2+tx%2;
    int row=lane/4+e%8/4*8,col=head*32+lane%4*2+e%4/2*8+e/8*16+(e&1);
    packed[i]=logical[(size_t(window)*64+cell*16+row)*C+col];
}

// Cross-head CxC mix and residual add. It emits the FP16 post-FFN residual view
// and the byte-identical E4M3 view consumed by the QKV stage below.
template<int C> __global__ void stage_ffn_mix_fp8(
    const uint8_t* grouped,const uint8_t* mix,const _Float16* seed,
    _Float16* post,int windows) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,out_tile=task%(C/16),tile=task/(C/16);
    int window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    F8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<C/16;++chunk){
        I2 weight={},activation={};
        for(int e=0;e<8;++e){
            int k=chunk*16+half*8+e;
            weight[e/4]|=unsigned(mix[k*C+out_tile*16+token])<<((e%4)*8);
            activation[e/4]|=unsigned(grouped[(size_t(window)*64+row)*C+k])<<((e%4)*8);
        }
        acc=fp8_mma(weight,activation,acc);
    }
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        size_t index=(size_t(window)*64+row)*C+channel;
        _Float16 result=hround(float(hround(acc[e]))+float(seed[index]));
        post[index]=result;
    }
}

// gfx1201 miscompiles FP8 byte publication when it is embedded in the
// divergent WMMA producer above. Publish from the materialized semantic FP16
// boundary in a uniform kernel. Four codes are committed by one 32-bit store.
__global__ void stage_pack_e4x4_from_fp16(const uint16_t* input,uint32_t* output,
    const uint8_t* lut,size_t scalar_count) {
    size_t word=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    size_t base=word*4;if(base>=scalar_count)return;
    uint32_t packed=0;
    #pragma clang loop unroll(full)
    for(int e=0;e<4;++e){
        size_t index=base+size_t(e);
        if(index<scalar_count)packed|=uint32_t(lut[input[index]])<<(e*8);
    }
    output[word]=packed;
}

extern "C" __declspec(dllexport) int nr_stage_debug_pack_e4x4_from_fp16(
    const void* input,void* output,const void* lut,size_t scalar_count,void* stream) {
    if(!input||!output||!lut||!scalar_count)return int(hipErrorInvalidValue);
    size_t words=(scalar_count+3)/4;
    hipLaunchKernelGGL(stage_pack_e4x4_from_fp16,
        dim3(unsigned((words+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(input),
        static_cast<uint32_t*>(output),static_cast<const uint8_t*>(lut),scalar_count);
    return int(hipGetLastError());
}

// Diagnostic-safe QKV boundary for the standard C32-C256 families. Projection
// is deliberately materialized as FP16 in a fixed arena slot, so its matrix
// result and the following normalization can be validated independently. Each
// workgroup owns one (window, 16-token tile, 16-output tile); there is no
// head-serial loop and no LDS allocation.
template<int C> __global__ void stage_qkv_project_fp8(
    const uint8_t* post, const uint8_t* qkv_column_major,
    const int32_t* permutation, _Float16* projection, int windows) {
    int lane = threadIdx.x, token = lane % 16, half = lane / 16;
    int output_tiles = 3 * C / 16;
    int task = blockIdx.x, out_tile = task % output_tiles;
    int tile = task / output_tiles, window = tile / 4;
    int row = tile % 4 * 16 + token;
    if (window >= windows) return;
    F8 acc = {};
    #pragma clang loop unroll(disable)
    for (int chunk = 0; chunk < C / 16; ++chunk) {
        I2 packed_weight = {}, packed_activation = {};
        for (int e = 0; e < 8; ++e) {
            int source = chunk * 16 + half * 8 + e;
            packed_weight[e / 4] |= unsigned(
                qkv_column_major[(out_tile * 16 + token) * C + source]) << ((e % 4) * 8);
            packed_activation[e / 4] |= unsigned(
                post[(size_t(window) * 64 + row) * C + permutation[source]]) << ((e % 4) * 8);
        }
        acc = fp8_mma(packed_weight, packed_activation, acc);
    }
    for (int e = 0; e < 8; ++e) {
        int out_channel = out_tile * 16 + half * 8 + e;
        projection[(size_t(window) * 64 + row) * (3 * C) + out_channel] = hround(acc[e]);
    }
}

// One wave owns one (window, head, token, Q/K/V-part) vector. Keep the
// pairwise-half reduction and the resident-byte publication in separate
// kernels: gfx1201 has returned decoded FP16 storage when a LUT byte publish is
// combined with the divergent shuffle path, despite emitting a b8 store in
// ISA. The projection arena is the bounded FP16 handoff and needs no extra
// workspace.
template<int C> __global__ void stage_qkv_norm_fp16_inplace(
    _Float16* projection, const _Float16* qscale, size_t vectors) {
    size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    size_t index = warp / 3;
    int part = int(warp % 3), lane = threadIdx.x % 32;
    if (index >= vectors) return;
    constexpr int HEADS = C / 32;
    int head = int(index / 64 % HEADS), token = int(index % 64);
    size_t window = index / (64 * HEADS);
    _Float16 h = projection[(window * 64 + token) * (3 * C) +
                            part * C + head * 32 + lane];
    _Float16 result = h;
    if (part != 2) {
        float value = float(h), sum = float(hround(value * value));
        for (int offset = 1; offset < 32; offset *= 2) {
            float other = __shfl_down(sum, offset, 32);
            if ((lane % (2 * offset)) == 0) sum = float(hround(sum + other));
        }
        sum = __shfl(sum, 0, 32);
        result = hround(value * float(hround(
            rsqrtf(fmaxf(sum, 6.198883056640625e-5f)))));
        if (part == 0) result = hround(float(result) * float(qscale[head]));
    }
    projection[(window * 64 + token) * (3 * C) +
               part * C + head * 32 + lane] = result;
}

// Fixed-target E4M3 publisher. Keep Q/K/V as three launches during correctness
// bring-up: the prior combined kernel selected one of three unrelated output
// pointers per lane and gfx1201 returned the normalized FP16 storage bytes even
// though its ISA contained the expected LUT load and b8 store. A compile-time
// part removes that final dynamic pointer selection without changing layout or
// quantization semantics.
template<int C, int PART> __global__ void stage_qkv_pack_e4x4_part_from_fp16(
    const uint16_t* projection_bits, uint32_t* target_words, const uint8_t* lut,
    size_t scalar_count) {
    static_assert(PART >= 0 && PART < 3, "QKV part must be fixed at compile time");
    size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t first_scalar = word * 4;
    if (first_scalar >= scalar_count) return;
    constexpr int HEADS = C / 32;
    uint32_t packed = 0;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        size_t scalar = first_scalar + item;
        uint8_t code = 0;
        if (scalar < scalar_count) {
            int lane = int(scalar % 32);
            size_t index = scalar / 32;
            int head = int(index / 64 % HEADS), token = int(index % 64);
            size_t window = index / (64 * HEADS);
            uint16_t bits = projection_bits[(window * 64 + token) * (3 * C) +
                                             PART * C + head * 32 + lane];
            code = lut[bits];
        }
        packed |= uint32_t(code) << (item * 8);
    }
    target_words[word] = packed;
}

// Stage-specific compact publisher. Projection and normalization keep their
// existing FP16 boundary; this kernel only combines the three independent,
// uniform Q/K/V publication passes. Every store remains a 32-bit transaction,
// avoiding the divergent byte-store pattern rejected during earlier gfx1201
// bring-up.
template<int C> __global__ void stage_qkv_pack_e4x4_all_from_fp16(
    const uint16_t* projection_bits, uint32_t* q_words, uint32_t* k_words,
    uint32_t* v_words, const uint8_t* lut, size_t scalar_count) {
    size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t first_scalar = word * 4;
    if (first_scalar >= scalar_count) return;
    constexpr int HEADS = C / 32;
    uint32_t packed[3] = {};
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        size_t scalar = first_scalar + item;
        if (scalar >= scalar_count) continue;
        int lane = int(scalar % 32);
        size_t index = scalar / 32;
        int head = int(index / 64 % HEADS), token = int(index % 64);
        size_t window = index / (64 * HEADS);
        #pragma unroll
        for (int part = 0; part < 3; ++part) {
            uint16_t bits = projection_bits[(window * 64 + token) * (3 * C) +
                                             part * C + head * 32 + lane];
            packed[part] |= uint32_t(lut[bits]) << (item * 8);
        }
    }
    q_words[word] = packed[0];
    k_words[word] = packed[1];
    v_words[word] = packed[2];
}

// C512 retains the prior bounded fused implementation until its dedicated
// split descriptor gains an independently validated projection workspace.
template<int C> __global__ void stage_qkv_norm_fused_fp8(
    const uint8_t* post, const uint8_t* qkv_column_major,
    const _Float16* qscale, const int32_t* permutation,
    uint8_t* q, uint8_t* k, uint8_t* v, int windows) {
    __shared__ _Float16 projection[3][32][16];
    constexpr int HEADS = C / 32;
    int lane = threadIdx.x, token = lane % 16, half = lane / 16;
    int task = blockIdx.x, head = task % HEADS, tile = task / HEADS;
    int window = tile / 4, row = tile % 4 * 16 + token;
    if (window >= windows) return;
    #pragma clang loop unroll(disable)
    for (int part = 0; part < 3; ++part) {
      #pragma clang loop unroll(disable)
      for (int out_half = 0; out_half < 2; ++out_half) {
        F8 acc = {};
        #pragma clang loop unroll(disable)
        for (int chunk = 0; chunk < C / 16; ++chunk) {
            I2 packed_weight = {}, packed_activation = {};
            for (int e = 0; e < 8; ++e) {
                int source = chunk * 16 + half * 8 + e;
                int out_channel = out_half * 16 + token;
                int global_out = part * C + head * 32 + out_channel;
                packed_weight[e / 4] |= unsigned(qkv_column_major[global_out * C + source]) << ((e % 4) * 8);
                packed_activation[e / 4] |= unsigned(post[(window * 64 + row) * C + permutation[source]]) << ((e % 4) * 8);
            }
            acc = fp8_mma(packed_weight, packed_activation, acc);
        }
        for (int e = 0; e < 8; ++e)
            projection[part][out_half * 16 + half * 8 + e][token] = hround(acc[e]);
      }
    }
    __syncthreads();
    const int channel = lane;
    for (int local_row = 0; local_row < 16; ++local_row) {
        size_t index = ((size_t(window) * HEADS + head) * 64 + tile % 4 * 16 + local_row) * 32 + channel;
        _Float16 value_v = projection[2][channel][local_row];
        v[index] = e4(float(value_v));
        for (int part = 0; part < 2; ++part) {
            float value = float(projection[part][channel][local_row]);
            float sum = float(hround(value * value));
            for (int offset = 1; offset < 32; offset *= 2) {
                float other = __shfl_down(sum, offset, 32);
                if ((lane % (2 * offset)) == 0) sum = float(hround(sum + other));
            }
            sum = __shfl(sum, 0, 32);
            _Float16 normalized = hround(value * float(hround(rsqrtf(fmaxf(sum, 6.198883056640625e-5f)))));
            if (part == 0) normalized = hround(float(normalized) * float(qscale[head]));
            (part == 0 ? q : k)[index] = e4(float(normalized));
        }
    }
}

__device__ __forceinline__ uint16_t half_bits(_Float16 value) {
    union Bits { _Float16 half; uint16_t bits; } converted;
    converted.half=value;return converted.bits;
}

// One bounded workgroup owns (window, head, 16-token tile). It keeps the same
// WMMA accumulation and FP16 normalization order as the split path, but stores
// Q/K/V through uniform LUT-backed 32-bit transactions. This is an opt-in
// candidate; the materialized projection path remains the default rollback.
template<int C,bool POST_FP16> __global__ void stage_qkv_full_fused_lut_fp8(
    const void* post_raw,const uint8_t* qkv_column_major,const _Float16* qscale,
    const int32_t* permutation,uint32_t* q_words,uint32_t* k_words,
    uint32_t* v_words,const uint8_t* lut,int windows) {
    __shared__ _Float16 projection[3][32][16];
    constexpr int HEADS=C/32;
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,head=task%HEADS,tile=task/HEADS;
    int window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    #pragma clang loop unroll(disable)
    for(int part=0;part<3;++part){
      #pragma clang loop unroll(disable)
      for(int out_half=0;out_half<2;++out_half){
        F8 acc={};
        #pragma clang loop unroll(disable)
        for(int chunk=0;chunk<C/16;++chunk){
            I2 packed_weight={},packed_activation={};
            for(int e=0;e<8;++e){
                int source=chunk*16+half*8+e;
                int global_out=part*C+head*32+out_half*16+token;
                packed_weight[e/4]|=unsigned(
                    qkv_column_major[global_out*C+source])<<((e%4)*8);
                size_t post_index=(size_t(window)*64+row)*C+permutation[source];
                uint8_t activation;
                if constexpr(POST_FP16)
                    activation=lut[half_bits(static_cast<const _Float16*>(post_raw)[post_index])];
                else activation=static_cast<const uint8_t*>(post_raw)[post_index];
                packed_activation[e/4]|=unsigned(activation)<<((e%4)*8);
            }
            acc=fp8_mma(packed_weight,packed_activation,acc);
        }
        for(int e=0;e<8;++e)
            projection[part][out_half*16+half*8+e][token]=hround(acc[e]);
      }
    }
    __syncthreads();
    uint32_t* targets[3]={q_words,k_words,v_words};
    for(int local_row=0;local_row<16;++local_row){
        size_t vector=((size_t(window)*HEADS+head)*64+tile%4*16+local_row)*32;
        #pragma unroll
        for(int part=0;part<3;++part){
            _Float16 h=projection[part][lane][local_row],result=h;
            if(part!=2){
                float value=float(h),sum=float(hround(value*value));
                for(int offset=1;offset<32;offset*=2){
                    float other=__shfl_down(sum,offset,32);
                    if((lane%(2*offset))==0)sum=float(hround(sum+other));
                }
                sum=__shfl(sum,0,32);
                result=hround(value*float(hround(
                    rsqrtf(fmaxf(sum,6.198883056640625e-5f)))));
                if(part==0)result=hround(float(result)*float(qscale[head]));
            }
            uint32_t code=lut[half_bits(result)],base=uint32_t(lane&~3);
            uint32_t packed=__shfl(code,base,32)|(__shfl(code,base+1,32)<<8)|
                (__shfl(code,base+2,32)<<16)|(__shfl(code,base+3,32)<<24);
            if((lane&3)==0)targets[part][vector/4+lane/4]=packed;
        }
    }
}

// Query-tile attention: 2 KiB FP16 scores + 1 KiB E4M3 probabilities.
// Q/K/V and output remain resident bytes; no [64,64] global tensor is created.
template<int C> __global__ void stage_attention_fp8(
    const uint8_t* q, const uint8_t* k, const uint8_t* v,
    const _Float16* position_bias, uint8_t* value, int windows) {
    __shared__ _Float16 scores[16][64];
    __shared__ uint8_t probability[16][64];
    constexpr int HEADS = C / 32;
    int lane = threadIdx.x, query = lane % 16, half = lane / 16;
    int task = blockIdx.x, head = task % HEADS, tile = task / HEADS;
    int window = tile / 4, query_tile = tile % 4, row = query_tile * 16 + query;
    if (window >= windows) return;
    size_t base = (size_t(window) * HEADS + head) * 64;
    #pragma clang loop unroll(disable)
    for (int key_tile = 0; key_tile < 4; ++key_tile) {
        F8 acc = {};
        for (int chunk = 0; chunk < 2; ++chunk) {
            I2 packed_k = {}, packed_q = {};
            for (int e = 0; e < 8; ++e) {
                int channel = chunk * 16 + half * 8 + e;
                int key = key_tile * 16 + query;
                packed_k[e / 4] |= unsigned(k[(base + key) * 32 + channel]) << ((e % 4) * 8);
                packed_q[e / 4] |= unsigned(q[(base + row) * 32 + channel]) << ((e % 4) * 8);
            }
            acc = fp8_mma(packed_k, packed_q, acc);
        }
        for (int e = 0; e < 8; ++e) {
            int key = key_tile * 16 + half * 8 + e;
            scores[query][key] = hround(float(hround(acc[e])) +
                float(position_bias[(head * 64 + row) * 64 + key]));
        }
    }
    __syncthreads();
    for (int local = 0; local < 16; ++local) {
        float x0 = float(scores[local][lane]);
        float x1 = float(scores[local][lane + 32]);
        float maximum = fmaxf(x0, x1);
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
        maximum = __shfl(maximum, 0, 32);
        float e0 = expf(x0 - maximum), e1 = expf(x1 - maximum), sum = e0 + e1;
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down(sum, offset, 32);
        sum = __shfl(sum, 0, 32);
        probability[local][lane] = e4(float(hround(e0 / sum)));
        probability[local][lane + 32] = e4(float(hround(e1 / sum)));
    }
    __syncthreads();
    #pragma clang loop unroll(disable)
    for (int out_half = 0; out_half < 2; ++out_half) {
        F8 acc = {};
        for (int key_chunk = 0; key_chunk < 4; ++key_chunk) {
            I2 packed_v = {}, packed_p = {};
            for (int e = 0; e < 8; ++e) {
                int key = key_chunk * 16 + half * 8 + e;
                int out_channel = out_half * 16 + query;
                packed_v[e / 4] |= unsigned(v[(base + key) * 32 + out_channel]) << ((e % 4) * 8);
                packed_p[e / 4] |= unsigned(probability[query][key]) << ((e % 4) * 8);
            }
            acc = fp8_mma(packed_v, packed_p, acc);
        }
        for (int e = 0; e < 8; ++e) {
            int channel = out_half * 16 + half * 8 + e;
            value[(base + row) * 32 + channel] = e4(float(hround(acc[e])));
        }
    }
}

// Cross-head projection plus FP16 residual. Output publishes both the semantic
// FP16 residual view and the one-byte E4M3 view consumed by the next block.
template<int C> __global__ void stage_project_fp8(
    const _Float16* post, const uint8_t* value, const uint8_t* project,
    const _Float16* scale, const int32_t* permutation,
    _Float16* output, uint8_t* next_resident, int windows) {
    constexpr int HEADS = C / 32;
    int lane = threadIdx.x, token = lane % 16, half = lane / 16;
    int task = blockIdx.x, out_tile = task % (C / 16), tile = task / (C / 16);
    int window = tile / 4, row = tile % 4 * 16 + token;
    if (window >= windows) return;
    F8 acc = {};
    #pragma clang loop unroll(disable)
    for (int chunk = 0; chunk < C / 16; ++chunk) {
        I2 packed_weight = {}, packed_value = {};
        for (int e = 0; e < 8; ++e) {
            int source = chunk * 16 + half * 8 + e;
            int logical = permutation[source], head = logical / 32, local = logical % 32;
            packed_weight[e / 4] |= unsigned(project[source * C + out_tile * 16 + token]) << ((e % 4) * 8);
            packed_value[e / 4] |= unsigned(value[((size_t(window) * HEADS + head) * 64 + row) * 32 + local]) << ((e % 4) * 8);
        }
        acc = fp8_mma(packed_weight, packed_value, acc);
    }
    for (int e = 0; e < 8; ++e) {
        int channel = out_tile * 16 + half * 8 + e;
        size_t index = (size_t(window) * 64 + row) * C + channel;
        _Float16 seed = hround(float(post[index]) * float(scale[channel]));
        _Float16 result = hround(float(hround(acc[e])) + float(seed));
        output[index] = result;
        next_resident[index] = e4(float(result));
    }
}

// C512 is a four-record split block rather than the grouped-FFN record used by
// C64/C128/C256.  Keep its two full 512x512 projections as bounded WMMA stages
// and only retain one 64->256 group in LDS at a time.
__global__ void stage_c512_preproject_fp8(
    const uint8_t* raw,const uint8_t* weight,const int32_t* ai,
    uint8_t* projected,int windows,int width,int height,int ox,int oy) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,out_tile=task%32,tile=task/32;
    int window=tile/4,row=tile%4*16+token;if(window>=windows)return;
    F8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<32;++chunk){
        I2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int k=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(weight[k*512+out_tile*16+token])<<((e%4)*8);
            int64_t source=window_source_index<512>(window,ai[row*512+k],width,height,ox,oy);
            packed_activation[e/4]|=unsigned(source<0?0:raw[source])<<((e%4)*8);
        }
        acc=fp8_mma(packed_weight,packed_activation,acc);
    }
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        projected[(size_t(window)*64+row)*512+channel]=e4(float(hround(acc[e])));
    }
}

__global__ void stage_c512_group_fp8(
    const uint8_t* projected,const uint8_t* expand,const uint8_t* contract,
    const int32_t* perm64,const int32_t* perm256,uint8_t* grouped,int windows) {
    __shared__ _Float16 hidden[256][16];
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,head=task%8,tile=task/8;
    int window=tile/4,row=tile%4*16+token;if(window>=windows)return;
    #pragma clang loop unroll(disable)
    for(int out_tile=0;out_tile<16;++out_tile){
        F8 acc={};
        for(int chunk=0;chunk<4;++chunk){
            I2 packed_weight={},packed_activation={};
            for(int e=0;e<8;++e){
                int k=chunk*16+half*8+e;
                packed_weight[e/4]|=unsigned(expand[(head*64+k)*256+out_tile*16+token])<<((e%4)*8);
                packed_activation[e/4]|=unsigned(projected[(size_t(window)*64+row)*512+
                    head*64+perm64[k]])<<((e%4)*8);
            }
            acc=fp8_mma(packed_weight,packed_activation,acc);
        }
        for(int e=0;e<8;++e)
            hidden[out_tile*16+half*8+e][token]=cubic(hround(acc[e]));
    }
    __syncthreads();
    #pragma clang loop unroll(disable)
    for(int out_tile=0;out_tile<4;++out_tile){
        F8 acc={};
        #pragma clang loop unroll(disable)
        for(int chunk=0;chunk<16;++chunk){
            I2 packed_weight={},packed_activation={};
            for(int e=0;e<8;++e){
                int k=chunk*16+half*8+e;
                packed_weight[e/4]|=unsigned(contract[(head*256+k)*64+out_tile*16+token])<<((e%4)*8);
                packed_activation[e/4]|=unsigned(e4(float(hidden[perm256[k]][token])))<<((e%4)*8);
            }
            acc=fp8_mma(packed_weight,packed_activation,acc);
        }
        for(int e=0;e<8;++e){
            int channel=head*64+out_tile*16+half*8+e;
            grouped[(size_t(window)*64+row)*512+channel]=e4(float(hround(acc[e])));
        }
    }
}

__global__ void stage_c512_ffn_project_fp8(
    const uint8_t* raw,const uint8_t* grouped,const uint8_t* weight,
    const _Float16* scale,const int32_t* ri,const int32_t* permutation,
    _Float16* post,uint8_t* resident,int windows,int width,int height,int ox,int oy) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,out_tile=task%32,tile=task/32;
    int window=tile/4,row=tile%4*16+token;if(window>=windows)return;
    F8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<32;++chunk){
        I2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int k=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(weight[k*512+out_tile*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(grouped[(size_t(window)*64+row)*512+
                permutation[k]])<<((e%4)*8);
        }
        acc=fp8_mma(packed_weight,packed_activation,acc);
    }
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        int64_t source=window_source_index<512>(window,ri[row*512+channel],width,height,ox,oy);
        _Float16 residual=source<0?_Float16(0):e4h(raw[source]);
        _Float16 seed=hround(float(residual)*float(scale[channel]));
        _Float16 result=hround(float(hround(acc[e]))+float(seed));
        size_t index=(size_t(window)*64+row)*512+channel;
        post[index]=result;resident[index]=e4(float(result));
    }
}

// C32 has one 128-channel FFN group and no cross-head CxC mix. One workgroup
// therefore completes the full FFN and publishes both semantic FP16 and
// resident E4M3 outputs without an intermediate global grouped tensor.
__global__ void stage_c32_ffn_fp8(
    const uint8_t* raw,const uint8_t* expand,const uint8_t* contract,
    const _Float16* scale,const int32_t* ai,const int32_t* ri,
    const int32_t* permutation,_Float16* post,uint8_t* resident,
    int windows,int width,int height,int ox,int oy) {
    __shared__ _Float16 hidden[128][16];
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,half=lane/16;
    int tile=blockIdx.x,window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    F8 expand_acc={};
    for(int chunk=0;chunk<2;++chunk){
        I2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int k=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(expand[k*128+wave*16+token])<<((e%4)*8);
            int64_t source=window_source_index<32>(window,ai[row*32+k],width,height,ox,oy);
            packed_activation[e/4]|=unsigned(source<0?0:raw[source])<<((e%4)*8);
        }
        expand_acc=fp8_mma(packed_weight,packed_activation,expand_acc);
    }
    for(int e=0;e<8;++e)
        hidden[wave*16+half*8+e][token]=cubic(hround(expand_acc[e]));
    __syncthreads();
    if(wave>=2)return;
    F8 contract_acc={};
    for(int chunk=0;chunk<8;++chunk){
        I2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int k=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(contract[k*32+wave*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(e4(float(hidden[permutation[k%32]+
                k/32*32][token])))<<((e%4)*8);
        }
        contract_acc=fp8_mma(packed_weight,packed_activation,contract_acc);
    }
    for(int e=0;e<8;++e){
        int channel=wave*16+half*8+e;
        int64_t source=window_source_index<32>(window,ri[row*32+channel],width,height,ox,oy);
        _Float16 residual=source<0?_Float16(0):e4h(raw[source]);
        _Float16 seed=hround(float(residual)*float(scale[channel]));
        _Float16 result=hround(float(hround(contract_acc[e]))+float(seed));
        size_t index=(size_t(window)*64+row)*32+channel;
        post[index]=result;resident[index]=e4(float(result));
    }
}

// Output Head already arrives as one resident 8x8-window stream.  Re-gathering
// it through a synthetic image would add a full-frame materialization and can
// change padded-lane semantics.  This edge-specific entry keeps the exact C32
// FFN math above but interprets authored a/residual indices within each window.
__global__ void stage_head_c32_ffn_fp8(
    const uint8_t* raw,const uint8_t* expand,const uint8_t* contract,
    const _Float16* scale,const int32_t* ai,const int32_t* ri,
    const int32_t* permutation,_Float16* post,uint8_t* resident,int windows) {
    __shared__ _Float16 hidden[128][16];
    const int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,half=lane/16;
    const int tile=blockIdx.x,window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    F8 expand_acc={};
    for(int chunk=0;chunk<2;++chunk){
        I2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            const int k=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(expand[k*128+wave*16+token])<<((e%4)*8);
            const int32_t source=ai[row*32+k];
            packed_activation[e/4]|=unsigned(raw[(size_t(window)*64*32)+source])<<((e%4)*8);
        }
        expand_acc=fp8_mma(packed_weight,packed_activation,expand_acc);
    }
    for(int e=0;e<8;++e)
        hidden[wave*16+half*8+e][token]=cubic(hround(expand_acc[e]));
    __syncthreads();
    if(wave>=2)return;
    F8 contract_acc={};
    for(int chunk=0;chunk<8;++chunk){
        I2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            const int k=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(contract[k*32+wave*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(e4(float(hidden[permutation[k%32]+
                k/32*32][token])))<<((e%4)*8);
        }
        contract_acc=fp8_mma(packed_weight,packed_activation,contract_acc);
    }
    for(int e=0;e<8;++e){
        const int channel=wave*16+half*8+e;
        const int32_t source=ri[row*32+channel];
        const _Float16 residual=e4h(raw[(size_t(window)*64*32)+source]);
        const _Float16 seed=hround(float(residual)*float(scale[channel]));
        const _Float16 result=hround(float(hround(contract_acc[e]))+float(seed));
        const size_t index=(size_t(window)*64+row)*32+channel;
        post[index]=result;resident[index]=e4(float(result));
    }
}

static int c32_ffn_entry(const void* raw,const void* expand,const void* contract,
    const void* scale,const void* ai,const void* ri,const void* permutation,
    void* post,void* resident,int windows,int width,int height,int ox,int oy,void* stream) {
    if(!raw||!expand||!contract||!scale||!ai||!ri||!permutation||!post||!resident||
       windows<1||windows>262144||width<=0||height<=0||width%4||height%4||
       (ox!=0&&ox!=-4)||(oy!=0&&oy!=-4))return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(stage_c32_ffn_fp8,dim3(windows*4),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(raw),
        static_cast<const uint8_t*>(expand),static_cast<const uint8_t*>(contract),
        static_cast<const _Float16*>(scale),static_cast<const int32_t*>(ai),
        static_cast<const int32_t*>(ri),static_cast<const int32_t*>(permutation),
        static_cast<_Float16*>(post),static_cast<uint8_t*>(resident),windows,width,
        height,ox,oy);
    return int(hipGetLastError());
}

static int c512_ffn_entry(const void* raw,const void* preproject,const void* expand,
    const void* contract,const void* ffn_project,const void* ffn_scale,const void* ai,
    const void* ri,const void* perm64,const void* perm256,const void* permutation,
    void* projected,void* grouped,void* post,void* resident,int windows,int width,
    int height,int ox,int oy,void* stream) {
    if(!raw||!preproject||!expand||!contract||!ffn_project||!ffn_scale||!ai||!ri||
       !perm64||!perm256||!permutation||!projected||!grouped||!post||!resident||
       windows<1||windows>262144||width<=0||height<=0||width%4||height%4||
       (ox!=0&&ox!=-4)||(oy!=0&&oy!=-4))return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(stage_c512_preproject_fp8,dim3(windows*4*32),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(raw),static_cast<const uint8_t*>(preproject),
        static_cast<const int32_t*>(ai),static_cast<uint8_t*>(projected),windows,width,height,ox,oy);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL(stage_c512_group_fp8,dim3(windows*4*8),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(projected),static_cast<const uint8_t*>(expand),
        static_cast<const uint8_t*>(contract),static_cast<const int32_t*>(perm64),
        static_cast<const int32_t*>(perm256),static_cast<uint8_t*>(grouped),windows);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL(stage_c512_ffn_project_fp8,dim3(windows*4*32),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(raw),static_cast<const uint8_t*>(grouped),
        static_cast<const uint8_t*>(ffn_project),static_cast<const _Float16*>(ffn_scale),
        static_cast<const int32_t*>(ri),static_cast<const int32_t*>(permutation),
        static_cast<_Float16*>(post),static_cast<uint8_t*>(resident),windows,width,height,ox,oy);
    return int(hipGetLastError());
}

template<int C> static int qkv_entry(const void* post,const void* qkv,const void* qscale,
    const void* permutation,void* projection,void* q,void* k,void* v,const void* lut,
    int windows,void* stream) {
    if(!post||!qkv||!qscale||!permutation||!projection||!q||!k||!v||
       !lut||windows<1||windows>262144)
        return int(hipErrorInvalidValue);
    auto* q_bytes=static_cast<uint8_t*>(q);
    auto* k_bytes=static_cast<uint8_t*>(k);
    auto* v_bytes=static_cast<uint8_t*>(v);
    size_t resident_stride=size_t(k_bytes-q_bytes);
    if(!resident_stride||v_bytes-k_bytes!=resident_stride)
        return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL((stage_qkv_project_fp8<C>),dim3(windows*4*(3*C/16)),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(post),static_cast<const uint8_t*>(qkv),
        static_cast<const int32_t*>(permutation),static_cast<_Float16*>(projection),windows);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    size_t vectors=size_t(windows)*(C/32)*64;
    size_t scalars=vectors*3*32;
    hipLaunchKernelGGL((stage_qkv_norm_fp16_inplace<C>),
        dim3(unsigned((scalars+255)/256)),dim3(256),0,hip_stream,
        static_cast<_Float16*>(projection),static_cast<const _Float16*>(qscale),vectors);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    size_t part_scalars=vectors*32;
    size_t part_words=(part_scalars+3)/4;
    hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<C,0>),
        dim3(unsigned((part_words+255)/256)),dim3(256),0,hip_stream,
        static_cast<const uint16_t*>(projection),reinterpret_cast<uint32_t*>(q_bytes),
        static_cast<const uint8_t*>(lut),part_scalars);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<C,1>),
        dim3(unsigned((part_words+255)/256)),dim3(256),0,hip_stream,
        static_cast<const uint16_t*>(projection),reinterpret_cast<uint32_t*>(k_bytes),
        static_cast<const uint8_t*>(lut),part_scalars);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<C,2>),
        dim3(unsigned((part_words+255)/256)),dim3(256),0,hip_stream,
        static_cast<const uint16_t*>(projection),reinterpret_cast<uint32_t*>(v_bytes),
        static_cast<const uint8_t*>(lut),part_scalars);
    return int(hipGetLastError());
}

template<int C> static int qkv_ordered_entry(const void* post,const void* qkv,
    const void* qscale,const void* permutation,void* projection,void* q,void* k,void* v,
    const void* lut,int windows,void* stream) {
    if(!post||!qkv||!qscale||!permutation||!projection||!q||!k||!v||!lut||
       windows<1||windows>262144)return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL((stage_qkv_project_fp8<C>),dim3(windows*4*(3*C/16)),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(post),static_cast<const uint8_t*>(qkv),
        static_cast<const int32_t*>(permutation),static_cast<_Float16*>(projection),windows);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    error=hipStreamSynchronize(hip_stream);if(error!=hipSuccess)return int(error);
    size_t vectors=size_t(windows)*(C/32)*64;
    size_t scalars=vectors*3*32;
    hipLaunchKernelGGL((stage_qkv_norm_fp16_inplace<C>),
        dim3(unsigned((scalars+255)/256)),dim3(256),0,hip_stream,
        static_cast<_Float16*>(projection),static_cast<const _Float16*>(qscale),vectors);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    error=hipStreamSynchronize(hip_stream);if(error!=hipSuccess)return int(error);
    size_t part_scalars=vectors*32;
    size_t part_words=(part_scalars+3)/4;
#define PUBLISH_ORDERED_QKV(PART,TARGET) \
    hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<C,PART>), \
        dim3(unsigned((part_words+255)/256)),dim3(256),0,hip_stream, \
        static_cast<const uint16_t*>(projection),reinterpret_cast<uint32_t*>(TARGET), \
        static_cast<const uint8_t*>(lut),part_scalars); \
    error=hipGetLastError();if(error!=hipSuccess)return int(error)
    PUBLISH_ORDERED_QKV(0,q);
    PUBLISH_ORDERED_QKV(1,k);
    PUBLISH_ORDERED_QKV(2,v);
#undef PUBLISH_ORDERED_QKV
    return int(hipStreamSynchronize(hip_stream));
}

template<int C> static int qkv_fused_entry(const void* post,const void* qkv,const void* qscale,
    const void* permutation,void* q,void* k,void* v,int windows,void* stream) {
    if(!post||!qkv||!qscale||!permutation||!q||!k||!v||windows<1||windows>262144)
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((stage_qkv_norm_fused_fp8<C>),dim3(windows*(C/32)*4),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(post),
        static_cast<const uint8_t*>(qkv),static_cast<const _Float16*>(qscale),
        static_cast<const int32_t*>(permutation),static_cast<uint8_t*>(q),
        static_cast<uint8_t*>(k),static_cast<uint8_t*>(v),windows);
    return int(hipGetLastError());
}

template<int C> static int qkv_compact_publish_entry(const void* post,const void* qkv,
    const void* qscale,const void* permutation,void* projection,void* q,void* k,void* v,
    const void* lut,int windows,void* stream) {
    if(!post||!qkv||!qscale||!permutation||!projection||!q||!k||!v||!lut||
       windows<1||windows>262144)return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL((stage_qkv_project_fp8<C>),dim3(windows*4*(3*C/16)),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(post),static_cast<const uint8_t*>(qkv),
        static_cast<const int32_t*>(permutation),static_cast<_Float16*>(projection),windows);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    size_t vectors=size_t(windows)*(C/32)*64;
    size_t scalars=vectors*3*32;
    hipLaunchKernelGGL((stage_qkv_norm_fp16_inplace<C>),
        dim3(unsigned((scalars+255)/256)),dim3(256),0,hip_stream,
        static_cast<_Float16*>(projection),static_cast<const _Float16*>(qscale),vectors);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    size_t part_scalars=vectors*32,part_words=(part_scalars+3)/4;
    hipLaunchKernelGGL((stage_qkv_pack_e4x4_all_from_fp16<C>),
        dim3(unsigned((part_words+255)/256)),dim3(256),0,hip_stream,
        static_cast<const uint16_t*>(projection),reinterpret_cast<uint32_t*>(q),
        reinterpret_cast<uint32_t*>(k),reinterpret_cast<uint32_t*>(v),
        static_cast<const uint8_t*>(lut),part_scalars);
    return int(hipGetLastError());
}

// Diagnostic-only direct publisher used to separate graph-capture defects from
// the gfx1201 byte/word publication path. It is not part of the frame hot path.
extern "C" __declspec(dllexport) int nr_stage_debug_pack_e4x4_c256(
    const void* projection, void* target, const void* lut, size_t scalar_count,
    int part, void* stream) {
    if (!projection || !target || !lut || !scalar_count || part < 0 || part > 2)
        return int(hipErrorInvalidValue);
    size_t words = (scalar_count + 3) / 4;
    auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    if (part == 0)
        hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<256,0>),
            dim3(unsigned((words + 255) / 256)), dim3(256), 0, hip_stream,
            static_cast<const uint16_t*>(projection), static_cast<uint32_t*>(target),
            static_cast<const uint8_t*>(lut), scalar_count);
    else if (part == 1)
        hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<256,1>),
            dim3(unsigned((words + 255) / 256)), dim3(256), 0, hip_stream,
            static_cast<const uint16_t*>(projection), static_cast<uint32_t*>(target),
            static_cast<const uint8_t*>(lut), scalar_count);
    else
        hipLaunchKernelGGL((stage_qkv_pack_e4x4_part_from_fp16<256,2>),
            dim3(unsigned((words + 255) / 256)), dim3(256), 0, hip_stream,
            static_cast<const uint16_t*>(projection), static_cast<uint32_t*>(target),
            static_cast<const uint8_t*>(lut), scalar_count);
    return int(hipGetLastError());
}

// Fixed-sequence bring-up entry: materialize the FP16 QKV projection and its
// normalization as a completed boundary before a separate resident publisher
// consumes it. This avoids relying on the failing arena-local producer/consumer
// visibility path while preserving the existing mathematical operations.
extern "C" __declspec(dllexport) int nr_stage_c256_qkv_projection_norm_fp16(
    const void* post,const void* qkv,const void* qscale,const void* permutation,
    void* projection,int windows,void* stream) {
    if(!post||!qkv||!qscale||!permutation||!projection||windows<1||windows>262144)
        return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL((stage_qkv_project_fp8<256>),dim3(windows*4*(3*256/16)),
        dim3(32),0,hip_stream,static_cast<const uint8_t*>(post),
        static_cast<const uint8_t*>(qkv),static_cast<const int32_t*>(permutation),
        static_cast<_Float16*>(projection),windows);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    error=hipStreamSynchronize(hip_stream);if(error!=hipSuccess)return int(error);
    constexpr size_t vectors_per_window=(256/32)*64;
    size_t vectors=size_t(windows)*vectors_per_window;
    size_t scalars=vectors*3*32;
    hipLaunchKernelGGL((stage_qkv_norm_fp16_inplace<256>),
        dim3(unsigned((scalars+255)/256)),dim3(256),0,hip_stream,
        static_cast<_Float16*>(projection),static_cast<const _Float16*>(qscale),vectors);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    return int(hipStreamSynchronize(hip_stream));
}

template<int C> static int ffn_entry(const void* raw,const void* expand,const void* contract,
    const void* mix,const void* scale,const void* ai,const void* ri,const void* permutation,
    const void* inverse_permutation,void* grouped,void* seed,void* post,void* post_resident,
    const void* lut,int windows,int width,int height,int ox,int oy,int force_publish_sync,
    void* stream) {
    if(!raw||!expand||!contract||!mix||!scale||!ai||!ri||!permutation||
       !inverse_permutation||!grouped||!seed||!post||!post_resident||!lut||
       windows<1||windows>262144||width<=0||height<=0||width%4||height%4||
       (ox!=0&&ox!=-4)||(oy!=0&&oy!=-4))return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL((stage_ffn_group_fp8<C>),dim3(windows*(C/32)*4),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(raw),static_cast<const uint8_t*>(expand),
        static_cast<const uint8_t*>(contract),static_cast<const _Float16*>(scale),
        static_cast<const int32_t*>(ai),static_cast<const int32_t*>(ri),
        static_cast<const int32_t*>(permutation),static_cast<const int32_t*>(inverse_permutation),
        static_cast<uint8_t*>(grouped),static_cast<_Float16*>(seed),windows,width,height,ox,oy);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL((stage_ffn_mix_fp8<C>),dim3(windows*4*(C/16)),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(grouped),static_cast<const uint8_t*>(mix),
        static_cast<const _Float16*>(seed),static_cast<_Float16*>(post),windows);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    // Diagnostic ordering fence for direct/fixed-sequence execution. Stream
    // synchronization is illegal while NRPlan records the equivalent graph.
    if(force_publish_sync){
        error=hipStreamSynchronize(hip_stream);if(error!=hipSuccess)return int(error);
    }
    size_t scalars=size_t(windows)*64*C,words=(scalars+3)/4;
    hipLaunchKernelGGL(stage_pack_e4x4_from_fp16,
        dim3(unsigned((words+255)/256)),dim3(256),0,hip_stream,
        static_cast<const uint16_t*>(post),static_cast<uint32_t*>(post_resident),
        static_cast<const uint8_t*>(lut),scalars);
    return int(hipGetLastError());
}

template<int C> static int scatter_entry(const void* logical,void* packed,int width,int height,
    int ox,int oy,void* stream) {
    if(!logical||!packed||width<=0||height<=0||width%4||height%4||
       (ox!=0&&ox!=-4)||(oy!=0&&oy!=-4))return int(hipErrorInvalidValue);
    size_t count=size_t(width)*height*C;
    hipLaunchKernelGGL((stage_scatter_fp8<C>),dim3(unsigned((count+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(logical),
        static_cast<uint8_t*>(packed),width,height,ox,oy,count);
    return int(hipGetLastError());
}

template<int C> static int attention_entry(const void* q,const void* k,const void* v,
    const void* bias,void* value,int windows,void* stream) {
    if(!q||!k||!v||!bias||!value||windows<1||windows>262144)
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((stage_attention_fp8<C>),dim3(windows*(C/32)*4),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(q),
        static_cast<const uint8_t*>(k),static_cast<const uint8_t*>(v),
        static_cast<const _Float16*>(bias),static_cast<uint8_t*>(value),windows);
    return int(hipGetLastError());
}

template<int C> static int project_entry(const void* post,const void* value,const void* weight,
    const void* scale,const void* permutation,void* output,void* next,int windows,void* stream) {
    if(!post||!value||!weight||!scale||!permutation||!output||!next||windows<1||windows>262144)
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((stage_project_fp8<C>),dim3(windows*4*(C/16)),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(post),
        static_cast<const uint8_t*>(value),static_cast<const uint8_t*>(weight),
        static_cast<const _Float16*>(scale),static_cast<const int32_t*>(permutation),
        static_cast<_Float16*>(output),static_cast<uint8_t*>(next),windows);
    return int(hipGetLastError());
}

// Keep the standard C64-C256 stage launch sequence in the same HIP translation
// unit as its kernels.  Besides reducing the runtime ABI surface, this gives the
// native plan one stable stage-specific entry point for fixed-sequence replay.
// The stop flag is diagnostic-only and cuts the sequence after Q/K/V publish.
template<int C> static int standard_stage_entry(
    const void* raw,const void* expand,const void* contract,const void* mix,
    const void* ffn_scale,const void* ai,const void* ri,const void* ffn_permutation,
    const void* inverse_permutation,void* grouped,void* seed,void* post,
    void* post_resident,const void* qkv,const void* qscale,const void* permutation,
    void* qkv_projection,void* q,void* k,void* v,const void* lut,const void* bias,
    void* value,const void* project,const void* residual_scale,void* output,
    void* next_resident,int windows,int width,int height,int ox,int oy,int stop_after_qkv,
    int force_publish_sync,void* stream) {
    int error=ffn_entry<C>(raw,expand,contract,mix,ffn_scale,ai,ri,ffn_permutation,
        inverse_permutation,grouped,seed,post,post_resident,lut,windows,width,height,ox,oy,
        force_publish_sync,stream);
    if(error)return error;
    error=qkv_entry<C>(post_resident,qkv,qscale,permutation,qkv_projection,q,k,v,lut,
        windows,stream);
    if(error||stop_after_qkv)return error;
    error=attention_entry<C>(q,k,v,bias,value,windows,stream);
    if(error)return error;
    error=project_entry<C>(post,value,project,residual_scale,permutation,output,
        post_resident,windows,stream);
    if(error)return error;
    return scatter_entry<C>(post_resident,next_resident,width,height,ox,oy,stream);
}

#define EXPORT_STAGE(C) \
extern "C" __declspec(dllexport) int nr_stage_c##C##_ffn_fp8(const void* raw,const void* expand,const void* contract,const void* mix,const void* scale,const void* ai,const void* ri,const void* permutation,const void* inverse_permutation,void* grouped,void* seed,void* post,void* post_resident,const void* lut,int windows,int width,int height,int ox,int oy,int force_publish_sync,void* stream){return ffn_entry<C>(raw,expand,contract,mix,scale,ai,ri,permutation,inverse_permutation,grouped,seed,post,post_resident,lut,windows,width,height,ox,oy,force_publish_sync,stream);} \
extern "C" __declspec(dllexport) int nr_stage_c##C##_qkv_norm_fp8(const void* post,const void* qkv,const void* qscale,const void* permutation,void* projection,void* q,void* k,void* v,const void* lut,int windows,void* stream){return qkv_entry<C>(post,qkv,qscale,permutation,projection,q,k,v,lut,windows,stream);} \
extern "C" __declspec(dllexport) int nr_stage_c##C##_attention_fp8(const void* q,const void* k,const void* v,const void* bias,void* value,int windows,void* stream){return attention_entry<C>(q,k,v,bias,value,windows,stream);} \
extern "C" __declspec(dllexport) int nr_stage_c##C##_project_fp8(const void* post,const void* value,const void* weight,const void* scale,const void* permutation,void* output,void* next,int windows,void* stream){return project_entry<C>(post,value,weight,scale,permutation,output,next,windows,stream);} \
extern "C" __declspec(dllexport) int nr_stage_c##C##_scatter_fp8(const void* logical,void* packed,int width,int height,int ox,int oy,void* stream){return scatter_entry<C>(logical,packed,width,height,ox,oy,stream);} \
extern "C" __declspec(dllexport) int nr_stage_c##C##_block_fp8(const void* raw,const void* expand,const void* contract,const void* mix,const void* ffn_scale,const void* ai,const void* ri,const void* ffn_permutation,const void* inverse_permutation,void* grouped,void* seed,void* post,void* post_resident,const void* qkv,const void* qscale,const void* permutation,void* qkv_projection,void* q,void* k,void* v,const void* lut,const void* bias,void* value,const void* project,const void* residual_scale,void* output,void* next_resident,int windows,int width,int height,int ox,int oy,int stop_after_qkv,int force_publish_sync,void* stream){return standard_stage_entry<C>(raw,expand,contract,mix,ffn_scale,ai,ri,ffn_permutation,inverse_permutation,grouped,seed,post,post_resident,qkv,qscale,permutation,qkv_projection,q,k,v,lut,bias,value,project,residual_scale,output,next_resident,windows,width,height,ox,oy,stop_after_qkv,force_publish_sync,stream);}

#define EXPORT_STAGE_LAUNCH(C) \
extern "C" __declspec(dllexport) int nr_stage_c##C##_block_launch(const NRStandardStageLaunch* p){if(!p)return int(hipErrorInvalidValue);return standard_stage_entry<C>(p->raw,p->expand,p->contract,p->mix,p->ffn_scale,p->a_index,p->residual_index,p->ffn_permutation,p->inverse_permutation,p->grouped,p->seed,p->post,p->post_resident,p->qkv,p->qscale,p->permutation,p->qkv_projection,p->q,p->k,p->v,p->lut,p->bias,p->value,p->project,p->residual_scale,p->output,p->next_resident,p->windows,p->width,p->height,p->origin_x,p->origin_y,p->stop_after_qkv,p->force_publish_sync,p->stream);}

#define EXPORT_ORDERED_QKV(C) \
extern "C" __declspec(dllexport) int nr_stage_c##C##_qkv_ordered_fp8(const void* post,const void* qkv,const void* qscale,const void* permutation,void* projection,void* q,void* k,void* v,const void* lut,int windows,void* stream){return qkv_ordered_entry<C>(post,qkv,qscale,permutation,projection,q,k,v,lut,windows,stream);}

#define EXPORT_COMPACT_QKV(C) \
extern "C" __declspec(dllexport) int nr_stage_c##C##_qkv_compact_publish_fp8(const void* post,const void* qkv,const void* qscale,const void* permutation,void* projection,void* q,void* k,void* v,const void* lut,int windows,void* stream){return qkv_compact_publish_entry<C>(post,qkv,qscale,permutation,projection,q,k,v,lut,windows,stream);}

#define EXPORT_FULL_FUSED_QKV(C) \
extern "C" __declspec(dllexport) int nr_stage_c##C##_qkv_full_fused_fp8(const void* post,const void* qkv,const void* qscale,const void* permutation,void* q,void* k,void* v,const void* lut,int windows,void* stream){if(!post||!qkv||!qscale||!permutation||!q||!k||!v||!lut||windows<1||windows>262144)return int(hipErrorInvalidValue);hipLaunchKernelGGL((stage_qkv_full_fused_lut_fp8<C,false>),dim3(windows*(C/32)*4),dim3(32),0,reinterpret_cast<hipStream_t>(stream),post,static_cast<const uint8_t*>(qkv),static_cast<const _Float16*>(qscale),static_cast<const int32_t*>(permutation),reinterpret_cast<uint32_t*>(q),reinterpret_cast<uint32_t*>(k),reinterpret_cast<uint32_t*>(v),static_cast<const uint8_t*>(lut),windows);return int(hipGetLastError());} \
extern "C" __declspec(dllexport) int nr_stage_c##C##_qkv_full_fused_from_fp16(const void* post,const void* qkv,const void* qscale,const void* permutation,void* q,void* k,void* v,const void* lut,int windows,void* stream){if(!post||!qkv||!qscale||!permutation||!q||!k||!v||!lut||windows<1||windows>262144)return int(hipErrorInvalidValue);hipLaunchKernelGGL((stage_qkv_full_fused_lut_fp8<C,true>),dim3(windows*(C/32)*4),dim3(32),0,reinterpret_cast<hipStream_t>(stream),post,static_cast<const uint8_t*>(qkv),static_cast<const _Float16*>(qscale),static_cast<const int32_t*>(permutation),reinterpret_cast<uint32_t*>(q),reinterpret_cast<uint32_t*>(k),reinterpret_cast<uint32_t*>(v),static_cast<const uint8_t*>(lut),windows);return int(hipGetLastError());}

EXPORT_STAGE(64)
EXPORT_STAGE(128)
EXPORT_STAGE(256)
EXPORT_STAGE_LAUNCH(64)
EXPORT_STAGE_LAUNCH(128)
EXPORT_STAGE_LAUNCH(256)
EXPORT_ORDERED_QKV(32)
EXPORT_ORDERED_QKV(64)
EXPORT_ORDERED_QKV(128)
EXPORT_ORDERED_QKV(256)
EXPORT_COMPACT_QKV(64)
EXPORT_COMPACT_QKV(128)
EXPORT_COMPACT_QKV(256)
EXPORT_FULL_FUSED_QKV(32)
EXPORT_FULL_FUSED_QKV(64)
EXPORT_FULL_FUSED_QKV(128)
EXPORT_FULL_FUSED_QKV(256)

extern "C" __declspec(dllexport) int nr_stage_c32_ffn_fp8(const void* raw,
    const void* expand,const void* contract,const void*,const void* scale,const void* ai,
    const void* ri,const void* permutation,const void*,void*,void*,void* post,
    void* resident,int windows,int width,int height,int ox,int oy,void* stream) {
    return c32_ffn_entry(raw,expand,contract,scale,ai,ri,permutation,post,resident,
        windows,width,height,ox,oy,stream);
}
extern "C" __declspec(dllexport) int nr_stage_head_c32_ffn_fp8(const void* raw,
    const void* expand,const void* contract,const void* scale,const void* ai,
    const void* ri,const void* permutation,void* post,void* resident,int windows,
    void* stream) {
    if(!raw||!expand||!contract||!scale||!ai||!ri||!permutation||!post||!resident||
       windows<1||windows>262144)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(stage_head_c32_ffn_fp8,dim3(windows*4),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(raw),
        static_cast<const uint8_t*>(expand),static_cast<const uint8_t*>(contract),
        static_cast<const _Float16*>(scale),static_cast<const int32_t*>(ai),
        static_cast<const int32_t*>(ri),static_cast<const int32_t*>(permutation),
        static_cast<_Float16*>(post),static_cast<uint8_t*>(resident),windows);
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_stage_c32_qkv_norm_fp8(const void* post,
    const void* qkv,const void* qscale,const void* permutation,void* projection,
    void* q,void* k,void* v,const void* lut,int windows,void* stream) {
    return qkv_entry<32>(post,qkv,qscale,permutation,projection,q,k,v,lut,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c32_qkv_compact_publish_fp8(
    const void* post,const void* qkv,const void* qscale,const void* permutation,
    void* projection,void* q,void* k,void* v,const void* lut,int windows,void* stream) {
    return qkv_compact_publish_entry<32>(post,qkv,qscale,permutation,projection,q,k,v,
        lut,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c32_attention_fp8(const void* q,
    const void* k,const void* v,const void* bias,void* value,int windows,void* stream) {
    return attention_entry<32>(q,k,v,bias,value,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c32_project_fp8(const void* post,
    const void* value,const void* weight,const void* scale,const void* permutation,
    void* output,void* next,int windows,void* stream) {
    return project_entry<32>(post,value,weight,scale,permutation,output,next,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c32_scatter_fp8(const void* logical,
    void* packed,int width,int height,int ox,int oy,void* stream) {
    return scatter_entry<32>(logical,packed,width,height,ox,oy,stream);
}

extern "C" __declspec(dllexport) int nr_stage_c512_ffn_fp8(
    const void* raw,const void* preproject,const void* expand,const void* contract,
    const void* ffn_project,const void* ffn_scale,const void* ai,const void* ri,
    const void* perm64,const void* perm256,const void* permutation,void* projected,
    void* grouped,void* post,void* resident,int windows,int width,int height,int ox,
    int oy,void* stream) {
    return c512_ffn_entry(raw,preproject,expand,contract,ffn_project,ffn_scale,ai,ri,
        perm64,perm256,permutation,projected,grouped,post,resident,windows,width,height,
        ox,oy,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c512_qkv_norm_fp8(const void* post,
    const void* qkv,const void* qscale,const void* permutation,void* q,void* k,void* v,
    int windows,void* stream) {
    return qkv_fused_entry<512>(post,qkv,qscale,permutation,q,k,v,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c512_attention_fp8(const void* q,
    const void* k,const void* v,const void* bias,void* value,int windows,void* stream) {
    return attention_entry<512>(q,k,v,bias,value,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c512_project_fp8(const void* post,
    const void* value,const void* weight,const void* scale,const void* permutation,
    void* output,void* next,int windows,void* stream) {
    return project_entry<512>(post,value,weight,scale,permutation,output,next,windows,stream);
}
extern "C" __declspec(dllexport) int nr_stage_c512_scatter_fp8(const void* logical,
    void* packed,int width,int height,int ox,int oy,void* stream) {
    return scatter_entry<512>(logical,packed,width,height,ox,oy,stream);
}
#undef EXPORT_STAGE
#undef EXPORT_STAGE_LAUNCH
#undef EXPORT_ORDERED_QKV
