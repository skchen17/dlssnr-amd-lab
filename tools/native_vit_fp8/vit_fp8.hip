#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

// Approximate resident-FP8 implementation of the recovered 1024-channel ViT.
// It intentionally streams global attention one 16-key tile at a time and
// never materializes an N-by-N score/probability tensor. The five launches are
// fixed for every block: FFN expand, FFN contract/residual, QKV/norm, streamed
// attention, and output projection/residual.

using VitF8 = float __attribute__((ext_vector_type(8)));
using VitI2 = int __attribute__((ext_vector_type(2)));

__device__ static _Float16 vit_hround(float x) {
    return (_Float16)__float2half_rn(x);
}
__device__ static uint8_t vit_e4(float x) {
    return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3);
}
__device__ static _Float16 vit_e4h(uint8_t x) {
    return (_Float16)__half(__hip_cvt_fp8_to_halfraw(x, __HIP_E4M3));
}
__device__ static VitF8 vit_mma(VitI2 a, VitI2 b, VitF8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
__device__ static _Float16 vit_cubic(_Float16 hx) {
    float x=float(hx),v=fminf(4.f,fmaxf(-4.f,x));
    _Float16 a=vit_hround(-0.055908203125f*fabsf(v)+0.447265625f);
    _Float16 b=vit_hround(v*float(a)+0.89453125f);
    return vit_e4h(vit_e4(float(vit_hround(x*float(b)))));
}
__device__ static _Float16 vit_exp_candidate(_Float16 score) {
    _Float16 mapped=vit_hround(float(score)*0.08953857421875f+1.708984375f);
    mapped=vit_hround(fminf(1.9775390625f,fmaxf(1.439453125f,float(mapped))));
    __half h=__float2half(float(mapped));
    uint16_t bits=__half_as_ushort(h);
    return (_Float16)__ushort_as_half(uint16_t((uint32_t(bits)*16u+16384u)&65535u));
}

// Each wave computes a 16-token x 16-output tile. The producer writes the
// exact resident layout consumed by the contract kernel, so there is no
// logical FP16 hidden tensor or per-frame FP8 packing launch.
__global__ void vit_ffn_expand_fp8(const uint8_t* input,const uint8_t* weight,
    const int32_t* permutation,uint8_t* hidden,int tokens) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int out_tile=blockIdx.x%256,token_tile=blockIdx.x/256;
    int row=token_tile*16+token;if(row>=tokens)return;
    VitF8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<64;++chunk){
        VitI2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int source=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(weight[source*4096+out_tile*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(input[size_t(row)*1024+permutation[source]])<<((e%4)*8);
        }
        acc=vit_mma(packed_weight,packed_activation,acc);
    }
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        hidden[size_t(row)*4096+channel]=vit_e4(float(vit_cubic(vit_hround(acc[e]))));
    }
}

__global__ void vit_ffn_contract_fp8(const uint8_t* input,const uint8_t* hidden,
    const uint8_t* weight,const _Float16* scale,const int32_t* permutation,
    uint8_t* post,int tokens) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int out_tile=blockIdx.x%64,token_tile=blockIdx.x/64;
    int row=token_tile*16+token;if(row>=tokens)return;
    VitF8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<256;++chunk){
        VitI2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int source=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(weight[source*1024+out_tile*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(hidden[size_t(row)*4096+permutation[source]])<<((e%4)*8);
        }
        acc=vit_mma(packed_weight,packed_activation,acc);
    }
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        _Float16 residual=vit_hround(float(vit_e4h(input[size_t(row)*1024+channel]))*
                                     float(scale[channel]));
        post[size_t(row)*1024+channel]=vit_e4(float(vit_hround(float(vit_hround(acc[e]))+
                                                               float(residual))));
    }
}

__global__ void vit_qkv_norm_fp8(const uint8_t* post,const uint8_t* qkv,
    const _Float16* qscale,const int32_t* permutation,uint8_t* q,uint8_t* k,
    uint8_t* v,int tokens) {
    __shared__ _Float16 projection[3][32][16];
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int head=blockIdx.x%32,token_tile=blockIdx.x/32,row=token_tile*16+token;
    if(row>=tokens)return;
    for(int part=0;part<3;++part)for(int out_half=0;out_half<2;++out_half){
        VitF8 acc={};
        #pragma clang loop unroll(disable)
        for(int chunk=0;chunk<64;++chunk){
            VitI2 packed_weight={},packed_activation={};
            for(int e=0;e<8;++e){
                int source=chunk*16+half*8+e;
                int global_out=part*1024+head*32+out_half*16+token;
                packed_weight[e/4]|=unsigned(qkv[global_out*1024+source])<<((e%4)*8);
                packed_activation[e/4]|=unsigned(post[size_t(row)*1024+permutation[source]])<<((e%4)*8);
            }
            acc=vit_mma(packed_weight,packed_activation,acc);
        }
        for(int e=0;e<8;++e)
            projection[part][out_half*16+half*8+e][token]=vit_hround(acc[e]);
    }
    __syncthreads();
    int channel=lane;
    for(int local_row=0;local_row<16&&token_tile*16+local_row<tokens;++local_row){
        size_t index=(size_t(head)*tokens+token_tile*16+local_row)*32+channel;
        v[index]=vit_e4(float(projection[2][channel][local_row]));
        for(int part=0;part<2;++part){
            float value=float(projection[part][channel][local_row]);
            float sum=float(vit_hround(value*value));
            for(int offset=1;offset<32;offset*=2){
                float other=__shfl_down(sum,offset,32);
                if((lane%(2*offset))==0)sum=float(vit_hround(sum+other));
            }
            sum=__shfl(sum,0,32);
            _Float16 normalized=vit_hround(value*float(vit_hround(
                rsqrtf(fmaxf(sum,6.198883056640625e-5f)))));
            if(part==0)normalized=vit_hround(float(normalized)*float(qscale[head]));
            (part==0?q:k)[index]=vit_e4(float(normalized));
        }
    }
}

// Two waves own one (head, 16-query) task. Wave 0 publishes one 16x16
// exponent tile, both waves immediately consume it for their 16 output
// channels, then the tile is reused. Only 512 bytes of LDS are live.
__global__ void vit_global_attention_fp8(const uint8_t* q,const uint8_t* k,
    const uint8_t* v,uint8_t* value,int tokens) {
    __shared__ uint8_t exponent[16][16];
    __shared__ float denominator[16];
    int wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,half=lane/16;
    int head=blockIdx.x%32,query_tile=blockIdx.x/32,row=query_tile*16+query;
    if(row>=tokens)return;
    if(wave==0&&lane<16)denominator[lane]=0.f;
    VitF8 numerator={};
    __syncthreads();
    int key_tiles=(tokens+15)/16;
    for(int key_tile=0;key_tile<key_tiles;++key_tile){
        if(wave==0){
            VitF8 score={};
            for(int chunk=0;chunk<2;++chunk){
                VitI2 packed_k={},packed_q={};
                for(int e=0;e<8;++e){
                    int channel=chunk*16+half*8+e,key=key_tile*16+query;
                    packed_k[e/4]|=unsigned(key<tokens?k[(size_t(head)*tokens+key)*32+channel]:0)<<((e%4)*8);
                    packed_q[e/4]|=unsigned(q[(size_t(head)*tokens+row)*32+channel])<<((e%4)*8);
                }
                score=vit_mma(packed_k,packed_q,score);
            }
            float partial=0.f;
            for(int e=0;e<8;++e){
                int key=key_tile*16+half*8+e;
                uint8_t encoded=key<tokens?vit_e4(float(vit_exp_candidate(vit_hround(score[e])))):0;
                exponent[query][half*8+e]=encoded;
                partial+=key<tokens?float(vit_e4h(encoded)):0.f;
            }
            partial+=__shfl_xor(partial,16,32);
            if(half==0)denominator[query]+=partial;
        }
        __syncthreads();
        VitI2 packed_v={},packed_p={};
        for(int e=0;e<8;++e){
            int key=key_tile*16+half*8+e,out_channel=wave*16+query;
            packed_v[e/4]|=unsigned(key<tokens?v[(size_t(head)*tokens+key)*32+out_channel]:0)<<((e%4)*8);
            packed_p[e/4]|=unsigned(exponent[query][half*8+e])<<((e%4)*8);
        }
        numerator=vit_mma(packed_v,packed_p,numerator);
        if((key_tile&1)==1)for(int e=0;e<8;++e)numerator[e]=float(vit_hround(numerator[e]));
        __syncthreads();
    }
    float reciprocal=float(vit_hround(1.f/fmaxf(denominator[query],6.198883056640625e-5f)));
    for(int e=0;e<8;++e){
        int channel=wave*16+half*8+e;
        value[(size_t(head)*tokens+row)*32+channel]=vit_e4(float(vit_hround(numerator[e]*reciprocal)));
    }
}

__global__ void vit_project_fp8(const uint8_t* post,const uint8_t* value,
    const uint8_t* weight,const _Float16* scale,const int32_t* permutation,
    uint8_t* next,int tokens) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int out_tile=blockIdx.x%64,token_tile=blockIdx.x/64,row=token_tile*16+token;
    if(row>=tokens)return;
    VitF8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<64;++chunk){
        VitI2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int source=chunk*16+half*8+e;
            int head=permutation[source]/32,channel=permutation[source]%32;
            packed_weight[e/4]|=unsigned(weight[source*1024+out_tile*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(value[(size_t(head)*tokens+row)*32+channel])<<((e%4)*8);
        }
        acc=vit_mma(packed_weight,packed_activation,acc);
    }
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        _Float16 residual=vit_hround(float(vit_e4h(post[size_t(row)*1024+channel]))*
                                     float(scale[channel]));
        next[size_t(row)*1024+channel]=vit_e4(float(vit_hround(float(vit_hround(acc[e]))+
                                                               float(residual))));
    }
}

extern "C" __declspec(dllexport) int nr_vit_block_fp8(const void* input,
    const void* expand,const void* contract,const void* qkv,const void* qscale,
    const void* project,const void* ffn_scale,const void* attention_scale,
    const void* perm1024,const void* perm4096,void* hidden,void* post,void* q,
    void* k,void* v,void* value,void* next,int tokens,void* stream) {
    if(!input||!expand||!contract||!qkv||!qscale||!project||!ffn_scale||
       !attention_scale||!perm1024||!perm4096||!hidden||!post||!q||!k||!v||
       !value||!next||tokens<=0||tokens>4096)return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    int token_tiles=(tokens+15)/16;
    hipLaunchKernelGGL(vit_ffn_expand_fp8,dim3(token_tiles*256),dim3(32),0,hip_stream,
        static_cast<const uint8_t*>(input),static_cast<const uint8_t*>(expand),
        static_cast<const int32_t*>(perm1024),static_cast<uint8_t*>(hidden),tokens);
    hipError_t error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL(vit_ffn_contract_fp8,dim3(token_tiles*64),dim3(32),0,hip_stream,
        static_cast<const uint8_t*>(input),static_cast<const uint8_t*>(hidden),
        static_cast<const uint8_t*>(contract),static_cast<const _Float16*>(ffn_scale),
        static_cast<const int32_t*>(perm4096),static_cast<uint8_t*>(post),tokens);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL(vit_qkv_norm_fp8,dim3(token_tiles*32),dim3(32),0,hip_stream,
        static_cast<const uint8_t*>(post),static_cast<const uint8_t*>(qkv),
        static_cast<const _Float16*>(qscale),static_cast<const int32_t*>(perm1024),
        static_cast<uint8_t*>(q),static_cast<uint8_t*>(k),static_cast<uint8_t*>(v),tokens);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL(vit_global_attention_fp8,dim3(token_tiles*32),dim3(64),0,hip_stream,
        static_cast<const uint8_t*>(q),static_cast<const uint8_t*>(k),
        static_cast<const uint8_t*>(v),static_cast<uint8_t*>(value),tokens);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    hipLaunchKernelGGL(vit_project_fp8,dim3(token_tiles*64),dim3(32),0,hip_stream,
        static_cast<const uint8_t*>(post),static_cast<const uint8_t*>(value),
        static_cast<const uint8_t*>(project),static_cast<const _Float16*>(attention_scale),
        static_cast<const int32_t*>(perm1024),static_cast<uint8_t*>(next),tokens);
    return int(hipGetLastError());
}
