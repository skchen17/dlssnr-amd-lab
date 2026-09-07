#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using I2 = int __attribute__((ext_vector_type(2)));

__device__ static uint8_t to_e4m3(_Float16 x) {
    return __hip_cvt_float_to_fp8(float(x),__HIP_SATFINITE,__HIP_E4M3);
}

__device__ static _Float16 from_e4m3(uint8_t x) {
    return (_Float16)__half(__hip_cvt_fp8_to_halfraw(x,__HIP_E4M3));
}

__global__ static void pack_weight_e4m3(const _Float16* source,uint8_t* packed,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)packed[i]=to_e4m3(source[i]);
}

// Existing semantic boundary: materialize the E4M3-rounded value as FP16.
__global__ static void producer_fp16_boundary(const _Float16* source,_Float16* boundary,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)boundary[i]=from_e4m3(to_e4m3(source[i]));
}

// Resident boundary: preserve exactly the same E4M3 byte instead of expanding
// it to FP16 only to repack it in the next kernel.
__global__ static void producer_fp8_boundary(const _Float16* source,uint8_t* boundary,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)boundary[i]=to_e4m3(source[i]);
}

template<bool RESIDENT> __global__ static void consumer_fp8_wmma(
    const void* activation,const uint8_t* packed_weight,_Float16* output,int matrices) {
    int matrix=blockIdx.x,lane=threadIdx.x,token=lane%16,group=lane/16;
    if(matrix>=matrices)return;
    I2 a={},b={};
    for(int e=0;e<8;++e){
        int k=group*8+e;
        uint8_t weight=packed_weight[matrix*256+k*16+token];
        uint8_t value;
        if constexpr(RESIDENT)value=static_cast<const uint8_t*>(activation)[matrix*256+token*16+k];
        else value=to_e4m3(static_cast<const _Float16*>(activation)[matrix*256+token*16+k]);
        a[e/4]|=unsigned(weight)<<((e%4)*8);
        b[e/4]|=unsigned(value)<<((e%4)*8);
    }
    F8 acc={};
    acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);
    for(int e=0;e<8;++e)
        output[matrix*256+token*16+group*8+e]=(_Float16)__float2half_rn(acc[e]);
}

extern "C" __declspec(dllexport) int nr_fp8_prepack_weights(
    const void* source,void* packed,size_t count,void* stream) {
    if(!source||!packed||!count)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(pack_weight_e4m3,dim3(unsigned((count+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(source),
        static_cast<uint8_t*>(packed),count);
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_fp8_chain_fp16_boundary(
    const void* source,const void* packed_weight,void* boundary,void* output,
    int matrices,void* stream) {
    if(!source||!packed_weight||!boundary||!output||matrices<1||matrices>1048576)
        return int(hipErrorInvalidValue);
    size_t count=size_t(matrices)*256;
    hipLaunchKernelGGL(producer_fp16_boundary,dim3(unsigned((count+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(source),
        static_cast<_Float16*>(boundary),count);
    hipLaunchKernelGGL((consumer_fp8_wmma<false>),dim3(matrices),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),boundary,static_cast<const uint8_t*>(packed_weight),
        static_cast<_Float16*>(output),matrices);
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_fp8_chain_resident(
    const void* source,const void* packed_weight,void* boundary,void* output,
    int matrices,void* stream) {
    if(!source||!packed_weight||!boundary||!output||matrices<1||matrices>1048576)
        return int(hipErrorInvalidValue);
    size_t count=size_t(matrices)*256;
    hipLaunchKernelGGL(producer_fp8_boundary,dim3(unsigned((count+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(source),
        static_cast<uint8_t*>(boundary),count);
    hipLaunchKernelGGL((consumer_fp8_wmma<true>),dim3(matrices),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),boundary,static_cast<const uint8_t*>(packed_weight),
        static_cast<_Float16*>(output),matrices);
    return int(hipGetLastError());
}
