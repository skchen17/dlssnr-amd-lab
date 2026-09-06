#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>

// Pure bit manipulation: exact FP16 -> saturating E4M3FN -> FP16 for finite
// inputs and infinities. NaN payload/sign are deliberately outside acceptance.
__host__ __device__ static unsigned rshift_even(unsigned v, unsigned s) {
    const unsigned base=v>>s, rem=v&((1u<<s)-1), half=1u<<(s-1);
    return base+(rem>half || (rem==half && (base&1)));
}
__host__ __device__ static uint16_t quantize_bits(uint16_t bits) {
    const unsigned sign=bits&0x8000, exponent=(bits>>10)&31, mantissa=bits&1023;
    if(exponent==31 && mantissa) return 0x7e00;
    unsigned code;
    if(exponent<9) code=rshift_even(1024+mantissa,16-exponent);
    else {
        const unsigned rounded=rshift_even(mantissa,7), carry=rounded>=8;
        code=((exponent-8+carry)<<3)|(carry?0:rounded);
    }
    if(code>126) code=126;
    unsigned e=code>>3, m=code&7;
    if(e) return uint16_t(sign|((e+8)<<10)|(m<<7));
    if(!m) return uint16_t(sign);
    unsigned he=9;
    while(m<8) {m<<=1; --he;}
    return uint16_t(sign|(he<<10)|((m-8)<<7));
}
__global__ static void quantize_kernel(const uint16_t* input,uint16_t* output,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) output[i]=quantize_bits(input[i]);
}
extern "C" __declspec(dllexport) int native_fusion_quantize_f16(
    const void* input,void* output,uint64_t n,void* stream) {
    if(!input || !output || !n || n>(uint64_t(1)<<32)) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(quantize_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(input),
        static_cast<uint16_t*>(output),size_t(n));
    return int(hipGetLastError());
}
// Exhaustive CPU gate exercises exactly the same conversion as device kernel;
// this function does not initialize HIP or issue work to any device.
extern "C" __declspec(dllexport) void native_fusion_quantize_host(
    const uint16_t* input,uint16_t* output,uint64_t n) {
    for(uint64_t i=0;i<n;++i) output[i]=quantize_bits(input[i]);
}

// Matches native_grouped_ffn.cubic_silu, NOT a single-round polynomial.
// -ffp-contract=off is mandatory: each FP32 multiply/add stays separate, then
// each of the three documented FP16 stores is rounded to nearest-even.
__host__ __device__ static uint16_t cubic_quantize_bits(uint16_t bits) {
    const float x=__half2float(__ushort_as_half(bits));
    const float clamped=x>4.0f?4.0f:(x< -4.0f?-4.0f:x);
    const float absolute=clamped<0.0f?-clamped:clamped;
    const float first_mul=-0.055908203125f*absolute;
    const float first=__half2float(__float2half_rn(first_mul+0.447265625f));
    const float second_mul=clamped*first;
    const float second=__half2float(__float2half_rn(second_mul+0.89453125f));
    const uint16_t hidden=__half_as_ushort(__float2half_rn(x*second));
    return quantize_bits(hidden);
}
__global__ static void cubic_quantize_kernel(const uint16_t* input,uint16_t* output,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) output[i]=cubic_quantize_bits(input[i]);
}
extern "C" __declspec(dllexport) int native_fusion_cubic_quantize_f16(
    const void* input,void* output,uint64_t n,void* stream) {
    if(!input || !output || !n || n>(uint64_t(1)<<32)) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(cubic_quantize_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(input),
        static_cast<uint16_t*>(output),size_t(n));
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) void native_fusion_cubic_quantize_host(
    const uint16_t* input,uint16_t* output,uint64_t n) {
    for(uint64_t i=0;i<n;++i) output[i]=cubic_quantize_bits(input[i]);
}
