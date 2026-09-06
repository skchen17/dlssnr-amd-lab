#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstddef>
#include <cstdint>

// Exact finite FP16 -> saturating E4M3FN -> FP16 boundary used by the
// recovered reference. NaN payloads are outside the accepted model inputs.
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

__device__ static size_t packed_image_offset(int y,int x,int channel,int width,int channels) {
    const int row=(y&3)*4+(x&3), col=channel&31;
    const int mma=(col>>4)*2+((col&15)>>3);
    const int pairs[8]={0,2,1,3,4,6,5,7};
    const int element=(row>=8)*2+(col&1);
    const int operation=pairs[mma*2+element/2];
    const int local=(channel/32)*512+((row&7)*4+(col&7)/2)*16+operation*2+(element&1);
    const size_t cell=(size_t(y/4)*(width/4)+x/4)*size_t(16*channels);
    return cell+size_t(local);
}

// Bounded encoder transition phase 1. One launch:
//   * quantizes the current-scale packed skip in-place layout;
//   * reads each global 2x2 neighbourhood directly from packed storage;
//   * preserves the reference's three explicit FP16 add boundaries;
//   * applies the recovered channel permutation while writing the contiguous
//     matrix input for the existing, numerically unchanged projection.
__global__ static void pool_permute_skip_kernel(const uint16_t* source,const int64_t* permutation,
    uint16_t* pooled,uint16_t* skip,int width,int height,int channels,size_t source_n,size_t pooled_n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<source_n) skip[i]=quantize_bits(source[i]);
    if(i>=pooled_n) return;
    const int output_channel=int(i%channels);
    const size_t pixel=i/channels;
    const int target_width=width/2;
    const int x=int(pixel%target_width), y=int(pixel/target_width);
    const int source_channel=int(permutation[output_channel]);
    const float p00=__half2float(__ushort_as_half(source[packed_image_offset(y*2,x*2,source_channel,width,channels)]));
    const float p01=__half2float(__ushort_as_half(source[packed_image_offset(y*2,x*2+1,source_channel,width,channels)]));
    const float p10=__half2float(__ushort_as_half(source[packed_image_offset(y*2+1,x*2,source_channel,width,channels)]));
    const float p11=__half2float(__ushort_as_half(source[packed_image_offset(y*2+1,x*2+1,source_channel,width,channels)]));
    const __half a=__float2half_rn(p00+p01);
    const __half b=__float2half_rn(p10+p11);
    const __half sum=__float2half_rn(__half2float(a)+__half2float(b));
    pooled[i]=__half_as_ushort(__float2half_rn(__half2float(sum)*0.25f));
}

extern "C" __declspec(dllexport) int native_transition_encoder_pool_permute_skip(
    const void* source,const void* permutation,void* pooled,void* skip,
    int width,int height,int channels,void* stream) {
    if(!source||!permutation||!pooled||!skip||width<=0||height<=0||width%8||height%8||channels<=0||channels%32)
        return int(hipErrorInvalidValue);
    const size_t source_n=size_t(width)*height*channels;
    const size_t pooled_n=size_t(width/2)*(height/2)*channels;
    const size_t n=source_n>pooled_n?source_n:pooled_n;
    hipLaunchKernelGGL(pool_permute_skip_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(source),
        static_cast<const int64_t*>(permutation),static_cast<uint16_t*>(pooled),
        static_cast<uint16_t*>(skip),width,height,channels,source_n,pooled_n);
    return int(hipGetLastError());
}

// Bounded encoder transition phase 3. The projection output stays logical only
// across the library GEMM boundary, then is quantized and written directly into
// the next scale's resident packed layout. No outview is materialized.
__global__ static void quantize_pack_kernel(const uint16_t* logical,uint16_t* resident,
    int width,int height,int channels,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int channel=int(i%channels);
    const size_t pixel=i/channels;
    const int x=int(pixel%width), y=int(pixel/width);
    resident[packed_image_offset(y,x,channel,width,channels)]=quantize_bits(logical[i]);
}

extern "C" __declspec(dllexport) int native_transition_encoder_quantize_pack(
    const void* logical,void* resident,int width,int height,int channels,void* stream) {
    if(!logical||!resident||width<=0||height<=0||width%4||height%4||channels<=0||channels%32)
        return int(hipErrorInvalidValue);
    const size_t n=size_t(width)*height*channels;
    hipLaunchKernelGGL(quantize_pack_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(logical),
        static_cast<uint16_t*>(resident),width,height,channels,n);
    return int(hipGetLastError());
}

// Decoder phase 1: consume the previous scale's resident packed feature and
// form the channel-permuted matrix input without an outview or logical HWC
// round trip.
__global__ static void unpack_permute_kernel(const uint16_t* resident,const int64_t* permutation,
    uint16_t* logical,int width,int height,int channels,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int output_channel=int(i%channels);
    const size_t pixel=i/channels;
    const int x=int(pixel%width), y=int(pixel/width);
    logical[i]=resident[packed_image_offset(y,x,int(permutation[output_channel]),width,channels)];
}

extern "C" __declspec(dllexport) int native_transition_decoder_unpack_permute(
    const void* resident,const void* permutation,void* logical,
    int width,int height,int channels,void* stream) {
    if(!resident||!permutation||!logical||width<=0||height<=0||width%4||height%4||channels<=0||channels%32)
        return int(hipErrorInvalidValue);
    const size_t n=size_t(width)*height*channels;
    hipLaunchKernelGGL(unpack_permute_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(resident),
        static_cast<const int64_t*>(permutation),static_cast<uint16_t*>(logical),width,height,channels,n);
    return int(hipGetLastError());
}

// Decoder phase 3: perform 2x replication and the original half-rounded skip
// scaling/addition, then write directly into the destination Swin packed layout.
__global__ static void expand_skip_pack_kernel(const uint16_t* projected,const uint16_t* skip,
    const uint16_t* scale,uint16_t* resident,int width,int height,int channels,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int channel=int(i%channels);
    const size_t pixel=i/channels;
    const int x=int(pixel%width), y=int(pixel/width);
    const size_t low=(size_t(y/2)*(width/2)+x/2)*channels+channel;
    const float skip_value=__half2float(__ushort_as_half(skip[packed_image_offset(y,x,channel,width,channels)]));
    const float scale_value=__half2float(__ushort_as_half(scale[channel]));
    const __half scaled=__float2half_rn(skip_value*scale_value);
    const float value=__half2float(__ushort_as_half(projected[low]))+__half2float(scaled);
    resident[packed_image_offset(y,x,channel,width,channels)]=__half_as_ushort(__float2half_rn(value));
}

extern "C" __declspec(dllexport) int native_transition_decoder_expand_skip_pack(
    const void* projected,const void* skip,const void* scale,void* resident,
    int width,int height,int channels,void* stream) {
    if(!projected||!skip||!scale||!resident||width<=0||height<=0||width%8||height%8||channels<=0||channels%32)
        return int(hipErrorInvalidValue);
    const size_t n=size_t(width)*height*channels;
    hipLaunchKernelGGL(expand_skip_pack_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(projected),
        static_cast<const uint16_t*>(skip),static_cast<const uint16_t*>(scale),
        static_cast<uint16_t*>(resident),width,height,channels,n);
    return int(hipGetLastError());
}
