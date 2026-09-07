#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

using BottleF8 = float __attribute__((ext_vector_type(8)));
using BottleI2 = int __attribute__((ext_vector_type(2)));

__device__ static _Float16 bottle_hround(float x) { return (_Float16)__float2half_rn(x); }
__device__ static uint8_t bottle_e4(float x) {
    return __hip_cvt_float_to_fp8(x,__HIP_SATFINITE,__HIP_E4M3);
}
__device__ static _Float16 bottle_e4h(uint8_t x) {
    return (_Float16)__half(__hip_cvt_fp8_to_halfraw(x,__HIP_E4M3));
}
__device__ static BottleF8 bottle_mma(BottleI2 a,BottleI2 b,BottleF8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c);
}

template<int C> __device__ static size_t packed_image_index(int x,int y,int channel,int width) {
    int token=(y&3)*4+(x&3),col=channel&31;
    int mma=col/16*2+(col%16)/8;
    int element=(token>=8)*2+(col&1);
    int pair=mma*2+element/2;
    int operation=(pair&4)|((pair&1)<<1)|((pair&2)>>1);
    size_t cell=size_t(y/4)*(width/4)+x/4;
    return cell*(16*C)+(channel/32)*512+(token%8*4+col%8/2)*16+
           operation*2+(element&1);
}

__device__ static uint8_t pooled_c512(const uint8_t* input,int x,int y,int channel,
    int width) {
    _Float16 a=bottle_hround(float(bottle_e4h(input[packed_image_index<512>(x*2,y*2,channel,width)]))+
                             float(bottle_e4h(input[packed_image_index<512>(x*2+1,y*2,channel,width)])));
    _Float16 b=bottle_hround(float(bottle_e4h(input[packed_image_index<512>(x*2,y*2+1,channel,width)]))+
                             float(bottle_e4h(input[packed_image_index<512>(x*2+1,y*2+1,channel,width)])));
    return bottle_e4(float(bottle_hround(float(bottle_hround(float(a)+float(b)))*.25f)));
}

// block30 boundary: preserve the C512 skip, pool 60x36 to 30x18, pad to
// 32x20, project 512->1024, and write the resident ViT input directly.
__global__ void encoder_final_to_vit_fp8(const uint8_t* input,const uint8_t* weight,
    const int32_t* permutation,uint8_t* vit,int source_width,int source_height,
    int padded_low_width,int padded_low_height) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int out_tile=blockIdx.x%64,token_tile=blockIdx.x/64,row=token_tile*16+token;
    int low_count=padded_low_width*padded_low_height;if(row>=low_count)return;
    int x=row%padded_low_width,y=row/padded_low_width;
    bool valid=x<source_width/2&&y<source_height/2;
    BottleF8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<32;++chunk){
        BottleI2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int source=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(weight[source*1024+out_tile*16+token])<<((e%4)*8);
            uint8_t activation=valid?pooled_c512(input,x,y,permutation[source],source_width):0;
            packed_activation[e/4]|=unsigned(activation)<<((e%4)*8);
        }
        acc=bottle_mma(packed_weight,packed_activation,acc);
    }
    for(int e=0;e<8;++e)
        vit[size_t(row)*1024+out_tile*16+half*8+e]=bottle_e4(float(bottle_hround(acc[e])));
}

// block39 boundary: project the padded 32x20 ViT field 1024->512, expand the
// valid 30x18 area to 60x36, add the preserved encoder skip, and directly
// publish the packed resident C512 image consumed by decoder block40.
__global__ void decoder_input_from_vit_fp8(const uint8_t* vit,const uint8_t* skip,
    const uint8_t* weight,const _Float16* scale,const int32_t* permutation,
    uint8_t* output,int width,int height,int low_width,int low_height) {
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int out_tile=blockIdx.x%32,token_tile=blockIdx.x/32,row=token_tile*16+token;
    if(row>=low_width*low_height)return;
    int x=row%low_width,y=row/low_width;
    BottleF8 acc={};
    #pragma clang loop unroll(disable)
    for(int chunk=0;chunk<64;++chunk){
        BottleI2 packed_weight={},packed_activation={};
        for(int e=0;e<8;++e){
            int source=chunk*16+half*8+e;
            packed_weight[e/4]|=unsigned(weight[source*512+out_tile*16+token])<<((e%4)*8);
            packed_activation[e/4]|=unsigned(vit[size_t(row)*1024+permutation[source]])<<((e%4)*8);
        }
        acc=bottle_mma(packed_weight,packed_activation,acc);
    }
    if(x>=width/2||y>=height/2)return;
    for(int e=0;e<8;++e){
        int channel=out_tile*16+half*8+e;
        _Float16 projected=bottle_hround(acc[e]);
        for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            size_t target=packed_image_index<512>(x*2+dx,y*2+dy,channel,width);
            _Float16 residual=bottle_hround(float(bottle_e4h(skip[target]))*float(scale[channel]));
            output[target]=bottle_e4(float(bottle_hround(float(projected)+float(residual))));
        }
    }
}

extern "C" __declspec(dllexport) int nr_encoder_final_to_vit_fp8(const void* input,
    void* skip,const void* weight,const void* permutation,void* vit,int width,int height,
    int low_width,int low_height,void* stream) {
    if(!input||!skip||!weight||!permutation||!vit||width<=0||height<=0||width%4||
       height%4||low_width<width/2||low_height<height/2||low_width%4||low_height%4)
        return int(hipErrorInvalidValue);
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    hipError_t error=hipMemcpyAsync(skip,input,size_t(width)*height*512,
                                    hipMemcpyDeviceToDevice,hip_stream);
    if(error!=hipSuccess)return int(error);
    int tokens=low_width*low_height;
    hipLaunchKernelGGL(encoder_final_to_vit_fp8,dim3(((tokens+15)/16)*64),dim3(32),0,
        hip_stream,static_cast<const uint8_t*>(input),static_cast<const uint8_t*>(weight),
        static_cast<const int32_t*>(permutation),static_cast<uint8_t*>(vit),width,height,
        low_width,low_height);
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_decoder_input_from_vit_fp8(const void* vit,
    const void* skip,const void* weight,const void* scale,const void* permutation,
    void* output,int width,int height,int low_width,int low_height,void* stream) {
    if(!vit||!skip||!weight||!scale||!permutation||!output||width<=0||height<=0||
       width%4||height%4||low_width<width/2||low_height<height/2||low_width%4||low_height%4)
        return int(hipErrorInvalidValue);
    int tokens=low_width*low_height;
    hipLaunchKernelGGL(decoder_input_from_vit_fp8,dim3(((tokens+15)/16)*32),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(vit),
        static_cast<const uint8_t*>(skip),static_cast<const uint8_t*>(weight),
        static_cast<const _Float16*>(scale),static_cast<const int32_t*>(permutation),
        static_cast<uint8_t*>(output),width,height,low_width,low_height);
    return int(hipGetLastError());
}
