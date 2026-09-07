#include "../native_nr_plan/nr_plan.h"
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ _Float16 hround(float value) {
    return (_Float16)__float2half_rn(value);
}

__device__ __forceinline__ uint8_t encode_e4(float value) {
    return __hip_cvt_float_to_fp8(value, __HIP_SATFINITE, __HIP_E4M3);
}

__device__ __forceinline__ _Float16 decode_e4(uint8_t value) {
    return (_Float16)__half(__hip_cvt_fp8_to_halfraw(value, __HIP_E4M3));
}

__device__ __forceinline__ uint64_t packed_c32_offset(
    int y, int x, int channel, int width) {
    const int row=(y&3)*4+(x&3),col=channel&31;
    const int mma=(col>>4)*2+((col&15)>>3);
    const int pairs[8]={0,2,1,3,4,6,5,7};
    const int element=(row>=8)*2+(col&1);
    const int operation=pairs[mma*2+element/2];
    const int local=((row&7)*4+(col&7)/2)*16+operation*2+(element&1);
    return (uint64_t(y/4)*(width/4)+x/4)*512+local;
}

__device__ __forceinline__ uint32_t pcg_step(uint32_t value) {
    const uint32_t shift=(value>>28)+4;
    value=((value>>shift)^value)*277803737u;
    return (value>>22)^value;
}

__device__ void positional_noise(int x,int y,uint32_t seed,float* n) {
    uint32_t hashed=pcg_step(uint32_t(x)*uint32_t(-1918454973) ^
        uint32_t(y)*uint32_t(-669632447) ^ seed*uint32_t(-1640531527) ^ 608135816u);
    constexpr uint32_t a[4]={747796405u,uint32_t(-93469191),uint32_t(-895109107),
                             uint32_t(-2094846927)};
    constexpr uint32_t b[4]={uint32_t(-1403630843),1192405134u,568162667u,878960812u};
    float u[4];
    #pragma unroll
    for(int i=0;i<4;++i){
        uint32_t state=hashed*a[i]+b[i];
        uint32_t value=((state>>((state>>28)+4))^state)*277803737u;
        uint32_t sample=(value>>30)^(value>>8);
        u[i]=float(sample+1u)*0x1p-24f;
    }
    const float radius0=sqrtf(-2.f*logf(u[0]));
    const float radius1=sqrtf(-2.f*logf(u[2]));
    n[0]=radius0*cosf(u[1]*6.2831854820251465f);
    n[1]=radius0*sinf(u[1]*6.2831854820251465f);
    n[2]=radius1*cosf(u[3]*6.2831854820251465f);
}

__global__ void pre_project_pack_v3(const NRFrameBindingsV3* bindings,
    const _Float16* weight,uint8_t* packed,int padded_width,int padded_height,
    float color_scale,float c0,float c1,float c2,float c3,float c4,uint32_t seed_xor) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t pixels=uint64_t(padded_width)*padded_height;
    if(i>=pixels*32)return;
    const uint64_t pixel=i/32;const int out_channel=int(i%32);
    const int x=int(pixel%padded_width),y=int(pixel/padded_width);
    const int width=int(bindings->render_width),height=int(bindings->render_height);
    const auto* color=static_cast<const _Float16*>(bindings->current_color);
    if(!color||width<=0||height<=0)return;
    int rx=width==1?0:x%(2*width-2),ry=height==1?0:y%(2*height-2);
    rx=rx<width?rx:2*width-2-rx;ry=ry<height?ry:2*height-2-ry;
    float noise[3];positional_noise(x,y,uint32_t(bindings->frame_id)^seed_xor,noise);
    const float constants[5]={c0,c1,c2,c3,c4};
    float sum=0.f;
    #pragma clang loop unroll(disable)
    for(int k=0;k<16;++k){
        _Float16 feature;
        if(k<3)feature=hround(noise[k]);
        else if(k==3)feature=hround(1.f);
        else if(k<10){
            const float source=float(color[(uint64_t(ry)*width+rx)*4+(k-4)%3]);
            feature=hround(float(hround(source-.5f))*color_scale);
        }else if(k<15)feature=hround(constants[k-10]);
        else feature=hround(0.f);
        sum+=float(feature)*float(weight[k*32+out_channel]);
    }
    packed[packed_c32_offset(y,x,out_channel,padded_width)]=encode_e4(float(hround(sum)));
}

__device__ __forceinline__ uint64_t pre_window_offset(
    int y,int x,int channel,int width) {
    const int window=(y/8)*(width/8)+x/8;
    const int token=((y&7)/4*2+(x&7)/4)*16+(y&3)*4+(x&3);
    return (uint64_t(window)*64+token)*32+channel;
}

__global__ void pre_pool_fp8(const _Float16* source,uint8_t* skip,uint8_t* target,
    int width,int height) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int target_width=width/2,target_height=height/2;
    const uint64_t values=uint64_t(target_width)*target_height*32;
    if(i>=values)return;
    const int channel=int(i%32);const uint64_t pixel=i/32;
    const int x=int(pixel%target_width),y=int(pixel/target_width);
    _Float16 p[4];
    p[0]=source[pre_window_offset(y*2,x*2,channel,width)];
    p[1]=source[pre_window_offset(y*2,x*2+1,channel,width)];
    p[2]=source[pre_window_offset(y*2+1,x*2,channel,width)];
    p[3]=source[pre_window_offset(y*2+1,x*2+1,channel,width)];
    skip[packed_c32_offset(y*2,x*2,channel,width)]=encode_e4(float(p[0]));
    skip[packed_c32_offset(y*2,x*2+1,channel,width)]=encode_e4(float(p[1]));
    skip[packed_c32_offset(y*2+1,x*2,channel,width)]=encode_e4(float(p[2]));
    skip[packed_c32_offset(y*2+1,x*2+1,channel,width)]=encode_e4(float(p[3]));
    const _Float16 upper=hround(float(p[0])+float(p[1]));
    const _Float16 lower=hround(float(p[2])+float(p[3]));
    const _Float16 sum=hround(float(upper)+float(lower));
    target[packed_c32_offset(y,x,channel,target_width)]=
        encode_e4(float(hround(float(sum)*.25f)));
}

__global__ void head_input_fp8(const uint8_t* main,const uint8_t* skip,
    const _Float16* main_scale,const _Float16* skip_scale,uint8_t* out,
    int width,int height,uint64_t values) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=values)return;
    const int local=int(i%2048),cta=int(i/2048),gx=(width+11)/8;
    const int pair=local&1;int within=(local>>1)%8;const int lane=(local>>4)%32;
    const int cell=local/512;
    const int order[8]={0,2,1,3,4,6,5,7},bases[8]={0,1,4,5,2,3,6,7};
    const int skip_order[8]={0,4,2,6,8,12,10,14};
    const int operation=cell*8+order[within];within=operation%8;
    const int source=bases[(operation>=16)*4+within/2];
    const int output=operation%16/8+(within&1)*2;
    const int q=(lane&19)|((lane<<1)&8)|((lane>>1)&4);
    const int selected=output^((lane&16)?2:0)^((lane&4)?1:0);
    const int source_lane=q^(selected*8),load=source/2;
    const int main_width=width/2,main_height=height/2;
    const int x=(-4+cta%gx*8)/2+source_lane/4%4;
    const int y=(-4+cta/gx*8)/2+source_lane/16+(load&1)*2;
    _Float16 a=hround(0.f),b=hround(0.f);
    if(x>=0&&x<main_width&&y>=0&&y<main_height){
        const int physical=source_lane%4*4+(source&1)*2+pair;
        const int logical=(load/2)*16+physical/4*2+(physical%4/2)*8+(physical&1);
        a=decode_e4(main[packed_c32_offset(y,x,logical,main_width)]);
    }
    const int sx=(-4+cta%gx*8)/4+cell%2,sy=(-4+cta/gx*8)/4+cell/2;
    if(sx>=0&&sx<width/4&&sy>=0&&sy<height/4)
        b=decode_e4(skip[(uint64_t(sy)*(width/4)+sx)*512+lane*16+
                         skip_order[within]+pair]);
    const int scale_index=(operation/2%4)*8+lane%4*2+pair;
    const _Float16 first=hround(float(a)*float(main_scale[scale_index]));
    const _Float16 second=hround(float(b)*float(skip_scale[scale_index]));
    out[i]=encode_e4(float(hround(float(first)+float(second))));
}

__global__ void head_tail_v3(const _Float16* input,const _Float16* weight,
    const NRFrameBindingsV3* bindings,uint64_t rows,int padded_width) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*4)return;
    auto* output=static_cast<_Float16*>(bindings->output_residual);if(!output)return;
    const uint64_t row=i/4;const int col=int(i%4);float first=0.f,second=0.f;
    #pragma clang loop unroll(disable)
    for(int k=0;k<16;++k)first+=float(input[row*32+k])*float(weight[k*4+col]);
    const _Float16 rounded=hround(first);
    #pragma clang loop unroll(disable)
    for(int k=16;k<32;++k)second+=float(input[row*32+k])*float(weight[k*4+col]);
    const uint64_t window=row/64;const int token=int(row%64);
    const int group=token/16,within=token%16,gx=(padded_width+11)/8;
    const int x=int(window%gx)*8-4+(group&1)*4+within%4;
    const int y=int(window/gx)*8-4+(group>>1)*4+within/4;
    if(x>=0&&x<int(bindings->output_width)&&y>=0&&y<int(bindings->output_height))
        output[(uint64_t(y)*bindings->output_width+x)*4+col]=
            hround(float(rounded)+second);
}

} // namespace

extern "C" __declspec(dllexport) int nr_io_pre_project_pack_v3(
    const void* bindings,const void* weight,void* packed,int padded_width,
    int padded_height,float color_scale,const float* conditioning,uint32_t seed_xor,
    void* stream) {
    if(!bindings||!weight||!packed||!conditioning||padded_width<=0||padded_height<=0||
       (padded_width&127)||(padded_height&127))return int(hipErrorInvalidValue);
    const uint64_t values=uint64_t(padded_width)*padded_height*32;
    hipLaunchKernelGGL(pre_project_pack_v3,dim3(unsigned((values+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const NRFrameBindingsV3*>(bindings),
        static_cast<const _Float16*>(weight),static_cast<uint8_t*>(packed),padded_width,
        padded_height,color_scale,conditioning[0],conditioning[1],conditioning[2],
        conditioning[3],conditioning[4],seed_xor);
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_io_pre_pool_fp8(const void* source,
    void* skip,void* target,int width,int height,void* stream) {
    if(!source||!skip||!target||width<=0||height<=0||(width&127)||(height&127))
        return int(hipErrorInvalidValue);
    const uint64_t values=uint64_t(width/2)*(height/2)*32;
    hipLaunchKernelGGL(pre_pool_fp8,dim3(unsigned((values+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(source),
        static_cast<uint8_t*>(skip),static_cast<uint8_t*>(target),width,height);
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_io_head_input_fp8(const void* main,
    const void* skip,const void* main_scale,const void* skip_scale,void* output,
    int width,int height,void* stream) {
    if(!main||!skip||!main_scale||!skip_scale||!output||width<=0||height<=0||
       (width&127)||(height&127))return int(hipErrorInvalidValue);
    const uint64_t windows=uint64_t((width+11)/8)*((height+11)/8);
    const uint64_t values=windows*2048;
    hipLaunchKernelGGL(head_input_fp8,dim3(unsigned((values+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint8_t*>(main),
        static_cast<const uint8_t*>(skip),static_cast<const _Float16*>(main_scale),
        static_cast<const _Float16*>(skip_scale),static_cast<uint8_t*>(output),
        width,height,values);
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_io_head_tail_v3(const void* input,
    const void* weight,const void* bindings,uint64_t rows,int padded_width,void* stream) {
    if(!input||!weight||!bindings||!rows||padded_width<=0||(padded_width&127))
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(head_tail_v3,dim3(unsigned((rows*4+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(input),
        static_cast<const _Float16*>(weight),static_cast<const NRFrameBindingsV3*>(bindings),
        rows,padded_width);
    return int(hipGetLastError());
}
