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

// Head input geometry, two gathers, zero padding, scales and residual in one
// launch. Preserve the intermediate half product and separate float multiply/add.
__global__ static void head_input_kernel(const __half* main,const __half* skip,
    const __half* ms,const __half* ss,__half* out,int width,int height,int start,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    int local=i%2048,cta=start+int(i/2048),gx=(width+11)/8;
    int pair=local&1,within=(local>>1)%8,lane=(local>>4)%32,cell=local/512;
    const int order[8]={0,2,1,3,4,6,5,7},bases[8]={0,1,4,5,2,3,6,7};
    const int skiporder[8]={0,4,2,6,8,12,10,14};
    int operation=cell*8+order[within];within=operation%8;
    int source=bases[(operation>=16)*4+within/2];
    int output=operation%16/8+(within&1)*2;
    int q=(lane&19)|((lane<<1)&8)|((lane>>1)&4);
    int selected=output^((lane&16)?2:0)^((lane&4)?1:0),sl=q^(selected*8),load=source/2;
    int x=(-4+cta%gx*8)/2+sl/4%4,y=(-4+cta/gx*8)/2+sl/16+(load&1)*2;
    float a=0,b=0;
    if(x>=0&&x<width/2&&y>=0&&y<height/2)
        a=__half2float(main[((size_t(load/2)*(height/2)+y)*(width/2)+x)*16+sl%4*4+(source&1)*2+pair]);
    int sx=(-4+cta%gx*8)/4+cell%2,sy=(-4+cta/gx*8)/4+cell/2;
    if(sx>=0&&sx<width/4&&sy>=0&&sy<height/4)
        b=__half2float(skip[(size_t(sy)*(width/4)+sx)*512+lane*16+skiporder[within]+pair]);
    int si=(operation/2%4)*8+lane%4*2+pair;
    float sm=__half2float(ms[si]),sb=__half2float(ss[si]);
    float first=__half2float(__float2half_rn(a*sm)),second=b*sb;
    __half result=__float2half_rn(first+second);
    // HIP optimization of padded zero products can discard signed zero. The
    // reference preserves it through both stores: -0 + -0 must remain -0.
    if(first==0.0f && second==0.0f) {
        unsigned sign=((__float_as_uint(a)^__float_as_uint(sm)) &
                       (__float_as_uint(b)^__float_as_uint(sb)))>>16 & 0x8000;
        result=__ushort_as_half(static_cast<unsigned short>(sign));
    }
    out[i]=result;
}
extern "C" __declspec(dllexport) int native_fusion_head_input(
    const void* main,const void* skip,const void* ms,const void* ss,void* out,
    int width,int height,int start,int count,void* stream) {
    if(!main||!skip||!ms||!ss||!out||width<=0||height<=0||width%8||height%8||start<0||count<=0
       ||int64_t(start)+count>int64_t((width+11)/8)*((height+11)/8))return int(hipErrorInvalidValue);
    size_t n=size_t(count)*2048;
    hipLaunchKernelGGL(head_input_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const __half*>(main),static_cast<const __half*>(skip),
        static_cast<const __half*>(ms),static_cast<const __half*>(ss),static_cast<__half*>(out),width,height,start,n);
    return int(hipGetLastError());
}

__global__ static void head_compose_kernel(const __half* residual,const __half* base,__half* out,
    int width,int height,int padded_width) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(width)*height*4)return;
    int c=i%4,x=(i/4)%width,y=i/(size_t(width)*4);
    if(c==3){out[i]=__float2half_rn(1.0f);return;}
    int lx=(x+4)%8,ly=(y+4)%8,token=(ly/4*2+lx/4)*16+ly%4*4+lx%4;
    size_t cta=size_t((y+4)/8)*((padded_width+11)/8)+(x+4)/8;
    float value=__half2float(base[i])+__half2float(residual[(cta*64+token)*4+c])*0.25f;
    value=value<0.0f?0.0f:(value>1.0f?1.0f:value);
    out[i]=__float2half_rn(value);
}
extern "C" __declspec(dllexport) int native_fusion_head_compose(const void* residual,const void* base,
    void* out,int width,int height,int padded_width,void* stream) {
    if(!residual||!base||!out||width<=0||height<=0||padded_width<width||padded_width%8)return int(hipErrorInvalidValue);
    size_t n=size_t(width)*height*4;
    hipLaunchKernelGGL(head_compose_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const __half*>(residual),static_cast<const __half*>(base),
        static_cast<__half*>(out),width,height,padded_width);
    return int(hipGetLastError());
}

__global__ static void head_gather_e4_kernel(const uint16_t* source,const int64_t* index,uint16_t* out,
    size_t n,int columns,int indices) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)out[i]=quantize_bits(source[(i/indices)*columns+index[i%indices]]);
}
extern "C" __declspec(dllexport) int native_fusion_head_gather_e4(const void* source,const void* index,
    void* out,uint64_t n,int columns,int indices,void* stream) {
    if(!source||!index||!out||!n||columns<=0||indices<=0||n%indices)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(head_gather_e4_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const uint16_t*>(source),static_cast<const int64_t*>(index),
        static_cast<uint16_t*>(out),size_t(n),columns,indices);
    return int(hipGetLastError());
}

__global__ static void pre_features_kernel(const __half* color,const __half* noise,const __half* constants,
    __half* out,int width,int height,int pw,size_t pixels,float scale) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=pixels*16)return;
    size_t p=i/16;int channel=i%16,x=p%pw,y=p/pw;
    if(channel<3){out[i]=noise[p*3+channel];return;}
    if(channel==3||channel==15){out[i]=__float2half_rn(channel==3?1.0f:0.0f);return;}
    if(channel>=10){out[i]=constants[channel-10];return;}
    int rx=width==1?0:x%(2*width-2),ry=height==1?0:y%(2*height-2);
    rx=rx<width?rx:2*width-2-rx;ry=ry<height?ry:2*height-2-ry;
    float centered=__half2float(__float2half_rn(__half2float(color[(size_t(ry)*width+rx)*4+(channel-4)%3])-0.5f));
    out[i]=__float2half_rn(centered*scale);
}
extern "C" __declspec(dllexport) int native_fusion_pre_features(const void* color,const void* noise,const void* constants,
    void* out,int width,int height,int pw,int ph,float scale,void* stream) {
    if(!color||!noise||!constants||!out||width<=0||height<=0||pw<width||ph<height)return int(hipErrorInvalidValue);
    size_t pixels=size_t(pw)*ph;
    hipLaunchKernelGGL(pre_features_kernel,dim3(unsigned((pixels*16+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const __half*>(color),static_cast<const __half*>(noise),
        static_cast<const __half*>(constants),static_cast<__half*>(out),width,height,pw,pixels,scale);
    return int(hipGetLastError());
}

// Build the authored 16-channel input vector, perform the FP16 16x32
// projection, and write the final packed C32 cell layout in one pass. Noise is
// still supplied by the validated generator, so this changes neither its hash
// nor its special-function semantics.
__global__ static void pre_project_pack_kernel(const __half* color,const __half* noise,
    const __half* constants,const __half* weight,__half* packed,int width,int height,int pw,size_t pixels,float scale) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=pixels*32)return;
    size_t pixel=i/32;int channel=i%32,x=int(pixel%pw),y=int(pixel/pw);
    int rx=width==1?0:x%(2*width-2),ry=height==1?0:y%(2*height-2);
    rx=rx<width?rx:2*width-2-rx;ry=ry<height?ry:2*height-2-ry;
    float sum=0.f;
    for(int k=0;k<16;++k) {
        __half feature;
        if(k<3)feature=noise[pixel*3+k];
        else if(k==3)feature=__float2half_rn(1.f);
        else if(k<10) {
            float source=__half2float(color[(size_t(ry)*width+rx)*4+(k-4)%3]);
            float centered=__half2float(__float2half_rn(source-.5f));
            feature=__float2half_rn(centered*scale);
        } else if(k<15)feature=constants[k-10];
        else feature=__float2half_rn(0.f);
        sum+=__half2float(feature)*__half2float(weight[k*32+channel]);
    }
    int token=(y%4)*4+x%4,row=token%16,col=channel%32;
    int mma=col/16*2+(col%16)/8,pairs[8]={0,2,1,3,4,6,5,7};
    int element=(row>=8)*2+(col&1),operation=pairs[mma*2+element/2];
    int local=(row%8*4+col%8/2)*16+operation*2+(element&1);
    size_t cell=(size_t(y/4)*(pw/4)+x/4)*512;
    packed[cell+local]=__float2half_rn(sum);
}
extern "C" __declspec(dllexport) int native_fusion_pre_project_pack(const void* color,const void* noise,
    const void* constants,const void* weight,void* packed,int width,int height,int pw,int ph,float scale,void* stream) {
    if(!color||!noise||!constants||!weight||!packed||width<=0||height<=0||pw<width||ph<height||pw%4||ph%4)
        return int(hipErrorInvalidValue);
    size_t pixels=size_t(pw)*ph;
    hipLaunchKernelGGL(pre_project_pack_kernel,dim3(unsigned((pixels*32+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const __half*>(color),static_cast<const __half*>(noise),
        static_cast<const __half*>(constants),static_cast<const __half*>(weight),static_cast<__half*>(packed),
        width,height,pw,pixels,scale);
    return int(hipGetLastError());
}

__global__ static void c32_gather_kernel(const uint16_t* input,uint16_t* out,int width,int height,int ox,int oy,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    int cta=i/2048,cell=i%2048/512,local=i%512,gx=(width-ox+7)/8;
    int x=(cta%gx*8+ox)/4+cell%2,y=(cta/gx*8+oy)/4+cell/2;
    out[i]=(x>=0&&x<width/4&&y>=0&&y<height/4)?input[(size_t(y)*(width/4)+x)*512+local]:0;
}
__global__ static void c32_scatter_kernel(const uint16_t* input,uint16_t* out,int width,int height,int ox,int oy,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    int x=(i/512)%(width/4),y=(i/512)/(width/4),local=i%512,lane=local/16,e=local%16;
    int tx=x-ox/4,ty=y-oy/4,gx=(width-ox+7)/8;
    int cta=ty/2*gx+tx/2,cell=(ty%2)*2+tx%2;
    int row=lane/4+e%8/4*8,col=lane%4*2+e%4/2*8+e/8*16+(e&1);
    out[i]=input[(size_t(cta)*64+cell*16+row)*32+col];
}
extern "C" __declspec(dllexport) int native_fusion_c32_layout(const void* input,void* out,int width,int height,int ox,int oy,int scatter,void* stream) {
    if(!input||!out||width<=0||height<=0||width%4||height%4||(ox!=0&&ox!=-4)||(oy!=0&&oy!=-4))return int(hipErrorInvalidValue);
    size_t n=scatter?size_t(width)*height*32:size_t((width-ox+7)/8)*((height-oy+7)/8)*2048;
    if(scatter) {
        hipLaunchKernelGGL(c32_scatter_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,reinterpret_cast<hipStream_t>(stream),
            static_cast<const uint16_t*>(input),static_cast<uint16_t*>(out),width,height,ox,oy,n);
    } else {
        hipLaunchKernelGGL(c32_gather_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,reinterpret_cast<hipStream_t>(stream),
            static_cast<const uint16_t*>(input),static_cast<uint16_t*>(out),width,height,ox,oy,n);
    }
    return int(hipGetLastError());
}

// Whole-grid packed-cell <-> Swin-window adapter for all recovered image
// channel families.  A packed image stores one contiguous 16*C cell; a window
// contains four cells.  The adapter changes only scheduling/layout and never
// changes a numeric value.  C32 keeps the original export above for ABI
// compatibility while wider stages opt in through this explicit entry point.
__global__ static void packed_gather_kernel(const uint16_t* input,uint16_t* out,
    int width,int height,int channels,int ox,int oy,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    size_t window_elements=size_t(64)*channels,cell_elements=size_t(16)*channels;
    int cta=int(i/window_elements),cell=int((i%window_elements)/cell_elements);
    size_t local=i%cell_elements;int gx=(width-ox+7)/8;
    int x=(cta%gx*8+ox)/4+cell%2,y=(cta/gx*8+oy)/4+cell/2;
    out[i]=(x>=0&&x<width/4&&y>=0&&y<height/4)?
        input[(size_t(y)*(width/4)+x)*cell_elements+local]:0;
}
__global__ static void packed_scatter_kernel(const uint16_t* input,uint16_t* out,
    int width,int height,int channels,int ox,int oy,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    size_t cell_elements=size_t(16)*channels,cell_index=i/cell_elements;
    int x=int(cell_index%(width/4)),y=int(cell_index/(width/4));
    int local=int(i%cell_elements),head=local/512,within=local%512;
    int lane=within/16,e=within%16,tx=x-ox/4,ty=y-oy/4,gx=(width-ox+7)/8;
    int cta=ty/2*gx+tx/2,cell=(ty%2)*2+tx%2;
    int row=lane/4+e%8/4*8,col=head*32+lane%4*2+e%4/2*8+e/8*16+(e&1);
    out[i]=input[(size_t(cta)*64+cell*16+row)*channels+col];
}
extern "C" __declspec(dllexport) int native_fusion_packed_layout(const void* input,void* out,
    int width,int height,int channels,int ox,int oy,int scatter,void* stream) {
    if(!input||!out||width<=0||height<=0||width%4||height%4||
       (channels!=32&&channels!=64&&channels!=128&&channels!=256&&channels!=512)||
       (ox!=0&&ox!=-4)||(oy!=0&&oy!=-4))return int(hipErrorInvalidValue);
    size_t n=scatter?size_t(width)*height*channels:
        size_t((width-ox+7)/8)*((height-oy+7)/8)*64*channels;
    auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    if(scatter)hipLaunchKernelGGL(packed_scatter_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        hip_stream,static_cast<const uint16_t*>(input),static_cast<uint16_t*>(out),
        width,height,channels,ox,oy,n);
    else hipLaunchKernelGGL(packed_gather_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,
        hip_stream,static_cast<const uint16_t*>(input),static_cast<uint16_t*>(out),
        width,height,channels,ox,oy,n);
    return int(hipGetLastError());
}

// GEMM result boundary: optional half-rounded scaled seed + FP32 product -> half.
// GEMM accumulation itself remains with the reference library implementation.
__global__ static void head_add_kernel(const __half* seed,const float* product,const __half* scale,
    __half* out,size_t n,int channels) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    float first=__half2float(seed[i]);
    if(scale) {
        float s=__half2float(scale[i%channels]);
        float rounded=__half2float(__float2half_rn(first*s));
        if(rounded==0.0f)rounded=__uint_as_float((__float_as_uint(first)^__float_as_uint(s))&0x80000000u);
        first=rounded;
    }
    float second=product[i],sum=first+second;
    if(first==0.0f&&second==0.0f)sum=__uint_as_float(__float_as_uint(first)&__float_as_uint(second)&0x80000000u);
    out[i]=__float2half_rn(sum);
}
extern "C" __declspec(dllexport) int native_fusion_head_add(const void* seed,const void* product,const void* scale,
    void* out,uint64_t n,int channels,void* stream) {
    if(!seed||!product||!out||!n||channels<=0||n%channels)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(head_add_kernel,dim3(unsigned((n+255)/256)),dim3(256),0,reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(seed),static_cast<const float*>(product),static_cast<const __half*>(scale),
        static_cast<__half*>(out),size_t(n),channels);
    return int(hipGetLastError());
}

__global__ static void head_qkv_kernel(const __half* projection,const __half* qscale,
    uint16_t* q,uint16_t* k,uint16_t* v,size_t rows) {
    size_t warp=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32,row=warp/3;
    int part=warp%3,lane=threadIdx.x%32;if(row>=rows)return;
    uint16_t bits=__half_as_ushort(projection[row*96+part*32+lane]);
    if(part==2){v[row*32+lane]=quantize_bits(bits);return;}
    float t=__half2float(__ushort_as_half(bits));
    float sum=__half2float(__float2half_rn(t*t));
    for(int offset=1;offset<32;offset*=2) {
        float other=__shfl_down(sum,offset,32);
        if((lane%(2*offset))==0)sum=__half2float(__float2half_rn(sum+other));
    }
    sum=__shfl(sum,0,32);
    float clamped=sum<6.198883056640625e-5f?6.198883056640625e-5f:sum;
    float inv=__half2float(__float2half_rn(rsqrtf(clamped)));
    __half normalized=__float2half_rn(t*inv);
    if(part==0)normalized=__float2half_rn(__half2float(normalized)*__half2float(*qscale));
    (part==0?q:k)[row*32+lane]=quantize_bits(__half_as_ushort(normalized));
}
extern "C" __declspec(dllexport) int native_fusion_head_qkv(const void* projection,const void* scale,
    void* q,void* k,void* v,uint64_t rows,void* stream) {
    if(!projection||!scale||!q||!k||!v||!rows)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(head_qkv_kernel,dim3(unsigned((rows*96+255)/256)),dim3(256),0,reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(projection),static_cast<const __half*>(scale),static_cast<uint16_t*>(q),
        static_cast<uint16_t*>(k),static_cast<uint16_t*>(v),size_t(rows));
    return int(hipGetLastError());
}
