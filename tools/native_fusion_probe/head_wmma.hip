#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <cstdint>

using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using I2 = int __attribute__((ext_vector_type(2)));
__device__ static _Float16 rnd(float x) { return (_Float16)__float2half_rn(x); }
__device__ static unsigned char e4(float x) {return __hip_cvt_float_to_fp8(x,__HIP_SATFINITE,__HIP_E4M3);}
__device__ static _Float16 quant(_Float16 x) { return (_Float16)__half(__hip_cvt_fp8_to_halfraw(e4(float(x)),__HIP_E4M3)); }
__device__ static _Float16 cubic(_Float16 hx) {
    float x=float(hx),v=fminf(4.f,fmaxf(-4.f,x));
    _Float16 a=rnd(-0.055908203125f*fabsf(v)+0.447265625f);
    _Float16 b=rnd(v*float(a)+0.89453125f);
    return quant(rnd(x*float(b)));
}
template<bool FP8> __device__ static F8 mma(H8 a,H8 b,F8 c) {
    if constexpr(FP8) {
        I2 ap={},bp={};
        for(int i=0;i<8;++i) {ap[i/4]|=unsigned(e4(float(a[i])))<<((i%4)*8);bp[i/4]|=unsigned(e4(float(b[i])))<<((i%4)*8);}
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ap,bp,c);
    } else return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c);
}

// One wave processes 16 tokens. LDS only holds one 32-channel hidden slice
// per wave; no cross-workgroup dependencies or full expanded global tensor.
template<int WAVES,bool FP8> __global__ void head_ffn_wmma(
    const _Float16* raw,const _Float16* expand,const _Float16* contract,
    const _Float16* scale,const int64_t* ai,const int64_t* ri,const int64_t* perm,
    _Float16* output,int windows) {
    __shared__ _Float16 hidden[WAVES][32*16];
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;
    int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;
    bool active=window<windows;
    H8 result[2];
    for(int m=0;m<2;++m) for(int e=0;e<8;++e) {
        int c=m*16+group*8+e;
        result[m][e]=active?rnd(float(raw[window*2048+ri[row*32+c]])*float(scale[c])):0;
    }
    for(int p=0;p<4;++p) {
        for(int m=0;m<2;++m) {
            F8 acc={};
            for(int k=0;k<2;++k) {
                H8 a,b;
                for(int e=0;e<8;++e) {
                    int kk=k*16+group*8+e;
                    a[e]=expand[p*1024+kk*32+m*16+token];
                    b[e]=active?quant(raw[window*2048+ai[row*32+kk]]):0;
                }
                acc=mma<FP8>(a,b,acc);
            }
            for(int e=0;e<8;++e)hidden[wave][(m*16+group*8+e)*16+token]=cubic(rnd(acc[e]));
        }
        __syncthreads();
        for(int m=0;m<2;++m) {
            F8 acc={};
            for(int k=0;k<2;++k) {
                H8 a,b;
                for(int e=0;e<8;++e) {
                    int kk=k*16+group*8+e;
                    a[e]=contract[p*1024+kk*32+m*16+token];
                    b[e]=hidden[wave][int(perm[kk])*16+token];
                }
                acc=mma<FP8>(a,b,acc);
            }
            for(int e=0;e<8;++e)result[m][e]=rnd(float(result[m][e])+acc[e]);
        }
        __syncthreads();
    }
    if(active)for(int m=0;m<2;++m)for(int e=0;e<8;++e)
        output[(window*64+row)*32+m*16+group*8+e]=result[m][e];
}

// Attention projection boundary. The preceding QK/softmax/PV reduction is not
// changed here: this kernel only consumes its decoded half result, performs the
// model-authored permutation/E4 conversion, 32x32 projection and scaled skip.
template<int WAVES,bool FP8> __global__ void head_project_wmma(
    const _Float16* first,const _Float16* attention,const _Float16* weight,
    const _Float16* scale,const int64_t* perm,_Float16* output,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;
    int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;
    bool active=window<windows;
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int k=0;k<2;++k) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=k*16+group*8+e;
                a[e]=weight[kk*32+m*16+token];
                b[e]=active?quant(attention[(window*64+row)*32+perm[kk]]):0;
            }
            acc=mma<FP8>(a,b,acc);
        }
        if(active)for(int e=0;e<8;++e) {
            int c=m*16+group*8+e;
            _Float16 seed=rnd(float(first[(window*64+row)*32+c])*float(scale[c]));
            output[(window*64+row)*32+c]=rnd(float(seed)+acc[e]);
        }
    }
}

// QK keeps the reference's FP32 dot, FP32 position bias add, then half store.
// Each wave owns 16 query rows and iterates the four 16-key output tiles.
template<int WAVES,bool FP8> __global__ void head_qk_wmma(
    const _Float16* q,const _Float16* k,const _Float16* bias,_Float16* scores,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,group=lane/16;
    int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+query;
    bool active=window<windows;
    for(int kt=0;kt<4;++kt) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e,key=kt*16+query;
                a[e]=active?k[(window*64+key)*32+kk]:0;
                b[e]=active?q[(window*64+row)*32+kk]:0;
            }
            acc=mma<FP8>(a,b,acc);
        }
        if(active)for(int e=0;e<8;++e) {
            int key=kt*16+group*8+e;
            scores[(window*64+row)*64+key]=rnd(acc[e]+float(bias[row*64+key]));
        }
    }
}

// QKV projection only. Normalization/scale/E4 remains in the already validated
// cooperative kernel so its half reduction tree is unchanged.
template<int WAVES,bool FP8> __global__ void head_qkv_project_wmma(
    const _Float16* input,const _Float16* weight,const int64_t* perm,
    _Float16* projection,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;
    int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;
    bool active=window<windows;
    for(int m=0;m<6;++m) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e;
                a[e]=weight[kk*96+m*16+token];
                b[e]=active?quant(input[(window*64+row)*32+perm[kk]]):0;
            }
            acc=mma<FP8>(a,b,acc);
        }
        if(active)for(int e=0;e<8;++e) {
            int c=m*16+group*8+e;
            projection[(window*64+row)*96+c]=rnd(acc[e]);
        }
    }
}

// PV has the model's explicit K=32 half boundary: first two WMMA fragments are
// rounded to half before the second pair is accumulated and added.
template<int WAVES,bool FP8> __global__ void head_pv_wmma(
    const _Float16* probability,const _Float16* v,_Float16* output,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,group=lane/16;
    int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+query;
    bool active=window<windows;
    for(int m=0;m<2;++m) {
        F8 first={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int key=chunk*16+group*8+e,channel=m*16+query;
                a[e]=active?v[(window*64+key)*32+channel]:0;
                b[e]=active?probability[(window*64+row)*64+key]:0;
            }
            first=mma<FP8>(a,b,first);
        }
        H8 seed;for(int e=0;e<8;++e)seed[e]=rnd(first[e]);
        F8 second={};
        for(int chunk=2;chunk<4;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int key=chunk*16+group*8+e,channel=m*16+query;
                a[e]=active?v[(window*64+key)*32+channel]:0;
                b[e]=active?probability[(window*64+row)*64+key]:0;
            }
            second=mma<FP8>(a,b,second);
        }
        if(active)for(int e=0;e<8;++e) {
            int channel=m*16+group*8+e;
            output[(window*64+row)*32+channel]=rnd(float(seed[e])+second[e]);
        }
    }
}

// The recovered tail is two explicit K=16 FP32 products with a half store in
// between. Keep that boundary and write the four residual channels directly.
__global__ void head_tail_exact(const _Float16* input,const _Float16* weight,
    _Float16* output,size_t rows) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*4)return;
    size_t row=i/4;int col=i%4;
    float first=0.f,second=0.f;
    for(int k=0;k<16;++k)first+=float(input[row*32+k])*float(weight[k*4+col]);
    _Float16 rounded=rnd(first);
    for(int k=16;k<32;++k)second+=float(input[row*32+k])*float(weight[k*4+col]);
    output[i]=rnd(float(rounded)+second);
}

// Explicit experimental softmax boundary. Whether this reduction/exp ordering
// matches the tensor-library reference is decided by the strict GPU gate.
__global__ void head_softmax_e4(const _Float16* scores,_Float16* probability,size_t rows) {
    size_t row=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;int lane=threadIdx.x%32;
    if(row>=rows)return;
    float x0=float(scores[row*64+lane]),x1=float(scores[row*64+lane+32]);
    float maximum=fmaxf(x0,x1);
    for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,offset,32));
    maximum=__shfl(maximum,0,32);
    float e0=expf(x0-maximum),e1=expf(x1-maximum),sum=e0+e1;
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down(sum,offset,32);
    sum=__shfl(sum,0,32);
    probability[row*64+lane]=quant(rnd(e0/sum));
    probability[row*64+lane+32]=quant(rnd(e1/sum));
}

// One workgroup owns one complete 8x8 Head window. Q/K/V and the 64x64
// attention matrix stay in bounded LDS for their entire lifetime, so whole-grid
// scheduling does not materialize full-frame projection, Q/K/V, score,
// probability, or value tensors in VRAM. Four wave32s each own 16 rows.
template<bool FP8,bool C32_RULES=false> __global__ void head_attention_window_fused(
    const _Float16* first,const _Float16* qkv_weight,const _Float16* qscale,
    const _Float16* position_bias,const _Float16* project_weight,
    const _Float16* attention_scale,const int64_t* perm,_Float16* output,int windows) {
    __shared__ _Float16 qkv[64*96];
    __shared__ _Float16 scores[64*64];
    int window=blockIdx.x,wave=threadIdx.x/32,lane=threadIdx.x%32;
    int token=lane%16,group=lane/16,row=wave*16+token;
    if(window>=windows)return;

    // Quantized gather + 32x96 QKV projection, preserving the half store.
    for(int m=0;m<6;++m) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e;
                a[e]=qkv_weight[kk*96+m*16+token];
                b[e]=quant(first[(window*64+row)*32+perm[kk]]);
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)qkv[row*96+m*16+group*8+e]=rnd(acc[e]);
    }
    __syncthreads();

    // Exact recovered half reduction tree. Each wave serially normalizes its
    // sixteen rows while all 32 lanes cover one vector.
    for(int local=0;local<16;++local) {
        int norm_row=wave*16+local;
        for(int part=0;part<3;++part) {
            _Float16 h=qkv[norm_row*96+part*32+lane];
            if(part==2) {
                qkv[norm_row*96+64+lane]=quant(h);
            } else {
                float t=float(h),sum=float(rnd(t*t));
                for(int offset=1;offset<32;offset*=2) {
                    float other=__shfl_down(sum,offset,32);
                    if((lane%(2*offset))==0)sum=float(rnd(sum+other));
                }
                sum=__shfl(sum,0,32);
                float inv=float(rnd(rsqrtf(fmaxf(sum,6.198883056640625e-5f))));
                _Float16 normalized=rnd(t*inv);
                if(part==0)normalized=rnd(float(normalized)*float(*qscale));
                qkv[norm_row*96+part*32+lane]=quant(normalized);
            }
        }
    }
    __syncthreads();

    // QK and position bias; layout and WMMA accumulation match head_qk_wmma.
    for(int kt=0;kt<4;++kt) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e,key=kt*16+token;
                a[e]=qkv[key*96+32+kk];b[e]=qkv[row*96+kk];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e) {
            int key=kt*16+group*8+e;
            if constexpr(C32_RULES)scores[row*64+key]=rnd(float(rnd(acc[e]))+float(position_bias[row*64+key]));
            else scores[row*64+key]=rnd(acc[e]+float(position_bias[row*64+key]));
        }
    }
    __syncthreads();

    // In-place E4M3 softmax. One wave handles sixteen complete query rows.
    for(int local=0;local<16;++local) {
        int query_row=wave*16+local;
        float x0=float(scores[query_row*64+lane]),x1=float(scores[query_row*64+lane+32]);
        float maximum=fmaxf(x0,x1);
        for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,offset,32));
        maximum=__shfl(maximum,0,32);
        float e0=expf(x0-maximum),e1=expf(x1-maximum),sum=e0+e1;
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down(sum,offset,32);
        sum=__shfl(sum,0,32);
        scores[query_row*64+lane]=quant(rnd(e0/sum));
        scores[query_row*64+lane+32]=quant(rnd(e1/sum));
    }
    __syncthreads();

    // PV overwrites the now-dead Q region in qkv. Head keeps its explicit K=32
    // half boundary; C32 native_fp16 uses one K64 accumulation.
    for(int m=0;m<2;++m) {
        if constexpr(C32_RULES) {
            F8 acc={};
            for(int chunk=0;chunk<4;++chunk) {
                H8 a,b;
                for(int e=0;e<8;++e) {
                    int key=chunk*16+group*8+e,channel=m*16+token;
                    a[e]=qkv[key*96+64+channel];b[e]=scores[row*64+key];
                }
                acc=mma<FP8>(a,b,acc);
            }
            for(int e=0;e<8;++e)qkv[row*96+m*16+group*8+e]=rnd(acc[e]);
        } else {
            F8 first_half={};
            for(int chunk=0;chunk<2;++chunk) {
                H8 a,b;
                for(int e=0;e<8;++e) {
                    int key=chunk*16+group*8+e,channel=m*16+token;
                    a[e]=qkv[key*96+64+channel];b[e]=scores[row*64+key];
                }
                first_half=mma<FP8>(a,b,first_half);
            }
            H8 seed;for(int e=0;e<8;++e)seed[e]=rnd(first_half[e]);
            F8 second_half={};
            for(int chunk=2;chunk<4;++chunk) {
                H8 a,b;
                for(int e=0;e<8;++e) {
                    int key=chunk*16+group*8+e,channel=m*16+token;
                    a[e]=qkv[key*96+64+channel];b[e]=scores[row*64+key];
                }
                second_half=mma<FP8>(a,b,second_half);
            }
            for(int e=0;e<8;++e)qkv[row*96+m*16+group*8+e]=rnd(float(seed[e])+second_half[e]);
        }
    }
    __syncthreads();

    // Attention projection and scaled residual, preserving both half boundaries.
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e;
                a[e]=project_weight[kk*32+m*16+token];
                b[e]=quant(qkv[row*96+perm[kk]]);
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e) {
            int channel=m*16+group*8+e;
            _Float16 seed=rnd(float(first[(window*64+row)*32+channel])*float(attention_scale[channel]));
            if constexpr(C32_RULES)output[(window*64+row)*32+channel]=rnd(float(seed)+float(rnd(acc[e])));
            else output[(window*64+row)*32+channel]=rnd(float(seed)+acc[e]);
        }
    }
}

// Pre input projection is the model's original FP16 16x32 matrix, not an FP8
// FFN boundary. One wave consumes 16 independent full-frame pixels.
template<int WAVES> __global__ void pre_project_wmma(
    const _Float16* input,const _Float16* weight,_Float16* output,size_t rows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;
    size_t row=(size_t(blockIdx.x)*WAVES+wave)*16+token;bool active=row<rows;
    for(int m=0;m<2;++m) {
        H8 a,b;F8 acc={};
        for(int e=0;e<8;++e) {
            int kk=group*8+e;a[e]=weight[kk*32+m*16+token];b[e]=active?input[row*16+kk]:0;
        }
        acc=mma<false>(a,b,acc);
        if(active)for(int e=0;e<8;++e)output[row*32+m*16+group*8+e]=rnd(acc[e]);
    }
}

// Generic recovered C32 FFN. Unlike Head, the native_fp16 reference performs
// one K=128 contraction then a half residual add, so keep that boundary.
template<int WAVES,bool FP8> __global__ void c32_ffn_wmma(
    const _Float16* raw,const _Float16* expand,const _Float16* contract,
    const _Float16* scale,const int64_t* ai,const int64_t* ri,const int64_t* perm,
    _Float16* output,int windows) {
    __shared__ _Float16 hidden[WAVES][128*16];
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;
    int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;bool active=window<windows;
    for(int m=0;m<8;++m) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e;a[e]=expand[kk*128+m*16+token];b[e]=active?quant(raw[window*2048+ai[row*32+kk]]):0;}
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)hidden[wave][(m*16+group*8+e)*16+token]=cubic(rnd(acc[e]));
    }
    __syncthreads();
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int part=0;part<4;++part)for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e,k=part*32+kk;a[e]=contract[k*32+m*16+token];b[e]=hidden[wave][(part*32+perm[kk])*16+token];}
            acc=mma<FP8>(a,b,acc);
        }
        if(active)for(int e=0;e<8;++e){int c=m*16+group*8+e;_Float16 seed=rnd(float(raw[window*2048+ri[row*32+c]])*float(scale[c]));output[(window*64+row)*32+c]=rnd(float(rnd(acc[e]))+float(seed));}
    }
}

template<int C,int WAVES,bool FP8> __global__ void wide_ffn_wmma(
    const _Float16* raw,const _Float16* expand,const _Float16* contract,const _Float16* mix,
    const _Float16* scale,const int64_t* ai,const int64_t* ri,const int64_t* perm,
    _Float16* output,int windows) {
    __shared__ _Float16 storage[WAVES][4*C*16];
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;bool active=window<windows;
    for(int m=0;m<4*C/16;++m){F8 acc={};for(int ck=0;ck<C/16;++ck){H8 a,b;for(int e=0;e<8;++e){int kk=ck*16+group*8+e;a[e]=expand[kk*(4*C)+m*16+token];b[e]=active?quant(raw[window*64*C+ai[row*C+kk]]):0;}acc=mma<FP8>(a,b,acc);}for(int e=0;e<8;++e)storage[wave][(m*16+group*8+e)*16+token]=cubic(rnd(acc[e]));}
    __syncthreads();
    // Keep both output fragments live until this head has consumed all 128
    // hidden channels.  Storing m=0 early aliases hidden[0:16] for the same
    // head and corrupted the subsequent m=1 contraction on C64/C128.
    for(int head=0;head<C/32;++head){
        F8 contracted[2]={};
        for(int m=0;m<2;++m)for(int part=0;part<4;++part)for(int chunk=0;chunk<2;++chunk){
            H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e,k=part*32+kk;a[e]=contract[(head*128+k)*32+m*16+token];b[e]=storage[wave][(head*128+part*32+perm[kk])*16+token];}
            contracted[m]=mma<FP8>(a,b,contracted[m]);
        }
        for(int m=0;m<2;++m)for(int e=0;e<8;++e)storage[wave][(head*32+m*16+group*8+e)*16+token]=rnd(contracted[m][e]);
    }
    __syncthreads();
    for(int m=0;m<C/16;++m){F8 acc={};for(int ck=0;ck<C/16;++ck){H8 a,b;for(int e=0;e<8;++e){int kk=ck*16+group*8+e;a[e]=mix[kk*C+m*16+token];b[e]=quant(storage[wave][(ck/2*32+perm[kk%32])*16+token]);}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e){int c=m*16+group*8+e;_Float16 seed=rnd(float(raw[window*64*C+ri[row*C+c]])*float(scale[c]));output[(window*64+row)*C+c]=rnd(float(rnd(acc[e]))+float(seed));}}
}

// Resident-weight FP8 primitive. The A operand is already stored as eight
// E4M3 bytes, so the hot kernel only converts the activation operand. This is
// intentionally narrower than a "fully resident FP8" claim: FP16 semantic
// boundaries and the library CxC mix remain unchanged.
__device__ static F8 mma_fp8_resident_a(I2 packed_a,H8 b,F8 c) {
    I2 packed_b={};
    for(int i=0;i<8;++i)
        packed_b[i/4]|=unsigned(e4(float(b[i])))<<((i%4)*8);
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
        packed_a,packed_b,c);
}

// Low-LDS grouped FFN stage for C64/C128/C256.  A workgroup owns exactly one
// (window, head, 16-token tile) task.  It computes C -> 128 -> 32 for that
// head, quantizes at the established pre-mix boundary, and writes the mix
// operand plus residual seed.  The cross-head C -> C mix intentionally stays
// in the caller's library GEMM instead of serializing every head in one wave.
template<int C,bool FP8> __global__ void wide_group_ffn_wmma(
    const _Float16* raw,const _Float16* expand,const _Float16* contract,
    const _Float16* scale,const int64_t* ai,const int64_t* ri,
    const int64_t* perm,const int64_t* inverse_perm,
    _Float16* grouped,_Float16* seed,int windows) {
    __shared__ _Float16 hidden[128*16];
    constexpr int HEADS=C/32;
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,head=task%HEADS,tile=task/HEADS;
    int window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    for(int m=0;m<8;++m) {
        F8 acc={};
        for(int chunk=0;chunk<C/16;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+half*8+e;
                int hidden_col=head*128+m*16+token;
                a[e]=expand[kk*(4*C)+hidden_col];
                b[e]=quant(raw[window*64*C+ai[row*C+kk]]);
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)
            hidden[(m*16+half*8+e)*16+token]=cubic(rnd(acc[e]));
    }
    __syncthreads();
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int part=0;part<4;++part)for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+half*8+e,k=part*32+kk;
                a[e]=contract[(head*128+k)*32+m*16+token];
                b[e]=hidden[(part*32+perm[kk])*16+token];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e) {
            int local=m*16+half*8+e,channel=head*32+local;
            int mix_input=head*32+inverse_perm[local];
            grouped[(window*64+row)*C+mix_input]=quant(rnd(acc[e]));
            seed[(window*64+row)*C+channel]=
                rnd(float(raw[window*64*C+ri[row*C+channel]])*float(scale[channel]));
        }
    }
}

// Same mathematical boundaries and grid as wide_group_ffn_wmma<...,true>,
// but expand/contract weights are initialized once as resident E4M3 bytes.
// The grouped result remains the established E4M3-rounded FP16 representation
// because the following library GEMM currently consumes FP16.
template<int C> __global__ void wide_group_ffn_fp8w(
    const _Float16* raw,const uint8_t* expand,const uint8_t* contract,
    const _Float16* scale,const int64_t* ai,const int64_t* ri,
    const int64_t* perm,const int64_t* inverse_perm,
    _Float16* grouped,_Float16* seed,int windows) {
    __shared__ _Float16 hidden[128*16];
    constexpr int HEADS=C/32;
    int lane=threadIdx.x,token=lane%16,half=lane/16;
    int task=blockIdx.x,head=task%HEADS,tile=task/HEADS;
    int window=tile/4,row=tile%4*16+token;
    if(window>=windows)return;
    for(int m=0;m<8;++m) {
        F8 acc={};
        for(int chunk=0;chunk<C/16;++chunk) {
            I2 packed_weight={};H8 activation;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+half*8+e;
                int hidden_col=head*128+m*16+token;
                packed_weight[e/4]|=unsigned(expand[kk*(4*C)+hidden_col])<<((e%4)*8);
                activation[e]=quant(raw[window*64*C+ai[row*C+kk]]);
            }
            acc=mma_fp8_resident_a(packed_weight,activation,acc);
        }
        for(int e=0;e<8;++e)
            hidden[(m*16+half*8+e)*16+token]=cubic(rnd(acc[e]));
    }
    __syncthreads();
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int part=0;part<4;++part)for(int chunk=0;chunk<2;++chunk) {
            I2 packed_weight={};H8 activation;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+half*8+e,k=part*32+kk;
                packed_weight[e/4]|=unsigned(contract[(head*128+k)*32+m*16+token])<<((e%4)*8);
                activation[e]=hidden[(part*32+perm[kk])*16+token];
            }
            acc=mma_fp8_resident_a(packed_weight,activation,acc);
        }
        for(int e=0;e<8;++e) {
            int local=m*16+half*8+e,channel=head*32+local;
            int mix_input=head*32+inverse_perm[local];
            grouped[(window*64+row)*C+mix_input]=quant(rnd(acc[e]));
            seed[(window*64+row)*C+channel]=
                rnd(float(raw[window*64*C+ri[row*C+channel]])*float(scale[channel]));
        }
    }
}

// C512 uses a different four-record block.  Only its eight independent
// 64->256->64 grouped MLPs are fused here; the surrounding 512x512 projections
// remain library GEMMs so this candidate does not require a full-window LDS
// allocation or duplicate the recovered split-block boundaries.
template<int WAVES,bool FP8> __global__ void c512_group_ffn_wmma(
    const _Float16* projected,const _Float16* expand,const _Float16* contract,
    const int64_t* perm64,const int64_t* perm256,_Float16* output,int windows) {
    __shared__ _Float16 hidden[WAVES][256*16];
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,half=lane/16;
    int task=blockIdx.x*WAVES+wave,head=task%8,tile=task/8;
    int window=tile/4,row=tile%4*16+token;bool active=window<windows;
    for(int m=0;m<16;++m){
        F8 acc={};
        for(int chunk=0;chunk<4;++chunk){
            H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+half*8+e;
                a[e]=expand[(head*64+kk)*256+m*16+token];
                b[e]=active?quant(projected[(window*64+row)*512+head*64+perm64[kk]]):0;}
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)hidden[wave][(m*16+half*8+e)*16+token]=cubic(rnd(acc[e]));
    }
    __syncthreads();
    for(int m=0;m<4;++m){
        F8 acc={};
        for(int chunk=0;chunk<16;++chunk){
            H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+half*8+e;
                a[e]=contract[(head*256+kk)*64+m*16+token];
                b[e]=hidden[wave][perm256[kk]*16+token];}
            acc=mma<FP8>(a,b,acc);
        }
        if(active)for(int e=0;e<8;++e){int c=m*16+half*8+e;
            output[(window*64+row)*512+head*64+c]=quant(rnd(acc[e]));}
    }
}

__global__ void c32_qkv_norm(const _Float16* projection,const _Float16* qscale,
    _Float16* q,_Float16* k,_Float16* v,size_t rows) {
    size_t warp=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32,row=warp/3;int part=warp%3,lane=threadIdx.x%32;if(row>=rows)return;
    _Float16 h=projection[row*96+part*32+lane];
    if(part==2){v[row*32+lane]=quant(h);return;}
    float t=float(h),sum=float(rnd(t*t));
    for(int offset=1;offset<32;offset*=2){float other=__shfl_down(sum,offset,32);if((lane%(2*offset))==0)sum=float(rnd(sum+other));}
    sum=__shfl(sum,0,32);float inv=float(rnd(rsqrtf(fmaxf(sum,6.198883056640625e-5f))));_Float16 n=rnd(t*inv);
    if(part==0)n=rnd(float(n)*float(*qscale));(part==0?q:k)[row*32+lane]=quant(n);
}

template<int WAVES,bool FP8> __global__ void c32_qk_wmma(const _Float16* q,const _Float16* k,const _Float16* bias,_Float16* scores,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,group=lane/16;int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+query;bool active=window<windows;
    for(int kt=0;kt<4;++kt){F8 acc={};for(int chunk=0;chunk<2;++chunk){H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e,key=kt*16+query;a[e]=active?k[(window*64+key)*32+kk]:0;b[e]=active?q[(window*64+row)*32+kk]:0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e){int key=kt*16+group*8+e;scores[(window*64+row)*64+key]=rnd(float(rnd(acc[e]))+float(bias[row*64+key]));}}
}

template<int WAVES,bool FP8> __global__ void c32_pv_wmma(const _Float16* probability,const _Float16* v,_Float16* output,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,group=lane/16;int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+query;bool active=window<windows;
    for(int m=0;m<2;++m){F8 acc={};for(int chunk=0;chunk<4;++chunk){H8 a,b;for(int e=0;e<8;++e){int key=chunk*16+group*8+e,channel=m*16+query;a[e]=active?v[(window*64+key)*32+channel]:0;b[e]=active?probability[(window*64+row)*64+key]:0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e){int c=m*16+group*8+e;output[(window*64+row)*32+c]=rnd(acc[e]);}}
}

template<int WAVES,bool FP8> __global__ void c32_project_wmma(const _Float16* post,const _Float16* value,const _Float16* weight,const _Float16* scale,const int64_t* perm,_Float16* output,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;bool active=window<windows;
    for(int m=0;m<2;++m){F8 acc={};for(int chunk=0;chunk<2;++chunk){H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e;a[e]=weight[kk*32+m*16+token];b[e]=active?quant(value[(window*64+row)*32+perm[kk]]):0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e){int c=m*16+group*8+e;_Float16 seed=rnd(float(post[(window*64+row)*32+c])*float(scale[c]));output[(window*64+row)*32+c]=rnd(float(rnd(acc[e]))+float(seed));}}
}

// Three-stage C32 attention candidate. Stage one keeps only the transient
// 64x96 projection in LDS, applies the recovered normalization tree, and emits
// normalized Q/K/V. This removes the global unnormalized projection without
// coupling its 12 KiB lifetime to the score matrix.
template<bool FP8> __global__ void c32_qkv_norm_staged(
    const _Float16* post,const _Float16* qkv_weight,const _Float16* qscale,
    const int64_t* perm,_Float16* q,_Float16* k,_Float16* v,int windows) {
    __shared__ _Float16 projection[64*96];
    int window=blockIdx.x,wave=threadIdx.x/32,lane=threadIdx.x%32;
    int token=lane%16,group=lane/16,row=wave*16+token;
    if(window>=windows)return;
    for(int m=0;m<6;++m) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e;
                a[e]=qkv_weight[kk*96+m*16+token];
                b[e]=quant(post[(window*64+row)*32+perm[kk]]);
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)projection[row*96+m*16+group*8+e]=rnd(acc[e]);
    }
    __syncthreads();
    for(int local=0;local<16;++local) {
        int norm_row=wave*16+local;
        for(int part=0;part<3;++part) {
            _Float16 h=projection[norm_row*96+part*32+lane];
            _Float16* destination=part==0?q:(part==1?k:v);
            if(part==2) {
                destination[(window*64+norm_row)*32+lane]=quant(h);
            } else {
                float t=float(h),sum=float(rnd(t*t));
                for(int offset=1;offset<32;offset*=2) {
                    float other=__shfl_down(sum,offset,32);
                    if((lane%(2*offset))==0)sum=float(rnd(sum+other));
                }
                sum=__shfl(sum,0,32);
                float inv=float(rnd(rsqrtf(fmaxf(sum,6.198883056640625e-5f))));
                _Float16 normalized=rnd(t*inv);
                if(part==0)normalized=rnd(float(normalized)*float(*qscale));
                destination[(window*64+norm_row)*32+lane]=quant(normalized);
            }
        }
    }
}

// Stage two retains only the 64x64 score/probability matrix (8 KiB) in LDS.
// Q/K/V are read from the prior resident workspace, and only the 64x32 value
// tensor is written. QK, E4M3 softmax and the single-K64 C32 PV boundaries are
// intentionally identical to the established six-dispatch implementation.
template<bool FP8> __global__ void c32_attention_core_staged(
    const _Float16* q,const _Float16* k,const _Float16* v,
    const _Float16* position_bias,_Float16* value,int windows) {
    __shared__ _Float16 scores[64*64];
    int window=blockIdx.x,wave=threadIdx.x/32,lane=threadIdx.x%32;
    int query=lane%16,group=lane/16,row=wave*16+query;
    if(window>=windows)return;
    for(int kt=0;kt<4;++kt) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e,key=kt*16+query;
                a[e]=k[(window*64+key)*32+kk];
                b[e]=q[(window*64+row)*32+kk];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e) {
            int key=kt*16+group*8+e;
            scores[row*64+key]=rnd(float(rnd(acc[e]))+float(position_bias[row*64+key]));
        }
    }
    __syncthreads();
    for(int local=0;local<16;++local) {
        int query_row=wave*16+local;
        float x0=float(scores[query_row*64+lane]),x1=float(scores[query_row*64+lane+32]);
        float maximum=fmaxf(x0,x1);
        for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,offset,32));
        maximum=__shfl(maximum,0,32);
        float e0=expf(x0-maximum),e1=expf(x1-maximum),sum=e0+e1;
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down(sum,offset,32);
        sum=__shfl(sum,0,32);
        scores[query_row*64+lane]=quant(rnd(e0/sum));
        scores[query_row*64+lane+32]=quant(rnd(e1/sum));
    }
    __syncthreads();
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int chunk=0;chunk<4;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int key=chunk*16+group*8+e,channel=m*16+query;
                a[e]=v[(window*64+key)*32+channel];
                b[e]=scores[row*64+key];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)value[(window*64+row)*32+m*16+group*8+e]=rnd(acc[e]);
    }
}

template<int C,int WAVES,bool FP8> __global__ void wide_qkv_wmma(const _Float16* input,const _Float16* weight,const int64_t* perm,_Float16* projection,int windows) {
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;bool active=window<windows;
    for(int m=0;m<3*C/16;++m){F8 acc={};for(int chunk=0;chunk<C/16;++chunk){H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e;a[e]=weight[kk*(3*C)+m*16+token];b[e]=active?quant(input[(window*64+row)*C+perm[kk]]):0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e)projection[(window*64+row)*(3*C)+m*16+group*8+e]=rnd(acc[e]);}
}

template<int C> __global__ void wide_qkv_norm(const _Float16* projection,const _Float16* qscale,_Float16* q,_Float16* k,_Float16* v,size_t vectors) {
    size_t warp=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32,index=warp/3;int part=warp%3,lane=threadIdx.x%32;if(index>=vectors)return;
    int heads=C/32,head=index/64%heads,token=index%64,window=index/(64*heads);_Float16 h=projection[(window*64+token)*(3*C)+part*C+head*32+lane];
    if(part==2){v[index*32+lane]=quant(h);return;}float t=float(h),sum=float(rnd(t*t));for(int offset=1;offset<32;offset*=2){float other=__shfl_down(sum,offset,32);if((lane%(2*offset))==0)sum=float(rnd(sum+other));}sum=__shfl(sum,0,32);float inv=float(rnd(rsqrtf(fmaxf(sum,6.198883056640625e-5f))));_Float16 n=rnd(t*inv);if(part==0)n=rnd(float(n)*float(qscale[head]));(part==0?q:k)[index*32+lane]=quant(n);
}

template<int C,int WAVES,bool FP8> __global__ void wide_qk_wmma(const _Float16* q,const _Float16* k,const _Float16* bias,_Float16* scores,int windows) {
    int heads=C/32,wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,group=lane/16;int task=blockIdx.x*WAVES+wave,head=task%heads,tile=task/heads,window=tile/4,row=tile%4*16+query;bool active=window<windows;size_t base=(size_t(window)*heads+head)*64;
    for(int kt=0;kt<4;++kt){F8 acc={};for(int chunk=0;chunk<2;++chunk){H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e,key=kt*16+query;a[e]=active?k[(base+key)*32+kk]:0;b[e]=active?q[(base+row)*32+kk]:0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e){int key=kt*16+group*8+e;scores[(base+row)*64+key]=rnd(float(rnd(acc[e]))+float(bias[(head*64+row)*64+key]));}}
}

template<int C,int WAVES,bool FP8> __global__ void wide_pv_wmma(const _Float16* probability,const _Float16* v,_Float16* value,int windows) {
    int heads=C/32,wave=threadIdx.x/32,lane=threadIdx.x%32,query=lane%16,group=lane/16;int task=blockIdx.x*WAVES+wave,head=task%heads,tile=task/heads,window=tile/4,row=tile%4*16+query;bool active=window<windows;size_t base=(size_t(window)*heads+head)*64;
    for(int m=0;m<2;++m){F8 acc={};for(int chunk=0;chunk<4;++chunk){H8 a,b;for(int e=0;e<8;++e){int key=chunk*16+group*8+e,channel=m*16+query;a[e]=active?v[(base+key)*32+channel]:0;b[e]=active?probability[(base+row)*64+key]:0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e)value[(base+row)*32+m*16+group*8+e]=rnd(acc[e]);}
}

template<int C,int WAVES,bool FP8> __global__ void wide_project_wmma(const _Float16* post,const _Float16* value,const _Float16* weight,const _Float16* scale,const int64_t* perm,_Float16* output,int windows) {
    int heads=C/32,wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16;int tile=blockIdx.x*WAVES+wave,window=tile/4,row=tile%4*16+token;bool active=window<windows;
    for(int m=0;m<C/16;++m){F8 acc={};for(int chunk=0;chunk<C/16;++chunk){H8 a,b;for(int e=0;e<8;++e){int kk=chunk*16+group*8+e,source=perm[kk],head=source/32,local=source%32;a[e]=weight[kk*C+m*16+token];b[e]=active?quant(value[((window*heads+head)*64+row)*32+local]):0;}acc=mma<FP8>(a,b,acc);}if(active)for(int e=0;e<8;++e){int c=m*16+group*8+e;_Float16 seed=rnd(float(post[(window*64+row)*C+c])*float(scale[c]));output[(window*64+row)*C+c]=rnd(float(rnd(acc[e]))+float(seed));}}
}

// Bounded wide-family attention. One workgroup owns one (window,head) pair;
// Q/K/V and scores stay in 20 KiB LDS. Only the final per-head attention value
// is written globally. A separate projection kernel performs the intentional
// cross-head/channel mix, keeping the fusion boundary bounded for C64/C128.
template<int C,bool FP8> __global__ void wide_attention_head_fused(
    const _Float16* post,const _Float16* qkv_weight,const _Float16* qscale,
    const _Float16* position_bias,const int64_t* perm,_Float16* value,int windows) {
    __shared__ _Float16 qkv[64*96];
    __shared__ _Float16 scores[64*64];
    constexpr int HEADS=C/32;
    int task=blockIdx.x,head=task%HEADS,window=task/HEADS;
    int wave=threadIdx.x/32,lane=threadIdx.x%32,token=lane%16,group=lane/16,row=wave*16+token;
    if(window>=windows)return;
    for(int m=0;m<6;++m) {
        int part=m/2,sub=m%2;F8 acc={};
        for(int chunk=0;chunk<C/16;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e;
                int out_col=part*C+head*32+sub*16+token;
                a[e]=qkv_weight[kk*(3*C)+out_col];
                b[e]=quant(post[(window*64+row)*C+perm[kk]]);
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)qkv[row*96+m*16+group*8+e]=rnd(acc[e]);
    }
    __syncthreads();
    for(int local=0;local<16;++local) {
        int norm_row=wave*16+local;
        for(int part=0;part<3;++part) {
            _Float16 h=qkv[norm_row*96+part*32+lane];
            if(part==2)qkv[norm_row*96+64+lane]=quant(h);
            else {
                float t=float(h),sum=float(rnd(t*t));
                for(int offset=1;offset<32;offset*=2) {
                    float other=__shfl_down(sum,offset,32);
                    if((lane%(2*offset))==0)sum=float(rnd(sum+other));
                }
                sum=__shfl(sum,0,32);
                float inv=float(rnd(rsqrtf(fmaxf(sum,6.198883056640625e-5f))));
                _Float16 normalized=rnd(t*inv);
                if(part==0)normalized=rnd(float(normalized)*float(qscale[head]));
                qkv[norm_row*96+part*32+lane]=quant(normalized);
            }
        }
    }
    __syncthreads();
    for(int kt=0;kt<4;++kt) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e,key=kt*16+token;
                a[e]=qkv[key*96+32+kk];b[e]=qkv[row*96+kk];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e) {
            int key=kt*16+group*8+e;
            scores[row*64+key]=rnd(float(rnd(acc[e]))+float(position_bias[(head*64+row)*64+key]));
        }
    }
    __syncthreads();
    for(int local=0;local<16;++local) {
        int query_row=wave*16+local;
        float x0=float(scores[query_row*64+lane]),x1=float(scores[query_row*64+lane+32]);
        float maximum=fmaxf(x0,x1);
        for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,offset,32));
        maximum=__shfl(maximum,0,32);
        float e0=expf(x0-maximum),e1=expf(x1-maximum),sum=e0+e1;
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down(sum,offset,32);
        sum=__shfl(sum,0,32);
        scores[query_row*64+lane]=quant(rnd(e0/sum));
        scores[query_row*64+lane+32]=quant(rnd(e1/sum));
    }
    __syncthreads();
    size_t base=(size_t(window)*HEADS+head)*64;
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int chunk=0;chunk<4;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int key=chunk*16+group*8+e,channel=m*16+token;
                a[e]=qkv[key*96+64+channel];b[e]=scores[row*64+key];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)value[(base+row)*32+m*16+group*8+e]=rnd(acc[e]);
    }
}

// Conservative wide-family attention core. QKV projection and normalization
// remain in their established parallel kernels. One single-wave workgroup owns
// (window, head, 16-query tile), keeping only 16x64 scores/probabilities in
// 2 KiB LDS while fusing QK, E4M3 softmax and PV.
template<int C,bool FP8> __global__ void wide_attention_query_fused(
    const _Float16* q,const _Float16* k,const _Float16* v,
    const _Float16* position_bias,_Float16* value,int windows) {
    __shared__ _Float16 scores[16*64];
    constexpr int HEADS=C/32;
    int lane=threadIdx.x,query=lane%16,group=lane/16;
    int task=blockIdx.x,head=task%HEADS,tile=task/HEADS;
    int window=tile/4,query_tile=tile%4,row=query_tile*16+query;
    if(window>=windows)return;
    size_t base=(size_t(window)*HEADS+head)*64;
    for(int kt=0;kt<4;++kt) {
        F8 acc={};
        for(int chunk=0;chunk<2;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int kk=chunk*16+group*8+e,key=kt*16+query;
                a[e]=k[(base+key)*32+kk];
                b[e]=q[(base+row)*32+kk];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e) {
            int key=kt*16+group*8+e;
            scores[query*64+key]=rnd(float(rnd(acc[e]))+
                float(position_bias[(head*64+row)*64+key]));
        }
    }
    __syncthreads();
    for(int local=0;local<16;++local) {
        float x0=float(scores[local*64+lane]);
        float x1=float(scores[local*64+lane+32]);
        float maximum=fmaxf(x0,x1);
        for(int offset=16;offset;offset>>=1)
            maximum=fmaxf(maximum,__shfl_down(maximum,offset,32));
        maximum=__shfl(maximum,0,32);
        float e0=expf(x0-maximum),e1=expf(x1-maximum),sum=e0+e1;
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down(sum,offset,32);
        sum=__shfl(sum,0,32);
        scores[local*64+lane]=quant(rnd(e0/sum));
        scores[local*64+lane+32]=quant(rnd(e1/sum));
    }
    __syncthreads();
    for(int m=0;m<2;++m) {
        F8 acc={};
        for(int chunk=0;chunk<4;++chunk) {
            H8 a,b;
            for(int e=0;e<8;++e) {
                int key=chunk*16+group*8+e,channel=m*16+query;
                a[e]=v[(base+key)*32+channel];
                b[e]=scores[query*64+key];
            }
            acc=mma<FP8>(a,b,acc);
        }
        for(int e=0;e<8;++e)
            value[(base+row)*32+m*16+group*8+e]=rnd(acc[e]);
    }
}

__global__ void pre_project_exact(const _Float16* input,const _Float16* weight,
    _Float16* output,size_t rows) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*32)return;
    size_t row=i/32;int col=i%32;float sum=0.f;
    for(int k=0;k<16;++k)sum+=float(input[row*16+k])*float(weight[k*32+col]);
    output[i]=rnd(sum);
}

extern "C" __declspec(dllexport) int nr_head_ffn_wmma(
    const void* x,const void* we,const void* wc,const void* scale,const void* ai,
    const void* ri,const void* perm,void* out,int windows,int waves,int fp8,void* stream) {
    if(!x||!we||!wc||!scale||!ai||!ri||!perm||!out||windows<1||windows>262144||
       (waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define RUN(W,F) hipLaunchKernelGGL((head_ffn_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(x),static_cast<const _Float16*>(we),static_cast<const _Float16*>(wc),static_cast<const _Float16*>(scale),static_cast<const int64_t*>(ai),static_cast<const int64_t*>(ri),static_cast<const int64_t*>(perm),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){RUN(1,true);}else if(waves==2){RUN(2,true);}else{RUN(4,true);}}
    else {if(waves==1){RUN(1,false);}else if(waves==2){RUN(2,false);}else{RUN(4,false);}}
#undef RUN
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_project_wmma(
    const void* first,const void* attention,const void* weight,const void* scale,
    const void* perm,void* out,int windows,int waves,int fp8,void* stream) {
    if(!first||!attention||!weight||!scale||!perm||!out||windows<1||windows>262144||
       (waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define PROJECT(W,F) hipLaunchKernelGGL((head_project_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(first),static_cast<const _Float16*>(attention),static_cast<const _Float16*>(weight),static_cast<const _Float16*>(scale),static_cast<const int64_t*>(perm),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){PROJECT(1,true);}else if(waves==2){PROJECT(2,true);}else{PROJECT(4,true);}}
    else {if(waves==1){PROJECT(1,false);}else if(waves==2){PROJECT(2,false);}else{PROJECT(4,false);}}
#undef PROJECT
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_tail_exact(
    const void* input,const void* weight,void* out,uint64_t rows,void* stream) {
    if(!input||!weight||!out||!rows||rows>(uint64_t(1)<<32))return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(head_tail_exact,dim3(unsigned((rows*4+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(input),
        static_cast<const _Float16*>(weight),static_cast<_Float16*>(out),size_t(rows));
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_qk_wmma(
    const void* q,const void* k,const void* bias,void* scores,int windows,int waves,int fp8,void* stream) {
    if(!q||!k||!bias||!scores||windows<1||windows>262144||(waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define QK(W,F) hipLaunchKernelGGL((head_qk_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(q),static_cast<const _Float16*>(k),static_cast<const _Float16*>(bias),static_cast<_Float16*>(scores),windows)
    if(fp8){if(waves==1){QK(1,true);}else if(waves==2){QK(2,true);}else{QK(4,true);}}
    else {if(waves==1){QK(1,false);}else if(waves==2){QK(2,false);}else{QK(4,false);}}
#undef QK
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_pv_wmma(
    const void* probability,const void* v,void* out,int windows,int waves,int fp8,void* stream) {
    if(!probability||!v||!out||windows<1||windows>262144||(waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define PV(W,F) hipLaunchKernelGGL((head_pv_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(probability),static_cast<const _Float16*>(v),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){PV(1,true);}else if(waves==2){PV(2,true);}else{PV(4,true);}}
    else {if(waves==1){PV(1,false);}else if(waves==2){PV(2,false);}else{PV(4,false);}}
#undef PV
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_qkv_project_wmma(
    const void* input,const void* weight,const void* perm,void* out,int windows,int waves,int fp8,void* stream) {
    if(!input||!weight||!perm||!out||windows<1||windows>262144||(waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define QKV(W,F) hipLaunchKernelGGL((head_qkv_project_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(input),static_cast<const _Float16*>(weight),static_cast<const int64_t*>(perm),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){QKV(1,true);}else if(waves==2){QKV(2,true);}else{QKV(4,true);}}
    else {if(waves==1){QKV(1,false);}else if(waves==2){QKV(2,false);}else{QKV(4,false);}}
#undef QKV
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_softmax_e4(
    const void* scores,void* probability,uint64_t rows,void* stream) {
    if(!scores||!probability||!rows||rows>(uint64_t(1)<<36))return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(head_softmax_e4,dim3(unsigned((rows*32+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(scores),
        static_cast<_Float16*>(probability),size_t(rows));
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_head_attention_window_fused(
    const void* first,const void* qkv_weight,const void* qscale,const void* bias,
    const void* project_weight,const void* attention_scale,const void* perm,void* output,
    int windows,int fp8,void* stream) {
    if(!first||!qkv_weight||!qscale||!bias||!project_weight||!attention_scale||!perm||!output||
       windows<1||windows>262144||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define ATTN(F) hipLaunchKernelGGL((head_attention_window_fused<F,false>),dim3(windows),dim3(128),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(first),static_cast<const _Float16*>(qkv_weight),static_cast<const _Float16*>(qscale),static_cast<const _Float16*>(bias),static_cast<const _Float16*>(project_weight),static_cast<const _Float16*>(attention_scale),static_cast<const int64_t*>(perm),static_cast<_Float16*>(output),windows)
    if(fp8){ATTN(true);}else{ATTN(false);}
#undef ATTN
    return int(hipGetLastError());
}

// The recovered C32 window-attention contract has the same logical tensor,
// weight, scale, position-bias and half-rounding boundaries as Head attention.
// Keep a distinct ABI symbol so policies and evidence cannot accidentally
// promote one family based on the other's gate.
extern "C" __declspec(dllexport) int nr_c32_attention_window_fused(
    const void* first,const void* qkv_weight,const void* qscale,const void* bias,
    const void* project_weight,const void* attention_scale,const void* perm,
    void* output,int windows,int fp8,void* stream) {
    if(!first||!qkv_weight||!qscale||!bias||!project_weight||!attention_scale||!perm||!output||
       windows<1||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define C32_ATTN(F) hipLaunchKernelGGL((head_attention_window_fused<F,true>),dim3(windows),dim3(128),0,\
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(first),\
        static_cast<const _Float16*>(qkv_weight),static_cast<const _Float16*>(qscale),\
        static_cast<const _Float16*>(bias),static_cast<const _Float16*>(project_weight),\
        static_cast<const _Float16*>(attention_scale),static_cast<const int64_t*>(perm),\
        static_cast<_Float16*>(output),windows)
    if(fp8){C32_ATTN(true);}else{C32_ATTN(false);}
#undef C32_ATTN
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_pre_project_wmma(
    const void* input,const void* weight,void* output,uint64_t rows,int waves,void* stream) {
    if(!input||!weight||!output||!rows||rows>(uint64_t(1)<<32)||rows%16||(waves!=1&&waves!=2&&waves!=4))return int(hipErrorInvalidValue);
#define PRE(W) hipLaunchKernelGGL((pre_project_wmma<W>),dim3(unsigned((rows+16*W-1)/(16*W))),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(input),static_cast<const _Float16*>(weight),static_cast<_Float16*>(output),size_t(rows))
    if(waves==1){PRE(1);}else if(waves==2){PRE(2);}else{PRE(4);}
#undef PRE
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_pre_project_exact(
    const void* input,const void* weight,void* output,uint64_t rows,void* stream) {
    if(!input||!weight||!output||!rows||rows>(uint64_t(1)<<32))return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(pre_project_exact,dim3(unsigned((rows*32+255)/256)),dim3(256),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(input),
        static_cast<const _Float16*>(weight),static_cast<_Float16*>(output),size_t(rows));
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_c32_ffn_wmma(
    const void* raw,const void* expand,const void* contract,const void* scale,const void* ai,const void* ri,const void* perm,void* out,int windows,int waves,int fp8,void* stream) {
    if(!raw||!expand||!contract||!scale||!ai||!ri||!perm||!out||windows<1||windows>262144||(waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define C32(W,F) hipLaunchKernelGGL((c32_ffn_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(raw),static_cast<const _Float16*>(expand),static_cast<const _Float16*>(contract),static_cast<const _Float16*>(scale),static_cast<const int64_t*>(ai),static_cast<const int64_t*>(ri),static_cast<const int64_t*>(perm),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){C32(1,true);}else if(waves==2){C32(2,true);}else{C32(4,true);}}
    else {if(waves==1){C32(1,false);}else if(waves==2){C32(2,false);}else{C32(4,false);}}
#undef C32
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_c32_qkv_norm(const void* projection,const void* scale,void* q,void* k,void* v,uint64_t rows,void* stream) {
    if(!projection||!scale||!q||!k||!v||!rows)return int(hipErrorInvalidValue);hipLaunchKernelGGL(c32_qkv_norm,dim3(unsigned((rows*96+255)/256)),dim3(256),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(projection),static_cast<const _Float16*>(scale),static_cast<_Float16*>(q),static_cast<_Float16*>(k),static_cast<_Float16*>(v),size_t(rows));return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_c32_qk_wmma(const void* q,const void* k,const void* bias,void* out,int windows,int waves,int fp8,void* stream) {
    if(!q||!k||!bias||!out||windows<1||(waves!=1&&waves!=2&&waves!=4))return int(hipErrorInvalidValue);
#define C32QK(W,F) hipLaunchKernelGGL((c32_qk_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(q),static_cast<const _Float16*>(k),static_cast<const _Float16*>(bias),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){C32QK(1,true);}else if(waves==2){C32QK(2,true);}else{C32QK(4,true);}}else{if(waves==1){C32QK(1,false);}else if(waves==2){C32QK(2,false);}else{C32QK(4,false);}}
#undef C32QK
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_c32_pv_wmma(const void* p,const void* v,void* out,int windows,int waves,int fp8,void* stream) {
    if(!p||!v||!out||windows<1||(waves!=1&&waves!=2&&waves!=4))return int(hipErrorInvalidValue);
#define C32PV(W,F) hipLaunchKernelGGL((c32_pv_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(p),static_cast<const _Float16*>(v),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){C32PV(1,true);}else if(waves==2){C32PV(2,true);}else{C32PV(4,true);}}else{if(waves==1){C32PV(1,false);}else if(waves==2){C32PV(2,false);}else{C32PV(4,false);}}
#undef C32PV
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_c32_project_wmma(const void* post,const void* value,const void* weight,const void* scale,const void* perm,void* out,int windows,int waves,int fp8,void* stream) {
    if(!post||!value||!weight||!scale||!perm||!out||windows<1||(waves!=1&&waves!=2&&waves!=4))return int(hipErrorInvalidValue);
#define C32PROJ(W,F) hipLaunchKernelGGL((c32_project_wmma<W,F>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(post),static_cast<const _Float16*>(value),static_cast<const _Float16*>(weight),static_cast<const _Float16*>(scale),static_cast<const int64_t*>(perm),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){C32PROJ(1,true);}else if(waves==2){C32PROJ(2,true);}else{C32PROJ(4,true);}}else{if(waves==1){C32PROJ(1,false);}else if(waves==2){C32PROJ(2,false);}else{C32PROJ(4,false);}}
#undef C32PROJ
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_c32_qkv_norm_staged(
    const void* post,const void* qkv_weight,const void* qscale,const void* perm,
    void* q,void* k,void* v,int windows,int fp8,void* stream) {
    if(!post||!qkv_weight||!qscale||!perm||!q||!k||!v||windows<1||windows>262144||
       (fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define C32_STAGED_QKV(F) hipLaunchKernelGGL((c32_qkv_norm_staged<F>),dim3(windows),dim3(128),0,\
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(post),\
        static_cast<const _Float16*>(qkv_weight),static_cast<const _Float16*>(qscale),\
        static_cast<const int64_t*>(perm),static_cast<_Float16*>(q),static_cast<_Float16*>(k),\
        static_cast<_Float16*>(v),windows)
    if(fp8){C32_STAGED_QKV(true);}else{C32_STAGED_QKV(false);}
#undef C32_STAGED_QKV
    return int(hipGetLastError());
}

extern "C" __declspec(dllexport) int nr_c32_attention_core_staged(
    const void* q,const void* k,const void* v,const void* bias,void* value,
    int windows,int fp8,void* stream) {
    if(!q||!k||!v||!bias||!value||windows<1||windows>262144||
       (fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define C32_STAGED_CORE(F) hipLaunchKernelGGL((c32_attention_core_staged<F>),dim3(windows),dim3(128),0,\
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(q),\
        static_cast<const _Float16*>(k),static_cast<const _Float16*>(v),\
        static_cast<const _Float16*>(bias),static_cast<_Float16*>(value),windows)
    if(fp8){C32_STAGED_CORE(true);}else{C32_STAGED_CORE(false);}
#undef C32_STAGED_CORE
    return int(hipGetLastError());
}

template<int C,int W,bool FP8> static void launch_wide_qkv(const void* input,const void* weight,const void* perm,void* output,int windows,void* stream){hipLaunchKernelGGL((wide_qkv_wmma<C,W,FP8>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(input),static_cast<const _Float16*>(weight),static_cast<const int64_t*>(perm),static_cast<_Float16*>(output),windows);}
template<int C,int W,bool FP8> static void launch_wide_qk(const void* q,const void* k,const void* bias,void* output,int windows,void* stream){hipLaunchKernelGGL((wide_qk_wmma<C,W,FP8>),dim3((windows*4*(C/32)+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(q),static_cast<const _Float16*>(k),static_cast<const _Float16*>(bias),static_cast<_Float16*>(output),windows);}
template<int C,int W,bool FP8> static void launch_wide_pv(const void* p,const void* v,void* output,int windows,void* stream){hipLaunchKernelGGL((wide_pv_wmma<C,W,FP8>),dim3((windows*4*(C/32)+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(p),static_cast<const _Float16*>(v),static_cast<_Float16*>(output),windows);}
template<int C,int W,bool FP8> static void launch_wide_project(const void* post,const void* value,const void* weight,const void* scale,const void* perm,void* output,int windows,void* stream){hipLaunchKernelGGL((wide_project_wmma<C,W,FP8>),dim3((windows*4+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(post),static_cast<const _Float16*>(value),static_cast<const _Float16*>(weight),static_cast<const _Float16*>(scale),static_cast<const int64_t*>(perm),static_cast<_Float16*>(output),windows);}
template<int C,int KIND> static int wide_attention_entry(const void* a,const void* b,const void* c,const void* d,const void* e,void* out,int windows,int waves,int fp8,void* stream){
    if(!a||!b||!out||windows<1||(waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define RUN_ATTN(W,F) if constexpr(KIND==0)launch_wide_qkv<C,W,F>(a,b,c,out,windows,stream);else if constexpr(KIND==1)launch_wide_qk<C,W,F>(a,b,c,out,windows,stream);else if constexpr(KIND==2)launch_wide_pv<C,W,F>(a,b,out,windows,stream);else launch_wide_project<C,W,F>(a,b,c,d,e,out,windows,stream)
    if(fp8){if(waves==1){RUN_ATTN(1,true);}else if(waves==2){RUN_ATTN(2,true);}else{RUN_ATTN(4,true);}}
    else {if(waves==1){RUN_ATTN(1,false);}else if(waves==2){RUN_ATTN(2,false);}else{RUN_ATTN(4,false);}}
#undef RUN_ATTN
    return int(hipGetLastError());
}
#define WIDE_ATTN_EXPORT(C) \
extern "C" __declspec(dllexport) int nr_c##C##_qkv_wmma(const void* input,const void* weight,const void* perm,void* out,int windows,int waves,int fp8,void* stream){return wide_attention_entry<C,0>(input,weight,perm,nullptr,nullptr,out,windows,waves,fp8,stream);} \
extern "C" __declspec(dllexport) int nr_c##C##_qkv_norm(const void* projection,const void* scale,void* q,void* k,void* v,uint64_t vectors,void* stream){if(!projection||!scale||!q||!k||!v||!vectors)return int(hipErrorInvalidValue);hipLaunchKernelGGL((wide_qkv_norm<C>),dim3((vectors*3*32+255)/256),dim3(256),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(projection),static_cast<const _Float16*>(scale),static_cast<_Float16*>(q),static_cast<_Float16*>(k),static_cast<_Float16*>(v),vectors);return int(hipGetLastError());} \
extern "C" __declspec(dllexport) int nr_c##C##_qk_wmma(const void* q,const void* k,const void* bias,void* out,int windows,int waves,int fp8,void* stream){return wide_attention_entry<C,1>(q,k,bias,nullptr,nullptr,out,windows,waves,fp8,stream);} \
extern "C" __declspec(dllexport) int nr_c##C##_pv_wmma(const void* p,const void* v,void* out,int windows,int waves,int fp8,void* stream){return wide_attention_entry<C,2>(p,v,nullptr,nullptr,nullptr,out,windows,waves,fp8,stream);} \
extern "C" __declspec(dllexport) int nr_c##C##_project_wmma(const void* post,const void* value,const void* weight,const void* scale,const void* perm,void* out,int windows,int waves,int fp8,void* stream){return wide_attention_entry<C,3>(post,value,weight,scale,perm,out,windows,waves,fp8,stream);}
WIDE_ATTN_EXPORT(64)
WIDE_ATTN_EXPORT(128)
#undef WIDE_ATTN_EXPORT

template<int C,bool FP8> static void launch_wide_attention_head_fused(
    const void* post,const void* qkv,const void* qscale,const void* bias,const void* perm,
    void* value,int windows,void* stream) {
    hipLaunchKernelGGL((wide_attention_head_fused<C,FP8>),dim3(windows*(C/32)),dim3(128),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(post),
        static_cast<const _Float16*>(qkv),static_cast<const _Float16*>(qscale),
        static_cast<const _Float16*>(bias),static_cast<const int64_t*>(perm),
        static_cast<_Float16*>(value),windows);
}
template<int C> static int wide_attention_head_entry(const void* post,const void* qkv,const void* qscale,
    const void* bias,const void* perm,void* value,int windows,int fp8,void* stream) {
    if(!post||!qkv||!qscale||!bias||!perm||!value||windows<1||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
    if(fp8)launch_wide_attention_head_fused<C,true>(post,qkv,qscale,bias,perm,value,windows,stream);
    else launch_wide_attention_head_fused<C,false>(post,qkv,qscale,bias,perm,value,windows,stream);
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_c64_attention_head_fused(const void* post,const void* qkv,
    const void* qscale,const void* bias,const void* perm,void* value,int windows,int fp8,void* stream) {
    return wide_attention_head_entry<64>(post,qkv,qscale,bias,perm,value,windows,fp8,stream);
}
extern "C" __declspec(dllexport) int nr_c128_attention_head_fused(const void* post,const void* qkv,
    const void* qscale,const void* bias,const void* perm,void* value,int windows,int fp8,void* stream) {
    return wide_attention_head_entry<128>(post,qkv,qscale,bias,perm,value,windows,fp8,stream);
}

template<int C,bool FP8> static void launch_wide_attention_query(
    const void* q,const void* k,const void* v,const void* bias,void* value,
    int windows,void* stream) {
    hipLaunchKernelGGL((wide_attention_query_fused<C,FP8>),dim3(windows*(C/32)*4),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(q),
        static_cast<const _Float16*>(k),static_cast<const _Float16*>(v),
        static_cast<const _Float16*>(bias),static_cast<_Float16*>(value),windows);
}
template<int C> static int wide_attention_query_entry(
    const void* q,const void* k,const void* v,const void* bias,void* value,
    int windows,int fp8,void* stream) {
    if(!q||!k||!v||!bias||!value||windows<1||windows>262144||(fp8!=0&&fp8!=1))
        return int(hipErrorInvalidValue);
    if(fp8)launch_wide_attention_query<C,true>(q,k,v,bias,value,windows,stream);
    else launch_wide_attention_query<C,false>(q,k,v,bias,value,windows,stream);
    return int(hipGetLastError());
}
#define QUERY_ATTN_EXPORT(C) \
extern "C" __declspec(dllexport) int nr_c##C##_attention_query_fused( \
    const void* q,const void* k,const void* v,const void* bias,void* value, \
    int windows,int fp8,void* stream) { \
    return wide_attention_query_entry<C>(q,k,v,bias,value,windows,fp8,stream); \
}
QUERY_ATTN_EXPORT(64)
QUERY_ATTN_EXPORT(128)
#undef QUERY_ATTN_EXPORT

template<int C,bool FP8> static void launch_wide_group_ffn(
    const void* raw,const void* expand,const void* contract,const void* scale,
    const void* ai,const void* ri,const void* perm,const void* inverse_perm,
    void* grouped,void* seed,int windows,void* stream) {
    hipLaunchKernelGGL((wide_group_ffn_wmma<C,FP8>),dim3(windows*(C/32)*4),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(raw),
        static_cast<const _Float16*>(expand),static_cast<const _Float16*>(contract),
        static_cast<const _Float16*>(scale),static_cast<const int64_t*>(ai),
        static_cast<const int64_t*>(ri),static_cast<const int64_t*>(perm),
        static_cast<const int64_t*>(inverse_perm),static_cast<_Float16*>(grouped),
        static_cast<_Float16*>(seed),windows);
}
template<int C> static int wide_group_ffn_entry(
    const void* raw,const void* expand,const void* contract,const void* scale,
    const void* ai,const void* ri,const void* perm,const void* inverse_perm,
    void* grouped,void* seed,int windows,int fp8,void* stream) {
    if(!raw||!expand||!contract||!scale||!ai||!ri||!perm||!inverse_perm||
       !grouped||!seed||windows<1||windows>262144||(fp8!=0&&fp8!=1))
        return int(hipErrorInvalidValue);
    if(fp8)launch_wide_group_ffn<C,true>(raw,expand,contract,scale,ai,ri,perm,
        inverse_perm,grouped,seed,windows,stream);
    else launch_wide_group_ffn<C,false>(raw,expand,contract,scale,ai,ri,perm,
        inverse_perm,grouped,seed,windows,stream);
    return int(hipGetLastError());
}
#define GROUP_FFN_EXPORT(C) \
extern "C" __declspec(dllexport) int nr_c##C##_group_ffn_wmma( \
    const void* raw,const void* expand,const void* contract,const void* scale, \
    const void* ai,const void* ri,const void* perm,const void* inverse_perm, \
    void* grouped,void* seed,int windows,int fp8,void* stream) { \
    return wide_group_ffn_entry<C>(raw,expand,contract,scale,ai,ri,perm,inverse_perm, \
        grouped,seed,windows,fp8,stream); \
}
GROUP_FFN_EXPORT(64)
GROUP_FFN_EXPORT(128)
GROUP_FFN_EXPORT(256)
#undef GROUP_FFN_EXPORT

template<int C> static int wide_group_ffn_fp8w_entry(
    const void* raw,const void* expand,const void* contract,const void* scale,
    const void* ai,const void* ri,const void* perm,const void* inverse_perm,
    void* grouped,void* seed,int windows,void* stream) {
    if(!raw||!expand||!contract||!scale||!ai||!ri||!perm||!inverse_perm||
       !grouped||!seed||windows<1||windows>262144)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((wide_group_ffn_fp8w<C>),dim3(windows*(C/32)*4),dim3(32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(raw),
        static_cast<const uint8_t*>(expand),static_cast<const uint8_t*>(contract),
        static_cast<const _Float16*>(scale),static_cast<const int64_t*>(ai),
        static_cast<const int64_t*>(ri),static_cast<const int64_t*>(perm),
        static_cast<const int64_t*>(inverse_perm),static_cast<_Float16*>(grouped),
        static_cast<_Float16*>(seed),windows);
    return int(hipGetLastError());
}
#define GROUP_FFN_FP8W_EXPORT(C) \
extern "C" __declspec(dllexport) int nr_c##C##_group_ffn_fp8w( \
    const void* raw,const void* expand,const void* contract,const void* scale, \
    const void* ai,const void* ri,const void* perm,const void* inverse_perm, \
    void* grouped,void* seed,int windows,void* stream) { \
    return wide_group_ffn_fp8w_entry<C>(raw,expand,contract,scale,ai,ri,perm,inverse_perm, \
        grouped,seed,windows,stream); \
}
GROUP_FFN_FP8W_EXPORT(64)
GROUP_FFN_FP8W_EXPORT(128)
GROUP_FFN_FP8W_EXPORT(256)
#undef GROUP_FFN_FP8W_EXPORT

template<int C,int W,bool FP8> static void launch_wide(const void* raw,const void* expand,
    const void* contract,const void* mix,const void* scale,const void* ai,const void* ri,
    const void* perm,void* out,int windows,void* stream) {
    hipLaunchKernelGGL((wide_ffn_wmma<C,W,FP8>),dim3((windows*4+W-1)/W),dim3(W*32),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(raw),
        static_cast<const _Float16*>(expand),static_cast<const _Float16*>(contract),
        static_cast<const _Float16*>(mix),static_cast<const _Float16*>(scale),
        static_cast<const int64_t*>(ai),static_cast<const int64_t*>(ri),
        static_cast<const int64_t*>(perm),static_cast<_Float16*>(out),windows);
}
template<int C> static int wide_entry(const void* raw,const void* expand,const void* contract,
    const void* mix,const void* scale,const void* ai,const void* ri,const void* perm,void* out,
    int windows,int waves,int fp8,void* stream) {
    if(!raw||!expand||!contract||!mix||!scale||!ai||!ri||!perm||!out||windows<1||
       (waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
    if(fp8){if(waves==1)launch_wide<C,1,true>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);
        else if(waves==2)launch_wide<C,2,true>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);
        else launch_wide<C,4,true>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);}
    else {if(waves==1)launch_wide<C,1,false>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);
        else if(waves==2)launch_wide<C,2,false>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);
        else launch_wide<C,4,false>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);}
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_c64_ffn_wmma(const void* raw,const void* expand,const void* contract,const void* mix,const void* scale,const void* ai,const void* ri,const void* perm,void* out,int windows,int waves,int fp8,void* stream){return wide_entry<64>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,waves,fp8,stream);}
extern "C" __declspec(dllexport) int nr_c128_ffn_wmma(const void* raw,const void* expand,const void* contract,const void* mix,const void* scale,const void* ai,const void* ri,const void* perm,void* out,int windows,int waves,int fp8,void* stream){return wide_entry<128>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,waves,fp8,stream);}
// C256 consumes 32 KiB LDS per wave.  One and two waves are legal on this
// gfx1201 target; four waves would require 128 KiB and is rejected before any
// launch rather than relying on a device-side resource failure.
extern "C" __declspec(dllexport) int nr_c256_ffn_wmma(const void* raw,const void* expand,const void* contract,const void* mix,const void* scale,const void* ai,const void* ri,const void* perm,void* out,int windows,int waves,int fp8,void* stream){
    if(!raw||!expand||!contract||!mix||!scale||!ai||!ri||!perm||!out||windows<1||
       (waves!=1&&waves!=2)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
    if(fp8){if(waves==1)launch_wide<256,1,true>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);
        else launch_wide<256,2,true>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);}
    else {if(waves==1)launch_wide<256,1,false>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);
        else launch_wide<256,2,false>(raw,expand,contract,mix,scale,ai,ri,perm,out,windows,stream);}
    return int(hipGetLastError());
}
extern "C" __declspec(dllexport) int nr_c512_group_ffn_wmma(const void* projected,const void* expand,const void* contract,const void* perm64,const void* perm256,void* out,int windows,int waves,int fp8,void* stream){
    if(!projected||!expand||!contract||!perm64||!perm256||!out||windows<1||
       (waves!=1&&waves!=2&&waves!=4)||(fp8!=0&&fp8!=1))return int(hipErrorInvalidValue);
#define C512GROUP(W,F) hipLaunchKernelGGL((c512_group_ffn_wmma<W,F>),dim3((windows*32+W-1)/W),dim3(W*32),0,reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(projected),static_cast<const _Float16*>(expand),static_cast<const _Float16*>(contract),static_cast<const int64_t*>(perm64),static_cast<const int64_t*>(perm256),static_cast<_Float16*>(out),windows)
    if(fp8){if(waves==1){C512GROUP(1,true);}else if(waves==2){C512GROUP(2,true);}else{C512GROUP(4,true);}}
    else {if(waves==1){C512GROUP(1,false);}else if(waves==2){C512GROUP(2,false);}else{C512GROUP(4,false);}}
#undef C512GROUP
    return int(hipGetLastError());
}
