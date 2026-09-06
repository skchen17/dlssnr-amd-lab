#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_=(x); if(e_!=hipSuccess) \
  throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e_)); } while(0)

namespace {
constexpr uint32_t GW=81, GH=49, TOKENS=64, CHANNELS=32;
constexpr size_t CTAS=size_t(GW)*GH, VALUES=CTAS*TOKENS*CHANNELS;
constexpr size_t MAIN_OFF=13873152, MAIN_BYTES=2*192*320*16;
constexpr size_t SKIP_OFF=110592, SKIP_BYTES=96*160*512;
constexpr size_t HEAD_OFF=147429888, HEAD_BYTES=21808;
constexpr size_t A_BYTES=VALUES, H_BYTES=VALUES*2, HIDDEN_BYTES=VALUES*4;
constexpr size_t PROJ176_VALUES=CTAS*48*16*8;
constexpr size_t QK_VALUES=CTAS*32*16*8, SOFT_BYTES=CTAS*64*64;
constexpr size_t RESIDUAL_VALUES=CTAS*TOKENS*4;
constexpr uint32_t WIDTH=640, HEIGHT=360;
constexpr size_t SURFACE_VALUES=size_t(WIDTH)*HEIGHT*4;

std::vector<uint8_t> Read(const std::filesystem::path& p) {
  std::ifstream s(p,std::ios::binary|std::ios::ate);
  if(!s) throw std::runtime_error("cannot open "+p.string());
  size_t n=size_t(s.tellg()); s.seekg(0); std::vector<uint8_t>b(n);
  if(!s.read(reinterpret_cast<char*>(b.data()),std::streamsize(n))) throw std::runtime_error("read failed");
  return b;
}
std::vector<uint8_t> Slice(const std::filesystem::path&p,size_t o,size_t n) {
  std::ifstream s(p,std::ios::binary|std::ios::ate);
  if(!s) throw std::runtime_error("cannot open "+p.string());
  size_t z=size_t(s.tellg()); if(o>z||n>z-o) throw std::runtime_error("slice exceeds file");
  s.seekg(std::streamoff(o)); std::vector<uint8_t>b(n);
  if(!s.read(reinterpret_cast<char*>(b.data()),std::streamsize(n))) throw std::runtime_error("read failed");
  return b;
}
void Write(const std::filesystem::path&p,const void*d,size_t n) {
  if(p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  std::ofstream s(p,std::ios::binary); if(!s||!s.write(reinterpret_cast<const char*>(d),std::streamsize(n))) throw std::runtime_error("write failed");
}
uint64_t Hash(const void*d,size_t n) { const auto*p=static_cast<const uint8_t*>(d); uint64_t h=1469598103934665603ull; for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;} return h; }

__device__ uint32_t RShift(uint32_t v,uint32_t s){uint32_t b=v>>s,r=v&((1u<<s)-1),h=1u<<(s-1);return b+uint32_t(r>h||(r==h&&(b&1)));}
__device__ uint8_t Enc(uint16_t x){uint8_t sign=uint8_t((x>>8)&128);uint32_t e=(x>>10)&31,m=x&1023,c;if(e==31&&m)return 127;if(e<9)c=min(RShift(1024+m,16-e),8u);else{uint32_t r=RShift(m,7);bool q=r>=8;c=((e-8+uint32_t(q))<<3)|(q?0:r);c=min(c,126u);}return sign|uint8_t(c);}
__device__ float Dec(uint8_t x){uint8_t m=x&127;int e=m>>3,n=m&7;float v=e==0?ldexpf(float(n),-9):(m==127?nanf(""):ldexpf(1.0f+float(n)/8,e-7));return x&128?-v:v;}
__device__ uint16_t HRound(float x){return __half_as_ushort(__float2half_rn(x));}
__device__ float HVal(uint16_t x){return __half2float(__ushort_as_half(x));}
__device__ uint16_t HMul(uint16_t a,uint16_t b){return HRound(HVal(a)*HVal(b));}
__device__ uint16_t HAdd(uint16_t a,uint16_t b){return HRound(HVal(a)+HVal(b));}
__host__ __device__ size_t PA(uint32_t g,uint32_t r,uint32_t k){uint32_t l=(r%8)*4+(k%16)/4,e=(k%4)+(r>=8?4:0)+(k>=16?8:0);return size_t(g)*512+l*16+e;}
__device__ size_t PB(size_t tile,uint32_t k,uint32_t c){uint32_t l=(c%8)*4+(k%16)/4,e=(k%4)+(k>=16?4:0);return tile+l*16+(c/8)*8+e;}
__device__ uint32_t Perm32(uint32_t c){uint32_t b=c/16,w=c%16;return b*16+2*(w/4)+(w&1)+((w&2)?8:0);}

__device__ uint32_t ShuffleLane(uint32_t lane,uint32_t output){uint32_t q=(lane&19u)|((lane<<1)&8u)|((lane>>1)&4u),s[4]={q,q^8u,q^16u,q^24u},m[4];if(lane&16u){m[0]=s[2];m[1]=s[3];m[2]=s[0];m[3]=s[1];}else for(uint32_t i=0;i<4;++i)m[i]=s[i];if(!(lane&4u))return m[output];uint32_t w[4]={m[1],m[0],m[3],m[2]};return w[output];}
__device__ uint16_t FuseValue(const uint8_t*main,const uint8_t*skip,const uint8_t*head,uint32_t cx,uint32_t cy,uint32_t op,uint32_t lane,uint32_t ph){uint32_t cell=op/8,within=op%8,set=op<16?0:1,bases[2][4]={{0,1,4,5},{2,3,6,7}},src=bases[set][within/2],par=(op%16)/8,so=par+(within&1?2:0),sl=ShuffleLane(lane,so),load=src/2,plane=load/2;int32_t my=(-4+int32_t(cy)*8)/2+int32_t(sl/16)+(load&1?2:0),mx=(-4+int32_t(cx)*8)/2+int32_t((sl/4)%4);uint32_t cg=sl%4;uint8_t mc=0;if(my>=0&&my<192&&mx>=0&&mx<320)mc=main[((size_t(plane)*192+uint32_t(my))*320+uint32_t(mx))*16+cg*4+(src&1)*2+ph];uint32_t bo[8]={0,4,2,6,8,12,10,14};int32_t sy=(-4+int32_t(cy)*8)/4+int32_t(cell/2),sx=(-4+int32_t(cx)*8)/4+int32_t(cell%2);uint8_t sc=0;if(sy>=0&&sy<96&&sx>=0&&sx<160)sc=skip[(size_t(sy)*160+uint32_t(sx))*512+lane*16+bo[within]+ph];size_t o=size_t((op/2)%4)*16+(lane%4)*4+ph*2;uint16_t ms=uint16_t(head[8272+o])|uint16_t(head[8273+o])<<8,ss=uint16_t(head[8336+o])|uint16_t(head[8337+o])<<8;float rm=HVal(HRound(Dec(mc)*HVal(ms)));return HRound(fmaf(Dec(sc),HVal(ss),rm));}
__global__ void Activation(const uint8_t*main,const uint8_t*skip,const uint8_t*head,uint8_t*a,uint16_t*ah){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t cta=i/2048;uint32_t local=uint32_t(i%2048),ph=local&1,pw=(local>>1)%8,lane=(local>>4)%32,cell=local/(32*16),order[8]={0,2,1,3,4,6,5,7},op=cell*8+order[pw],cx=uint32_t(cta%GW),cy=uint32_t(cta/GW);uint16_t v=FuseValue(main,skip,head,cx,cy,op,lane,ph);a[i]=Enc(v);ah[i]=v;}

__device__ uint16_t Activate(uint16_t xb){float x=HVal(xb),c=fmaxf(-4,fminf(x,4)),a=fabsf(c),f=HVal(HRound(fmaf(-0.055908203125f,a,0.447265625f))),s=HVal(HRound(fmaf(c,f,0.89453125f)));return HRound(x*s);}
__global__ void Hidden(const uint8_t*a,const uint8_t*head,uint8_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES*4)return;size_t z=i;uint32_t col=uint32_t(z%32);z/=32;uint32_t row=uint32_t(z%16);z/=16;uint32_t g=uint32_t(z%4);z/=4;uint32_t pass=uint32_t(z%4);size_t cta=z/4,tile=size_t(pass)*1024+(col/16)*512;float acc=0;for(uint32_t k=0;k<32;++k)acc+=Dec(a[cta*2048+PA(g,row,k)])*Dec(head[PB(tile,k,col%16)]);uint8_t v=Enc(Activate(HRound(acc)));uint32_t mig=(col/16)*2+(col%16)/8,c8=col%8,l=(row%8)*4+c8/2,de=(row>=8?2:0)+(c8&1),conv=(mig/2)*4+(mig%2)+(de/2)*2,pe=conv*2+(de&1);out[(cta*4+pass)*2048+size_t(g)*512+l*16+pe]=v;}
__device__ uint16_t ResidualSeed(const uint16_t*ah,const uint8_t*head,uint32_t g,uint32_t r,uint32_t c){uint32_t mig=(c/16)*2+(c%16)/8,pairs[4][2]={{0,2},{1,3},{4,6},{5,7}},l=(r%8)*4+(c%8)/2,e=(r>=8?2:0)+(c&1),op=pairs[mig][e/2],ph=e&1;size_t ri=size_t(g)*512+l*16+op*2+ph,si=8208+(mig*4+l%4)*4+ph*2;uint16_t s=uint16_t(head[si])|uint16_t(head[si+1])<<8;return HRound(HVal(ah[ri])*HVal(s));}
__global__ void First128(const uint16_t*ah,const uint8_t*hidden,const uint8_t*head,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t z=i;uint32_t c=uint32_t(z%32);z/=32;uint32_t r=uint32_t(z%16);z/=16;uint32_t g=uint32_t(z%4);size_t cta=z/4;uint16_t ab=ResidualSeed(ah+cta*2048,head,g,r,c);for(uint32_t p=0;p<4;++p){float acc=HVal(ab);size_t hb=(cta*4+p)*2048,tile=4096+size_t(p)*1024+(c/16)*512;for(uint32_t k=0;k<32;++k)acc+=Dec(hidden[hb+PA(g,r,k)])*Dec(head[PB(tile,k,c%16)]);ab=HRound(acc);}out[i]=ab;}

__global__ void Pack176(const uint16_t*in,uint8_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t z=i;uint32_t c=uint32_t(z%32);z/=32;uint32_t r=uint32_t(z%16);z/=16;uint32_t g=uint32_t(z%4);size_t cta=z/4,mig=(c/16)*2+(c%16)/8,c8=c%8,l=(r%8)*4+c8/2,de=(r>=8?2:0)+(c8&1),conv=(mig/2)*4+(mig%2)+(de/2)*2,pe=conv*2+(de&1);out[cta*2048+size_t(g)*512+l*16+pe]=Enc(in[i]);}
__global__ void Project176(const uint8_t*a,const uint8_t*head,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=PROJ176_VALUES)return;size_t z=i;uint32_t c=uint32_t(z%8);z/=8;uint32_t r=uint32_t(z%16);z/=16;uint32_t m=uint32_t(z%48);size_t cta=z/48;uint32_t g=m/12,w=m%12;size_t tile=8400+size_t(w/2)*512;uint32_t oc=(w&1)*8+c;float acc=0;for(uint32_t k=0;k<32;++k)acc+=Dec(a[cta*2048+PA(g,r,k)])*Dec(head[PB(tile,k,oc)]);out[i]=HRound(acc);}

__device__ size_t ProjIndex(size_t cta,uint32_t seg,uint32_t tok,uint32_t ch){uint32_t g=tok/16,r=tok%16,m=g*12+seg*4+(ch/16)*2+(ch%16)/8;return cta*6144+size_t(m)*128+r*8+ch%8;}
__device__ size_t PKey(uint32_t block,uint32_t k,uint32_t c){uint32_t l=c*4+(k%16)/4,e=(k%4)+(k>=16?4:0);return size_t(block)*256+l*8+e;}
__global__ void PrepQK(const uint16_t*p,uint8_t*q,uint8_t*k,uint16_t scale,uint16_t eps){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=CTAS*64)return;size_t cta=i/64;uint32_t tok=uint32_t(i%64);uint16_t qv[32],kv[32],qs[32],ks[32];for(uint32_t c=0;c<32;++c){qv[c]=p[ProjIndex(cta,0,tok,c)];kv[c]=p[ProjIndex(cta,1,tok,c)];qs[c]=HMul(qv[c],qv[c]);ks[c]=HMul(kv[c],kv[c]);}for(uint32_t w=16;w;w>>=1)for(uint32_t j=0;j<w;++j){qs[j]=HAdd(qs[2*j],qs[2*j+1]);ks[j]=HAdd(ks[2*j],ks[2*j+1]);}float ev=HVal(eps);uint16_t qi=HRound(rsqrtf(max(HVal(qs[0]),ev))),ki=HRound(rsqrtf(max(HVal(ks[0]),ev)));uint32_t g=tok/16,r=tok%16,kb=tok/8,kc=tok%8;for(uint32_t pc=0;pc<32;++pc){uint32_t src=Perm32(pc);q[cta*2048+PA(g,r,pc)]=Enc(HMul(HMul(qv[src],qi),scale));k[cta*2048+PKey(kb,pc,kc)]=Enc(HMul(kv[src],ki));}}
__global__ void PrepV(const uint16_t*p,uint8_t*v){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t z=i;uint32_t f=uint32_t(z%32);z/=32;uint32_t dk=uint32_t(z%64);size_t cta=z/64,kchunk=dk/32,kw=dk%32,st=kchunk*32+Perm32(kw),fb=f/8,c=f%8,l=c*4+(kw%16)/4,e=(kw%4)+(kw>=16?4:0);v[cta*2048+size_t(kchunk*4+fb)*256+l*8+e]=Enc(p[ProjIndex(cta,2,st,f)]);}
__global__ void QK(const uint8_t*q,const uint8_t*k,const uint8_t*head,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=QK_VALUES)return;size_t z=i;uint32_t c=uint32_t(z%8);z/=8;uint32_t r=uint32_t(z%16);z/=16;uint32_t m=uint32_t(z%32);size_t cta=z/32,g=m/8,kb=m%8,seed=11472+g*2048+(kb/2)*512,sh=kb&1,l=(r%8)*4+c/2,e=(r>=8?2:0)+(c&1);size_t so=seed+l*16+sh*8+e*2;uint16_t sb=uint16_t(head[so])|uint16_t(head[so+1])<<8;float acc=HVal(sb);for(uint32_t ch=0;ch<32;++ch)acc+=Dec(q[cta*2048+PA(g,r,ch)])*Dec(k[cta*2048+PKey(kb,ch,c)]);out[i]=HRound(acc);}

__device__ size_t PASoft(uint32_t g,uint32_t r,uint32_t key){uint32_t l=(r%8)*4+(key%16)/4,e=(key%4)+(r>=8?4:0)+((key%32)>=16?8:0);return size_t(g)*1024+size_t(key/32)*512+l*16+e;}
__device__ uint16_t Score(const uint16_t*qk,size_t cta,uint32_t query,uint32_t key){uint32_t g=query/16,r=query%16;return qk[cta*4096+size_t(g*8+key/8)*128+r*8+key%8];}
__global__ void Softmax(const uint16_t*qk,uint8_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=CTAS*64)return;size_t cta=i/64;uint32_t query=uint32_t(i%64),g=query/16,r=query%16;float mx=-INFINITY;for(uint32_t key=0;key<64;++key)mx=max(mx,HVal(Score(qk,cta,query,key)));float den=0;for(uint32_t key=0;key<64;++key)den+=expf(HVal(Score(qk,cta,query,key))-mx);for(uint32_t pk=0;pk<64;++pk){uint32_t chunk=pk/32,key=chunk*32+Perm32(pk%32);out[cta*4096+PASoft(g,r,pk)]=Enc(HRound(expf(HVal(Score(qk,cta,query,key))-mx)/den));}}
__device__ size_t PV(uint32_t key,uint32_t f){uint32_t kc=key/32,k=key%32,fb=f/8,c=f%8,l=c*4+(k%16)/4,e=(k%4)+(k>=16?4:0);return size_t(kc*4+fb)*256+l*8+e;}
__global__ void Attention(const uint8_t*soft,const uint8_t*v,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t z=i;uint32_t f=uint32_t(z%32);z/=32;uint32_t query=uint32_t(z%64);size_t cta=z/64;uint32_t g=query/16,r=query%16;float acc=0;for(uint32_t k=0;k<32;++k)acc+=Dec(soft[cta*4096+PASoft(g,r,k)])*Dec(v[cta*2048+PV(k,f)]);uint16_t first=HRound(acc);acc=HVal(first);for(uint32_t k=32;k<64;++k)acc+=Dec(soft[cta*4096+PASoft(g,r,k)])*Dec(v[cta*2048+PV(k,f)]);out[i]=HRound(acc);}

__global__ void FinalPack(const uint16_t*in,uint8_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t z=i;uint32_t ch=uint32_t(z%32);z/=32;uint32_t tok=uint32_t(z%64);size_t cta=z/64;out[cta*2048+PA(tok/16,tok%16,ch)]=Enc(in[cta*2048+size_t(tok)*32+Perm32(ch)]);}
__global__ void FinalProject(const uint8_t*a,const uint16_t*res,const uint8_t*head,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=VALUES)return;size_t z=i;uint32_t oc=uint32_t(z%32);z/=32;uint32_t tok=uint32_t(z%64);size_t cta=z/64;uint32_t g=tok/16,r=tok%16,tile=19680+(oc/16)*512,c=oc%8,nh=(oc%16)/8;uint16_t s=uint16_t(head[20704+oc*2])|uint16_t(head[20705+oc*2])<<8;float acc=HVal(HMul(res[cta*2048+size_t(tok)*32+oc],s));for(uint32_t k=0;k<32;++k)acc+=Dec(a[cta*2048+PA(g,r,k)])*Dec(head[PB(tile,k,c)+nh*8]);out[i]=HRound(acc);}
__device__ uint16_t TailWeight(const uint8_t*h,uint32_t k,uint32_t c){uint32_t tile=20784+(k/16)*512,kk=k%16,l=c*4+(kk%8)/2,e=(kk%2)+(kk>=8?2:0);size_t o=tile+l*16+e*2;return uint16_t(h[o])|uint16_t(h[o+1])<<8;}
__global__ void Tail(const uint16_t*in,const uint8_t*h,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=RESIDUAL_VALUES)return;size_t z=i;uint32_t c=uint32_t(z%4);z/=4;uint32_t tok=uint32_t(z%64);size_t cta=z/64;float acc=0;for(uint32_t k=0;k<16;++k)acc+=HVal(in[cta*2048+size_t(tok)*32+k])*HVal(TailWeight(h,k,c));uint16_t first=HRound(acc);acc=HVal(first);for(uint32_t k=16;k<32;++k)acc+=HVal(in[cta*2048+size_t(tok)*32+k])*HVal(TailWeight(h,k,c));out[i]=HRound(acc);}
__global__ void Surface(const uint16_t*res,const uint16_t*base,uint16_t*out){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=CTAS*64)return;uint32_t tok=uint32_t(i%64);size_t cta=i/64;uint32_t g=tok/16,w=tok%16;int32_t x=int32_t((cta%GW)*8)-4+int32_t((g&1)*4+w%4),y=int32_t((cta/GW)*8)-4+int32_t((g>>1)*4+w/4);if(x<0||y<0||x>=int32_t(WIDTH)||y>=int32_t(HEIGHT))return;size_t d=(size_t(y)*WIDTH+uint32_t(x))*4,s=i*4;for(uint32_t c=0;c<3;++c){float b=base?HVal(base[d+c]):0,r=HVal(res[s+c]);out[d+c]=HRound(fminf(1,fmaxf(0,b+.25f*r)));}out[d+3]=HRound(1);}
}

#ifndef OUTPUT_HEAD_RESIDENT_KERNELS_ONLY
int main(int argc,char**argv){try{
  if(argc<8||argc>9){std::fprintf(stderr,"usage: output_head_resident <activation_arena.raw> <model_arena.raw> <base_rgba16f.raw|-> <reference_residual.raw|-> <residual_out.raw> <surface_out.raw> <result.json> [iterations]\n");return 2;}
  uint32_t it=argc==9?uint32_t(std::stoul(argv[8])):20;if(!it)throw std::runtime_error("iterations must be positive");
  auto main=Slice(argv[1],MAIN_OFF,MAIN_BYTES),skip=Slice(argv[1],SKIP_OFF,SKIP_BYTES),head=Slice(argv[2],HEAD_OFF,HEAD_BYTES);bool hasBase=std::string(argv[3])!="-",hasRef=std::string(argv[4])!="-";auto base=hasBase?Read(argv[3]):std::vector<uint8_t>{},ref=hasRef?Read(argv[4]):std::vector<uint8_t>{};if(hasBase&&base.size()!=SURFACE_VALUES*2)throw std::runtime_error("unexpected base size");if(hasRef&&ref.size()!=RESIDUAL_VALUES*2)throw std::runtime_error("unexpected reference size");
  uint8_t *dm=nullptr,*ds=nullptr,*dh=nullptr,*da=nullptr,*dhidden=nullptr,*dq=nullptr,*dk=nullptr,*dv=nullptr,*dsoft=nullptr;uint16_t *dah=nullptr,*dfirst=nullptr,*dp176=nullptr,*dqk=nullptr,*datt=nullptr,*dproj=nullptr,*dres=nullptr,*dbase=nullptr,*dsurf=nullptr;
  HIP_CHECK(hipMalloc(&dm,main.size()));HIP_CHECK(hipMalloc(&ds,skip.size()));HIP_CHECK(hipMalloc(&dh,head.size()));HIP_CHECK(hipMalloc(&da,A_BYTES));HIP_CHECK(hipMalloc(&dah,H_BYTES));HIP_CHECK(hipMalloc(&dhidden,HIDDEN_BYTES));HIP_CHECK(hipMalloc(&dfirst,H_BYTES));HIP_CHECK(hipMalloc(&dp176,PROJ176_VALUES*2));HIP_CHECK(hipMalloc(&dq,A_BYTES));HIP_CHECK(hipMalloc(&dk,A_BYTES));HIP_CHECK(hipMalloc(&dv,A_BYTES));HIP_CHECK(hipMalloc(&dqk,QK_VALUES*2));HIP_CHECK(hipMalloc(&dsoft,SOFT_BYTES));HIP_CHECK(hipMalloc(&datt,H_BYTES));HIP_CHECK(hipMalloc(&dproj,H_BYTES));HIP_CHECK(hipMalloc(&dres,RESIDUAL_VALUES*2));HIP_CHECK(hipMalloc(&dsurf,SURFACE_VALUES*2));
  HIP_CHECK(hipMemcpy(dm,main.data(),main.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(ds,skip.data(),skip.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dh,head.data(),head.size(),hipMemcpyHostToDevice));if(hasBase){HIP_CHECK(hipMalloc(&dbase,base.size()));HIP_CHECK(hipMemcpy(dbase,base.data(),base.size(),hipMemcpyHostToDevice));}
  float sf;std::memcpy(&sf,head.data()+19664,4);uint16_t scale=__half_as_ushort(__float2half_rn(sf)),eps=__half_as_ushort(__float2half_rn(6.199999916134402e-05f));auto grid=[](size_t n){return dim3(uint32_t((n+255)/256));};dim3 block(256);
  const char* debugEnv=std::getenv("OUTPUT_HEAD_RESIDENT_DEBUG_DIR");
  const std::filesystem::path debugDir=debugEnv?std::filesystem::path(debugEnv):std::filesystem::path{};
  auto dump=[&](const char*name,const void*device,size_t bytes){if(debugDir.empty())return;std::vector<uint8_t>host(bytes);HIP_CHECK(hipMemcpy(host.data(),device,bytes,hipMemcpyDeviceToHost));Write(debugDir/name,host.data(),bytes);};
  auto run=[&](bool capture=false){
    Activation<<<grid(VALUES),block>>>(dm,ds,dh,da,dah);if(capture){dump("activation_e4m3.raw",da,A_BYTES);dump("activation_fp16.raw",dah,H_BYTES);}
    Hidden<<<grid(VALUES*4),block>>>(da,dh,dhidden);First128<<<grid(VALUES),block>>>(dah,dhidden,dh,dfirst);if(capture)dump("first128_fp16.raw",dfirst,H_BYTES);
    Pack176<<<grid(VALUES),block>>>(dfirst,da);Project176<<<grid(PROJ176_VALUES),block>>>(da,dh,dp176);if(capture)dump("mma128_175_fp16.raw",dp176,PROJ176_VALUES*2);
    PrepQK<<<grid(CTAS*64),block>>>(dp176,dq,dk,scale,eps);PrepV<<<grid(VALUES),block>>>(dp176,dv);QK<<<grid(QK_VALUES),block>>>(dq,dk,dh,dqk);if(capture){dump("q_e4m3.raw",dq,A_BYTES);dump("k_e4m3.raw",dk,A_BYTES);dump("v_e4m3.raw",dv,A_BYTES);dump("qk_fp16.raw",dqk,QK_VALUES*2);}
    Softmax<<<grid(CTAS*64),block>>>(dqk,dsoft);Attention<<<grid(VALUES),block>>>(dsoft,dv,datt);if(capture){dump("softmax_e4m3.raw",dsoft,SOFT_BYTES);dump("attention_fp16.raw",datt,H_BYTES);}
    FinalPack<<<grid(VALUES),block>>>(datt,dq);FinalProject<<<grid(VALUES),block>>>(dq,dfirst,dh,dproj);if(capture)dump("attention_projected_fp16.raw",dproj,H_BYTES);
    Tail<<<grid(RESIDUAL_VALUES),block>>>(dproj,dh,dres);if(capture)dump("rgba_residual_fp16.raw",dres,RESIDUAL_VALUES*2);
    Surface<<<grid(CTAS*64),block>>>(dres,dbase,dsurf);if(capture)dump("output_rgba16f.raw",dsurf,SURFACE_VALUES*2);
  };
  run(true);HIP_CHECK(hipDeviceSynchronize());std::vector<uint16_t>res(RESIDUAL_VALUES),surf(SURFACE_VALUES),rep(RESIDUAL_VALUES);HIP_CHECK(hipMemcpy(res.data(),dres,res.size()*2,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(surf.data(),dsurf,surf.size()*2,hipMemcpyDeviceToHost));run();HIP_CHECK(hipDeviceSynchronize());HIP_CHECK(hipMemcpy(rep.data(),dres,rep.size()*2,hipMemcpyDeviceToHost));hipEvent_t st{},sp{};HIP_CHECK(hipEventCreate(&st));HIP_CHECK(hipEventCreate(&sp));HIP_CHECK(hipEventRecord(st));for(uint32_t i=0;i<it;++i)run();HIP_CHECK(hipEventRecord(sp));HIP_CHECK(hipEventSynchronize(sp));float total=0;HIP_CHECK(hipEventElapsedTime(&total,st,sp));
  uint64_t refMis=0,nf=0,changed=0;if(hasRef){auto*r=reinterpret_cast<const uint16_t*>(ref.data());for(size_t i=0;i<res.size();++i)refMis+=res[i]!=r[i];}for(size_t i=0;i<surf.size();++i)nf+=!std::isfinite(__half2float(__ushort_as_half(surf[i])));if(hasBase){auto*b=reinterpret_cast<const uint16_t*>(base.data());for(size_t p=0;p<SURFACE_VALUES/4;++p)for(uint32_t c=0;c<3;++c)changed+=surf[p*4+c]!=b[p*4+c];}else for(size_t p=0;p<SURFACE_VALUES/4;++p)for(uint32_t c=0;c<3;++c)changed+=surf[p*4+c]!=0;
  bool det=res==rep,pass=det&&nf==0&&changed>0&&(!hasRef||refMis==0);Write(argv[5],res.data(),res.size()*2);Write(argv[6],surf.data(),surf.size()*2);char rh[32]{},sh[32]{};std::snprintf(rh,sizeof(rh),"%016llX",(unsigned long long)Hash(res.data(),res.size()*2));std::snprintf(sh,sizeof(sh),"%016llX",(unsigned long long)Hash(surf.data(),surf.size()*2));hipDeviceProp_t prop{};HIP_CHECK(hipGetDeviceProperties(&prop,0));std::string j="{\n  \"schema\": 1,\n  \"experiment\": \"amd_output_head_resident\",\n  \"status\": \""+std::string(pass?"PASS":"FAIL")+"\",\n  \"classification\": \"AMD_NATIVE_RESIDENT_OUTPUT_HEAD\",\n  \"device\": \""+std::string(prop.name)+"\",\n  \"resident_gpu_intermediates\": true,\n  \"runtime_rtx_trace_dependency\": false,\n  \"stages_fused_in_process\": 8,\n  \"kernel_launches_per_frame\": 14,\n  \"reference_residual_supplied\": "+std::string(hasRef?"true":"false")+",\n  \"reference_half_mismatches\": "+std::to_string(refMis)+",\n  \"deterministic_repeat\": "+std::string(det?"true":"false")+",\n  \"non_finite_surface_components\": "+std::to_string(nf)+",\n  \"changed_rgb_components\": "+std::to_string(changed)+",\n  \"iterations\": "+std::to_string(it)+",\n  \"average_gpu_ms\": "+std::to_string(total/float(it))+",\n  \"residual_fnv1a64\": \""+rh+"\",\n  \"surface_fnv1a64\": \""+sh+"\",\n  \"next_gate\": \"Import D3D12 base/output allocations directly into this resident process\"\n}\n";Write(argv[7],j.data(),j.size());std::printf("[%s] %s resident output head: ref=%llu changed=%llu %.6f ms\n",pass?"PASS":"FAIL",prop.name,(unsigned long long)refMis,(unsigned long long)changed,total/float(it));
  hipEventDestroy(st);hipEventDestroy(sp);hipFree(dsurf);if(dbase)hipFree(dbase);hipFree(dres);hipFree(dproj);hipFree(datt);hipFree(dsoft);hipFree(dqk);hipFree(dv);hipFree(dk);hipFree(dq);hipFree(dp176);hipFree(dfirst);hipFree(dhidden);hipFree(dah);hipFree(da);hipFree(dh);hipFree(ds);hipFree(dm);return pass?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}}
#endif
