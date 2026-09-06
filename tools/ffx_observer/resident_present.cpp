#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "present_surface.h"
#include "ffx_observer.h"
#include "capture_session.h"
#include <dxgi1_6.h>
#include <atomic>
#include <mutex>
#include <map>
#include <memory>
#include <cstdio>

// Separate from FFX hooks: explicit pre-entry install, dormant until command 19.
// Factory creation supplies the EXACT presentation queue. No device-only guess.
namespace {
using Microsoft::WRL::ComPtr;using ffx_capture::Check;
using Create0=HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*,IUnknown*,DXGI_SWAP_CHAIN_DESC*,IDXGISwapChain**);
using Create1=HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*,IUnknown*,HWND,const DXGI_SWAP_CHAIN_DESC1*,const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*,IDXGIOutput*,IDXGISwapChain1**);
using Present0=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*,UINT,UINT);
using Present1=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*,UINT,UINT,const DXGI_PRESENT_PARAMETERS*);
using Resize0=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*,UINT,UINT,UINT,DXGI_FORMAT,UINT);
using Resize1=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*,UINT,UINT,UINT,DXGI_FORMAT,UINT,const UINT*,IUnknown*const*);
using Color=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*,DXGI_COLOR_SPACE_TYPE);
struct FactoryTable {Create0 c0;Create1 c1;};
struct SwapTable {Present0 p0;Present1 p1;Resize0 r0;Resize1 r1;Color color;};
struct Binding {ComPtr<IDXGISwapChain3> swap;ComPtr<ID3D12CommandQueue> queue;ComPtr<IUnknown> identity;DXGI_COLOR_SPACE_TYPE color=DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;HWND hwnd=nullptr;};
std::recursive_mutex serial;std::mutex logs;
std::map<void**,FactoryTable> factoryTables;std::map<void**,SwapTable> swapTables;
std::map<IUnknown*,Binding> bindings;ComPtr<IDXGIFactory2> factoryProbe;
std::atomic<bool> enabled{false},failed{false},busy{false};std::atomic<unsigned> frames{0};std::atomic<ULONGLONG> deadline{0};
bool installed=false,everStarted=false;std::filesystem::path root,nameFile;FILE* logFile=nullptr;
ffx_resident::Exchange* exchange=nullptr;IUnknown* selected=nullptr;
thread_local unsigned createDepth=0,presentDepth=0;
struct Depth {unsigned& n;Depth(unsigned& v):n(v){++n;}~Depth(){--n;}};
void Emit(const std::string& s){std::lock_guard<std::mutex> lock(logs);if(logFile){std::fprintf(logFile,"%s\n",s.c_str());std::fflush(logFile);}}
void Fail(const char* why){enabled=false;failed=true;std::string safe=why;for(auto& c:safe)if(c=='"'||c=='\\'||c<' ')c='_';Emit("{\"event\":\"failure\",\"reason\":\""+safe+"\"}");}
void Patch(void** slot,void* old,void* hook){DWORD p=0,unused=0;if(!VirtualProtect(slot,sizeof(void*),PAGE_READWRITE,&p))throw std::runtime_error("present hook page protect");auto found=InterlockedCompareExchangePointer(slot,hook,old);bool restored=VirtualProtect(slot,sizeof(void*),p,&unused)!=0;if(found!=old||!restored)throw std::runtime_error("present hook publication failed");}
template<class T> T Read(void* p){T t{};SIZE_T n=0;if(!p||!ReadProcessMemory(GetCurrentProcess(),p,&t,sizeof(t),&n)||n!=sizeof(t))throw std::invalid_argument("present argument unreadable");return t;}
void Status(){std::lock_guard<std::mutex> lock(logs);std::ofstream f(root/L"resident_status.json");f<<"{\"enabled\":"<<(enabled?"true":"false")<<",\"failed\":"<<(failed?"true":"false")<<",\"busy\":"<<(busy?"true":"false")<<",\"completed_frames\":"<<frames.load()<<",\"display_referred_debug_route\":true,\"dlss5_quality_verified\":false}\n";}
void Save(const nr_present::Surface::Result& r,unsigned seq){
    if(seq>2)return;auto dir=root/L"present_frames";std::filesystem::create_directories(dir);
    auto write=[&](const wchar_t* name,const std::vector<uint8_t>& bytes){std::ofstream f(dir/(std::wstring(name)+std::to_wstring(seq)+L".raw"),std::ios::binary);f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());if(!f)throw std::runtime_error("present audit write");};
    write(L"input",r.input);write(L"output",r.output);write(L"encoded_input",r.encodedInput);write(L"encoded_output",r.encodedOutput);
}
struct Work {bool processed=false;unsigned seq=0,index=0;UINT w=0,h=0;DXGI_FORMAT format{};};
Work Process(IDXGISwapChain* swap,UINT flags,bool partial){
    Work work;if(!enabled||failed)return work;
    if(GetTickCount64()>=deadline.load()||frames>=12){enabled=false;Status();return work;}
    if(flags&DXGI_PRESENT_TEST)return work;
    if((flags&(DXGI_PRESENT_DO_NOT_WAIT|DXGI_PRESENT_DO_NOT_SEQUENCE|DXGI_PRESENT_RESTART))||partial){Fail("unsupported nonstandard present flags/partial update");return work;}
    ComPtr<IUnknown> identity;Check(swap->QueryInterface(IID_PPV_ARGS(&identity)));auto it=bindings.find(identity.Get());
    if(it==bindings.end()){Fail("present queue binding not observed");return work;}
    auto& b=it->second;DWORD pid=0;GetWindowThreadProcessId(b.hwnd,&pid);if(pid!=GetCurrentProcessId()||!IsWindowVisible(b.hwnd)||IsIconic(b.hwnd))return work;
    if(selected&&selected!=identity.Get()){Fail("multiple active swapchains");return work;}
    if(b.color!=DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709){Emit("{\"event\":\"unsupported_color_space\",\"dxgi_color_space\":"+std::to_string(b.color)+"}");Fail("HDR or unknown color contract unsupported");return work;}
    selected=identity.Get();ComPtr<ID3D12Device> device;Check(b.queue->GetDevice(IID_PPV_ARGS(&device)));auto luid=device->GetAdapterLuid();ComPtr<IDXGIFactory4> f;Check(CreateDXGIFactory1(IID_PPV_ARGS(&f)));ComPtr<IDXGIAdapter1> adapter;Check(f->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter)));DXGI_ADAPTER_DESC1 ad{};Check(adapter->GetDesc1(&ad));
    // WARP is admitted only for a specifically named standalone fixture.
    wchar_t exe[32768]{};GetModuleFileNameW(nullptr,exe,32768);bool fixture=std::filesystem::path(exe).filename()==L"resident_present_selftest.exe";
    if(ad.VendorId!=0x1002&&!fixture){Fail("game adapter not AMD");return work;}
    work.index=b.swap->GetCurrentBackBufferIndex();ComPtr<ID3D12Resource> target;Check(b.swap->GetBuffer(work.index,IID_PPV_ARGS(&target)));
    nr_present::Surface* surface=nullptr;busy=true;
    try{surface=new nr_present::Surface(device.Get(),target.Get());auto result=surface->Run(b.queue.Get(),*exchange);delete surface;surface=nullptr;
        work.seq=unsigned(exchange->Sequence());work.w=result.width;work.h=result.height;work.format=result.format;Save(result,work.seq);work.processed=true;
    }catch(...){if(surface&&surface->SafeToDestroy())delete surface;busy=false;throw;}
    busy=false;return work;
}
void Complete(const Work& work,HRESULT hr){if(!work.processed)return;if(hr!=S_OK){Fail("Present did not accept processed frame");return;}++frames;
    Emit("{\"event\":\"resident_present_complete\",\"request_id\":"+std::to_string(work.seq)+",\"buffer_index\":"+std::to_string(work.index)+",\"width\":"+std::to_string(work.w)+",\"height\":"+std::to_string(work.h)+",\"format\":"+std::to_string(work.format)+",\"encoded_readback_exact\":true,\"present_hresult\":0,\"same_present_frame\":true,\"display_referred_debug_route\":true,\"dlss5_quality_verified\":false}");
    if(frames>=12)enabled=false;Status();
}
HRESULT STDMETHODCALLTYPE Present(IDXGISwapChain* s,UINT interval,UINT flags){
    std::unique_lock<std::recursive_mutex> lock(serial);auto original=swapTables.at(*reinterpret_cast<void***>(s)).p0;
    if(presentDepth){lock.unlock();return original(s,interval,flags);}Depth depth(presentDepth);Work work;
    try{work=Process(s,flags,false);}catch(const std::exception& e){Fail(e.what());}catch(...){Fail("present processing unknown failure");}
    // DXGI Present may depend on the window/message thread. Never carry our
    // observer mutex into that call; in particular, do not block a UI resize.
    lock.unlock();auto hr=original(s,interval,flags);try{Complete(work,hr);}catch(...){Fail("present completion audit failure");}return hr;
}
HRESULT STDMETHODCALLTYPE PresentEx(IDXGISwapChain1* s,UINT interval,UINT flags,const DXGI_PRESENT_PARAMETERS* p){
    std::unique_lock<std::recursive_mutex> lock(serial);auto original=swapTables.at(*reinterpret_cast<void***>(s)).p1;
    if(presentDepth){lock.unlock();return original(s,interval,flags,p);}Depth depth(presentDepth);Work work;
    try{auto a=Read<DXGI_PRESENT_PARAMETERS>(const_cast<DXGI_PRESENT_PARAMETERS*>(p));work=Process(s,flags,a.DirtyRectsCount||a.pScrollRect||a.pScrollOffset);}catch(const std::exception& e){Fail(e.what());}catch(...){Fail("present1 unknown failure");}
    lock.unlock();auto hr=original(s,interval,flags,p);try{Complete(work,hr);}catch(...){Fail("present1 completion audit failure");}return hr;
}
HRESULT STDMETHODCALLTYPE Resize(IDXGISwapChain* s,UINT n,UINT w,UINT h,DXGI_FORMAT f,UINT flags){std::lock_guard<std::recursive_mutex> lock(serial);enabled=false;Emit("{\"event\":\"resize_stop\"}");return swapTables.at(*reinterpret_cast<void***>(s)).r0(s,n,w,h,f,flags);}
HRESULT STDMETHODCALLTYPE ResizeEx(IDXGISwapChain3* s,UINT n,UINT w,UINT h,DXGI_FORMAT f,UINT flags,const UINT* masks,IUnknown*const* queues){std::lock_guard<std::recursive_mutex> lock(serial);enabled=false;Emit("{\"event\":\"resize1_stop_queue_binding_invalidated\"}");ComPtr<IUnknown> identity;if(SUCCEEDED(s->QueryInterface(IID_PPV_ARGS(&identity)))){if(selected==identity.Get())selected=nullptr;bindings.erase(identity.Get());}return swapTables.at(*reinterpret_cast<void***>(s)).r1(s,n,w,h,f,flags,masks,queues);}
HRESULT STDMETHODCALLTYPE SetColor(IDXGISwapChain3* s,DXGI_COLOR_SPACE_TYPE c){std::lock_guard<std::recursive_mutex> lock(serial);auto hr=swapTables.at(*reinterpret_cast<void***>(s)).color(s,c);try{if(SUCCEEDED(hr)){ComPtr<IUnknown> id;Check(s->QueryInterface(IID_PPV_ARGS(&id)));auto it=bindings.find(id.Get());if(it!=bindings.end())it->second.color=c;if(c!=DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)enabled=false;Emit("{\"event\":\"color_space_set\",\"dxgi_color_space\":"+std::to_string(c)+"}");}}catch(...){Fail("color-space observation failure");}return hr;}
void Bind(IUnknown* object,IUnknown* device,HWND hwnd){
    if(!object||!device||!hwnd)return;ComPtr<ID3D12CommandQueue> queue;if(FAILED(device->QueryInterface(IID_PPV_ARGS(&queue)))||queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)return;
    Binding b;Check(object->QueryInterface(IID_PPV_ARGS(&b.swap)));Check(object->QueryInterface(IID_PPV_ARGS(&b.identity)));b.queue=queue;b.hwnd=hwnd;
    ComPtr<ID3D12Device> qd,sd;Check(queue->GetDevice(IID_PPV_ARGS(&qd)));Check(b.swap->GetDevice(IID_PPV_ARGS(&sd)));if(!ffx_capture::Same(qd.Get(),sd.Get()))throw std::runtime_error("swap/queue device differs");
    if(bindings.count(b.identity.Get()))return;if(bindings.size()>=8)throw std::runtime_error("swapchain cap");
    auto** table=*reinterpret_cast<void***>(b.swap.Get());if(!swapTables.count(table)){
        SwapTable original{reinterpret_cast<Present0>(table[8]),reinterpret_cast<Present1>(table[22]),reinterpret_cast<Resize0>(table[13]),reinterpret_cast<Resize1>(table[39]),reinterpret_cast<Color>(table[38])};swapTables.emplace(table,original);
        Patch(table+8,reinterpret_cast<void*>(original.p0),reinterpret_cast<void*>(Present));Patch(table+22,reinterpret_cast<void*>(original.p1),reinterpret_cast<void*>(PresentEx));Patch(table+13,reinterpret_cast<void*>(original.r0),reinterpret_cast<void*>(Resize));Patch(table+39,reinterpret_cast<void*>(original.r1),reinterpret_cast<void*>(ResizeEx));Patch(table+38,reinterpret_cast<void*>(original.color),reinterpret_cast<void*>(SetColor));
    }
    // DXGI uses one inherited vtable for these interfaces in supported builds.
    ComPtr<IDXGISwapChain> base;ComPtr<IDXGISwapChain1> one;Check(b.swap.As(&base));Check(b.swap.As(&one));if(*reinterpret_cast<void***>(base.Get())!=table||*reinterpret_cast<void***>(one.Get())!=table)throw std::runtime_error("distinct swap interface tables unsupported");
    DXGI_SWAP_CHAIN_DESC1 d{};Check(b.swap->GetDesc1(&d));auto key=b.identity.Get();bindings.emplace(key,std::move(b));
    Emit("{\"event\":\"swapchain_bound\",\"width\":"+std::to_string(d.Width)+",\"height\":"+std::to_string(d.Height)+",\"format\":"+std::to_string(d.Format)+",\"exact_creation_queue\":true}");
}
HRESULT STDMETHODCALLTYPE CreateLegacy(IDXGIFactory* f,IUnknown* d,DXGI_SWAP_CHAIN_DESC* desc,IDXGISwapChain** out){std::lock_guard<std::recursive_mutex> lock(serial);Depth depth(createDepth);auto hr=factoryTables.at(*reinterpret_cast<void***>(f)).c0(f,d,desc,out);if(SUCCEEDED(hr)&&createDepth==1)try{Bind(out?*out:nullptr,d,desc?desc->OutputWindow:nullptr);}catch(const std::exception& e){Fail(e.what());}return hr;}
HRESULT STDMETHODCALLTYPE CreateHwnd(IDXGIFactory2* f,IUnknown* d,HWND hwnd,const DXGI_SWAP_CHAIN_DESC1* desc,const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* full,IDXGIOutput* output,IDXGISwapChain1** out){std::lock_guard<std::recursive_mutex> lock(serial);Depth depth(createDepth);auto hr=factoryTables.at(*reinterpret_cast<void***>(f)).c1(f,d,hwnd,desc,full,output,out);if(SUCCEEDED(hr)&&createDepth==1)try{Bind(out?*out:nullptr,d,hwnd);}catch(const std::exception& e){Fail(e.what());}return hr;}
}
extern "C" __declspec(dllexport) DWORD WINAPI FfxObserver_Bootstrap(void* pointer){
    std::lock_guard<std::recursive_mutex> lock(serial);if(installed)return 0;
    try{auto c=Read<FfxObserverBootstrapV2>(pointer);if(c.size!=sizeof(c)||c.version!=2||c.profile>1||c.log_path[1023])return 0;root=std::filesystem::path(c.log_path).parent_path();if(!root.is_absolute())return 0;
        wchar_t selfPath[32768]{};HMODULE self=nullptr;if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,reinterpret_cast<LPCWSTR>(&FfxObserver_Bootstrap),&self))return 0;GetModuleFileNameW(self,selfPath,32768);nameFile=std::filesystem::path(selfPath).parent_path()/L"worker_name.txt";
        HANDLE file=CreateFileW(c.log_path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(file==INVALID_HANDLE_VALUE)return 0;CloseHandle(file);logFile=_wfsopen(c.log_path,L"ab",_SH_DENYWR);if(!logFile)return 0;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factoryProbe)));auto** table=*reinterpret_cast<void***>(factoryProbe.Get());factoryTables.emplace(table,FactoryTable{reinterpret_cast<Create0>(table[10]),reinterpret_cast<Create1>(table[15])});auto& old=factoryTables.at(table);
        Patch(table+10,reinterpret_cast<void*>(old.c0),reinterpret_cast<void*>(CreateLegacy));Patch(table+15,reinterpret_cast<void*>(old.c1),reinterpret_cast<void*>(CreateHwnd));installed=true;Emit("{\"event\":\"present_observer_installed\",\"network_enabled\":false}");Status();return 1;
    }catch(const std::exception& e){Fail(e.what());return 0;}catch(...){return 0;}
}
extern "C" __declspec(dllexport) DWORD WINAPI FfxSession_Command(void* pointer){
    try{auto c=Read<FfxSessionCommandV1>(pointer);if(c.size!=sizeof(c)||!installed)return 0;
        if(c.command==20){enabled=false;Status();return 1;}if(c.command==21){Status();return 1;}
        if(c.command!=19)return 0;std::lock_guard<std::recursive_mutex> lock(serial);if(everStarted||failed||enabled||bindings.empty())return 0;
        exchange=new ffx_resident::Exchange(nameFile);everStarted=true;frames=0;deadline=GetTickCount64()+180000;enabled=true;Emit("{\"event\":\"resident_present_start\",\"frame_limit\":12,\"duration_ms\":180000}");Status();return 1;
    }catch(const std::exception& e){Fail(e.what());return 3;}catch(...){Fail("present control error");return 3;}
}
