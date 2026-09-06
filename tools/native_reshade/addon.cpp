#include "gpu_surface.h"
#include "worker_pipe.h"
#include "reshade.hpp"
#include <mutex>
using namespace native_preview;
namespace {
std::filesystem::path repo,run;std::string mode;UINT limit=1,width=0,height=0,frames=0,expectedFormat=0;bool failed=false,finished=false,seen=false,screenshot=false,pendingShot=false;
Surface* surface=nullptr;Worker* worker=nullptr;reshade::api::effect_runtime* selected=nullptr;DWORD owner=0;std::mutex mutex;
std::vector<reshade::api::swapchain*> swaps;
UINT64 presentIndex=0;
void presented(reshade::api::effect_runtime* rt){std::lock_guard lock(mutex);if(pendingShot&&rt==selected){pendingShot=false;rt->save_screenshot("NativeROCm");std::ofstream(run/L"addon.jsonl",std::ios::app)<<"{\"event\":\"processed_frame_reached_reshade_present\",\"present_index\":"<<presentIndex<<",\"screenshot_requested\":true}\n";}++presentIndex;}
void initSwap(reshade::api::swapchain* s,bool){std::lock_guard lock(mutex);if(swaps.size()<8)swaps.push_back(s);else failed=true;}
void destroySwap(reshade::api::swapchain* s,bool){std::lock_guard lock(mutex);swaps.erase(std::remove(swaps.begin(),swaps.end(),s),swaps.end());}
reshade::api::color_space color(reshade::api::effect_runtime* rt){
    reshade::api::swapchain* match=nullptr;for(auto* s:swaps)if(s->get_hwnd()==rt->get_hwnd()&&s->get_current_back_buffer().handle==rt->get_current_back_buffer().handle){if(match)return reshade::api::color_space::unknown;match=s;}
    return match?match->get_color_space():reshade::api::color_space::unknown;
}
std::wstring env(const wchar_t* n){wchar_t b[32768]{};auto z=GetEnvironmentVariableW(n,b,32768);if(!z||z>=32768)throw std::runtime_error("explicit preview environment missing");return b;}
void log(const std::string& m){if(!run.empty())std::ofstream(run/L"addon.jsonl",std::ios::app)<<m<<'\n';}
void fail(const char* why){failed=true;std::string s=why;for(auto& c:s)if(c=='"'||c=='\\'||c<' ')c='_';log("{\"event\":\"failure\",\"reason\":\""+s+"\",\"automatic_retry\":false}");}
void effects(reshade::api::effect_runtime* rt,reshade::api::command_list*,reshade::api::resource_view rtv,reshade::api::resource_view){
    std::unique_lock lock(mutex,std::try_to_lock);if(!lock.owns_lock()||failed||finished)return;
    try{
        if(run.empty()){repo=env(L"NR_PREVIEW_REPO");run=env(L"NR_PREVIEW_RUN");auto m=env(L"NR_PREVIEW_MODE");mode=std::string(m.begin(),m.end());
            width=std::stoul(env(L"NR_PREVIEW_WIDTH"));height=std::stoul(env(L"NR_PREVIEW_HEIGHT"));limit=std::stoul(env(L"NR_PREVIEW_FRAMES"));
            expectedFormat=std::stoul(env(L"NR_PREVIEW_FORMAT"));screenshot=env(L"NR_PREVIEW_SCREENSHOT")==L"1";
            if((mode!="observe"&&mode!="transport"&&mode!="network")||!width||!height||width>3840||height>2160||(limit!=1&&limit!=12)||!std::filesystem::is_directory(run))throw std::runtime_error("preview config invalid");}
        if(rt->get_device()->get_api()!=reshade::api::device_api::d3d12)throw std::runtime_error("D3D12 only");
        auto* d=reinterpret_cast<ID3D12Device*>(rt->get_device()->get_native());auto* q=reinterpret_cast<ID3D12CommandQueue*>(rt->get_command_queue()->get_native());
        auto* target=reinterpret_cast<ID3D12Resource*>(rt->get_device()->get_resource_from_view(rtv).handle);if(!target)throw std::runtime_error("effect target unavailable");auto td=target->GetDesc();
        if(!seen){seen=true;log("{\"event\":\"effect_target\",\"width\":"+std::to_string(td.Width)+",\"height\":"+std::to_string(td.Height)+",\"format\":"+std::to_string(td.Format)+",\"color_space\":"+std::to_string(int(color(rt)))+",\"network_enabled\":false}");}
        if(mode=="observe"||!std::filesystem::exists(run/L"ARM"))return;
        if(td.Width!=width||td.Height!=height||td.Format!=expectedFormat||!Surface::supported(td.Format)||color(rt)!=reshade::api::color_space::srgb_nonlinear)throw std::runtime_error("unapproved dimensions/format or HDR; original output preserved");
        if(selected&&(selected!=rt||owner!=GetCurrentThreadId()))throw std::runtime_error("runtime/thread changed");selected=rt;owner=GetCurrentThreadId();
        if(!worker){worker=new Worker;worker->launch(repo,run,mode=="network",limit,width,height);surface=new Surface;surface->init(d,q,width,height,td.Format,worker->luid);worker->prepare(surface->bridge);log("{\"event\":\"worker_ready\",\"gpu_pixels_only\":true}");}
        // ReShade owns this safe flush boundary; no surgery on game's command lists.
        rt->get_command_queue()->flush_immediate_command_list();
        surface->input(target);worker->frame(surface->bridge);surface->output(target);worker->consumed(surface->bridge);++frames;pendingShot=screenshot&&frames==1;
        log("{\"event\":\"frame_written\",\"sequence\":"+std::to_string(frames)+",\"present_index\":"+std::to_string(presentIndex)+",\"width\":"+std::to_string(width)+",\"height\":"+std::to_string(height)+",\"same_effect_frame\":true,\"gpu_pixels_only\":true,\"hud_protected\":false,\"realtime_accepted\":false}");
        if(frames>=limit){worker->close();surface->bridge.close();delete surface;delete worker;surface=nullptr;worker=nullptr;finished=true;log("{\"event\":\"bounded_session_complete\",\"frames\":"+std::to_string(frames)+",\"original_path_restored\":true}");}
        else{if(nr_bridge_next(&surface->bridge))throw std::runtime_error(nr_bridge_error());}
    }catch(const std::exception& e){fail(e.what());}catch(...){fail("unknown exception; retained resources");}
}
void destroyed(reshade::api::effect_runtime* rt){std::lock_guard lock(mutex);seen=false;if(rt==selected&&!finished){fail("runtime destroyed before bounded completion; resources retained");}}
}
extern "C" __declspec(dllexport) const char* NAME="Native ROCm SDR Preview";
extern "C" __declspec(dllexport) const char* DESCRIPTION="Bounded same-frame GPU preview; not HDR, HUD protection or realtime acceptance.";
BOOL WINAPI DllMain(HMODULE module,DWORD reason,LPVOID){
    if(reason==DLL_PROCESS_ATTACH){
        if(!reshade::register_addon(module))return FALSE;
        reshade::register_event<reshade::addon_event::init_swapchain>(initSwap);
        reshade::register_event<reshade::addon_event::destroy_swapchain>(destroySwap);
        reshade::register_event<reshade::addon_event::reshade_finish_effects>(effects);
        reshade::register_event<reshade::addon_event::reshade_present>(presented);
        reshade::register_event<reshade::addon_event::destroy_effect_runtime>(destroyed);
    }else if(reason==DLL_PROCESS_DETACH){
        reshade::unregister_event<reshade::addon_event::init_swapchain>(initSwap);
        reshade::unregister_event<reshade::addon_event::destroy_swapchain>(destroySwap);
        reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(effects);
        reshade::unregister_event<reshade::addon_event::reshade_present>(presented);
        reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(destroyed);
        reshade::unregister_addon(module);
    }return TRUE;
}
