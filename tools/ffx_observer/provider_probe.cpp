#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs=std::filesystem;
void Check(HRESULT h) { if(FAILED(h)) throw std::runtime_error("D3D12 setup failed"); }
std::string Quote(const char* s) {
    std::ostringstream out; out << '"';
    if(s) for(unsigned i=0;s[i] && i<256;++i) {
        unsigned char c=s[i];
        if(c=='"'||c=='\\') out << '\\' << c;
        else if(c<32 || c>=127) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
int wmain(int argc,wchar_t** argv) { try {
    if(argc!=3) throw std::invalid_argument("usage: ffx_provider_probe <verified AMD DLL> <new result.json>");
    fs::path library=fs::absolute(argv[1]),output=fs::absolute(argv[2]);
    if(fs::exists(output)) throw std::invalid_argument("result already exists");
    // No game EXE, proxy DLL, contexts, dispatch, queue submission or resource
    // capture. Only the five-export signed AMD provider is loaded for queries.
    HMODULE backend=LoadLibraryExW(library.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!backend) throw std::runtime_error("provider load failed");
    auto query=reinterpret_cast<PfnFfxQuery>(GetProcAddress(backend,"ffxQuery"));
    if(!query) throw std::runtime_error("ffxQuery absent");
    ComPtr<IDXGIFactory6> factory; Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 desc{}; bool found=false;
    for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i,adapter.Reset()) {
        Check(adapter->GetDesc1(&desc)); if(!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)&&desc.VendorId==0x1002){found=true;break;}
    }
    if(!found) throw std::runtime_error("AMD adapter absent");
    ComPtr<ID3D12Device> device; Check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)));
    uint64_t count=0;
    ffxQueryDescGetVersions versions{}; versions.header.type=FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE; versions.device=device.Get(); versions.outputCount=&count;
    auto versionStatus=query(nullptr,&versions.header);
    std::vector<uint64_t> ids; std::vector<const char*> names; uint32_t detailStatus=versionStatus;
    if(versionStatus==0 && count>0 && count<=32) {
        ids.resize(count);names.resize(count);versions.versionIds=ids.data();versions.versionNames=names.data();
        detailStatus=query(nullptr,&versions.header);
        if(count>ids.size()) throw std::runtime_error("provider exceeded version array capacity");
    }
    int32_t phase=-1;
    ffxQueryDescUpscaleGetJitterPhaseCount jitter{};jitter.header.type=FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    jitter.renderWidth=2560;jitter.displayWidth=3840;jitter.pOutPhaseCount=&phase;
    const auto phaseStatus=query(nullptr,&jitter.header);
    float x=0,y=0;
    ffxQueryDescUpscaleGetJitterOffset offset{};offset.header.type=FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
    offset.index=0;offset.phaseCount=phase;offset.pOutX=&x;offset.pOutY=&y;
    const auto offsetStatus=phaseStatus==0 && phase>0 ? query(nullptr,&offset.header) : UINT32_MAX;
    bool pass=phaseStatus==0 && phase==18 && offsetStatus==0 && std::isfinite(x) && std::isfinite(y) && std::abs(x)<=.5f && std::abs(y)<=.5f;
    std::ostringstream json;
    json << "{\n  \"status\":\"" << (pass?"QUERY_ABI_PASS":"QUERY_ABI_FAIL") << "\",\n"
         << "  \"vendor_id\":4098,\n  \"version_count_status\":" << versionStatus
         << ",\n  \"version_detail_status\":" << detailStatus << ",\n  \"providers\":[";
    if(detailStatus==0) for(size_t i=0;i<count && i<ids.size();++i) {
        if(i) json << ',';
        json << "{\"id\":" << ids[i] << ",\"name\":" << Quote(names[i]) << '}';
    }
    json << "],\n  \"jitter_phase_status\":" << phaseStatus << ",\n  \"jitter_phase\":" << phase
         << ",\n  \"jitter_offset_status\":" << offsetStatus << ",\n  \"jitter_offset\":[";
    if(std::isfinite(x))json << x;else json << "null";
    json << ',';if(std::isfinite(y))json << y;else json << "null";
    json << "],\n  \"dispatch_abi_verified\":false,\n  \"game_launched\":false,\n"
         << "  \"contexts_created\":0,\n  \"gpu_dispatches\":0,\n  \"game_runtime_ready\":false\n}\n";
    fs::create_directories(output.parent_path());std::ofstream stream(output,std::ios::binary);
    stream << json.str();stream.close();if(!stream)throw std::runtime_error("report write failed");
    FreeLibrary(backend); std::printf("%s",json.str().c_str());return pass?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;} }
