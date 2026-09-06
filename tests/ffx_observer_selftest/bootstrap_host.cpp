#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FFX_API_ENTRY
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include <fstream>
#include <filesystem>
extern "C" __declspec(dllimport) uint64_t Fixture_Count(unsigned);
int wmain(int argc,wchar_t** argv) {
    if(argc!=2)return 2;
    ffxContext c=nullptr;ffxCreateContextDescUpscale create{};create.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    const auto s0=ffxCreateContext(&c,&create.header,nullptr);
    ffxApiHeader q{0x5555,nullptr};const auto s1=ffxQuery(&c,&q);
    SetLastError(0xCAFE);const auto s2=ffxConfigure(&c,&q);
    ffxDispatchDescUpscale d{};d.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    const auto s3=ffxDispatch(&c,&d.header);const auto error=GetLastError();
    const auto s4=ffxDestroyContext(&c,nullptr);
    bool pass=s0==0&&s1==31&&s2==23&&s3==37&&s4==29&&error==0xFACE&&!c;
    for(unsigned i=0;i<5;++i)pass&=Fixture_Count(i)==1;
    auto log=std::filesystem::path(argv[1]).parent_path()/L"events.jsonl";
    HANDLE reader=CreateFileW(log.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,0,nullptr);
    char first=0;DWORD bytes=0;bool liveRead=reader!=INVALID_HANDLE_VALUE&&ReadFile(reader,&first,1,&bytes,nullptr)&&bytes==1&&first=='{';
    if(reader!=INVALID_HANDLE_VALUE)CloseHandle(reader);pass&=liveRead;
    std::ofstream(std::filesystem::path(argv[1]))<<"{\"status\":\""<<(pass?"HOST_PASS":"HOST_FAIL")<<"\",\"original_calls\":5,\"last_error\":"<<error<<",\"concurrent_log_read\":"<<(liveRead?"true":"false")<<"}\n";
    return pass?0:1;
}
