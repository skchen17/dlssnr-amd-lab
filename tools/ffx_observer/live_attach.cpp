// Attach only to an explicitly identified, already lab-observed GoWR process.
// Never launch, suspend, terminate, or detach hooks from the user's game.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "live_inspector.h"
#include "capture_session.h"
#include <tlhelp32.h>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <cstdio>
namespace fs=std::filesystem;
void Require(bool ok,const char* why){if(!ok)throw std::runtime_error(std::string(why)+" win32="+std::to_string(GetLastError()));}
uintptr_t Module(DWORD pid,const fs::path& path) {
    HANDLE h=INVALID_HANDLE_VALUE;
    for(int i=0;i<4&&h==INVALID_HANDLE_VALUE;++i)h=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    Require(h!=INVALID_HANDLE_VALUE,"module snapshot");
    MODULEENTRY32W m{};m.dwSize=sizeof(m);uintptr_t base=0;
    for(BOOL ok=Module32FirstW(h,&m);ok;ok=Module32NextW(h,&m))
        if(!_wcsicmp(fs::absolute(m.szExePath).c_str(),path.c_str())){base=reinterpret_cast<uintptr_t>(m.modBaseAddr);break;}
    CloseHandle(h);return base;
}
DWORD RemoteCall(HANDLE process,uintptr_t entry,const void* data,SIZE_T size) {
    auto* memory=VirtualAllocEx(process,nullptr,size,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
    Require(memory!=nullptr,"argument allocation");SIZE_T written=0;
    if(!WriteProcessMemory(process,memory,data,size,&written)||written!=size){VirtualFreeEx(process,memory,0,MEM_RELEASE);Require(false,"argument write");}
    HANDLE thread=CreateRemoteThread(process,nullptr,0,reinterpret_cast<LPTHREAD_START_ROUTINE>(entry),memory,0,nullptr);
    if(!thread){VirtualFreeEx(process,memory,0,MEM_RELEASE);Require(false,"remote call");}
    DWORD wait=WaitForSingleObject(thread,15000),code=0;
    // On uncertainty retain arguments; do NOT stop the game or retry the call.
    if(wait!=WAIT_OBJECT_0){CloseHandle(thread);throw std::runtime_error("remote call uncertain; game left untouched; do not retry");}
    BOOL ok=GetExitCodeThread(thread,&code);CloseHandle(thread);VirtualFreeEx(process,memory,0,MEM_RELEASE);
    Require(ok,"remote exit status");return code;
}
int wmain(int argc,wchar_t** argv) {
    if(argc!=6&&argc!=7){std::fprintf(stderr,"usage: ffx_live_attach <PID> <exact GoWR.exe> <loaded observer.dll> <new inspector.dll> <fresh log path> [deferred]\n");return 2;}
    HANDLE process=nullptr;
    try {
        wchar_t* end=nullptr;unsigned long number=wcstoul(argv[1],&end,10);Require(number&&end&&!*end,"PID");DWORD pid=number;
        auto exe=fs::canonical(argv[2]),observer=fs::canonical(argv[3]),dll=fs::canonical(argv[4]),log=fs::absolute(argv[5]);
        bool contract=dll.filename()==L"ffx_contract_inspector.dll"&&observer.filename()==L"ffx_live_inspector.dll";
        bool session=dll.filename()==L"ffx_capture_session.dll"&&observer.filename()==L"ffx_contract_inspector.dll";
        bool deferred=argc==7;Require(!deferred||(session&&std::wstring(argv[6])==L"deferred"),"deferred session only");
        Require(exe.filename()==L"GoWR.exe"&&(session||contract||(observer.filename()==L"ffx_observer.dll"&&dll.filename()==L"ffx_live_inspector.dll")),"restricted target names");
        Require(!fs::exists(log)&&fs::is_directory(log.parent_path())&&log.native().size()<1024,"fresh absolute log path");
        process=OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_VM_READ|PROCESS_VM_WRITE|PROCESS_VM_OPERATION|PROCESS_CREATE_THREAD|SYNCHRONIZE,FALSE,pid);
        Require(process!=nullptr,"open explicit PID");
        wchar_t actual[32768]{};DWORD size=32768;
        Require(QueryFullProcessImageNameW(process,0,actual,&size)&&!_wcsicmp(actual,exe.c_str()),"exact live image path");
        USHORT machine=0,native=0;Require(IsWow64Process2(process,&machine,&native)&&machine==IMAGE_FILE_MACHINE_UNKNOWN&&native==IMAGE_FILE_MACHINE_AMD64,"native x64 process");
        Require(WaitForSingleObject(process,0)==WAIT_TIMEOUT,"process not running");
        auto oldBase=Module(pid,observer);Require(oldBase!=0,"existing lab observer required");
        Require(!Module(pid,dll),"inspector already loaded; refusing duplicate or uncertain retry");
        HMODULE local=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        Require(local!=nullptr,"local inspector export resolution");auto start=GetProcAddress(local,session?"FfxSession_Attach":"FfxLive_Attach");
        Require(start!=nullptr,"attach export");auto rva=reinterpret_cast<uintptr_t>(start)-reinterpret_cast<uintptr_t>(local);FreeLibrary(local);
        auto loader=GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"LoadLibraryW");HMODULE owner=nullptr;
        Require(loader&&GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(loader),&owner),"loader owner");
        wchar_t ownerPath[32768]{};Require(GetModuleFileNameW(owner,ownerPath,32768)>0,"loader path");
        auto remoteOwner=Module(pid,fs::absolute(ownerPath));Require(remoteOwner!=0,"remote loader owner");
        auto remoteLoader=remoteOwner+reinterpret_cast<uintptr_t>(loader)-reinterpret_cast<uintptr_t>(owner);
        RemoteCall(process,remoteLoader,dll.c_str(),(dll.native().size()+1)*sizeof(wchar_t));
        auto remoteBase=Module(pid,dll);Require(remoteBase!=0,"loaded inspector base");
        if(session) {
            FfxSessionConfigV1 config{};config.size=sizeof(config);config.version=1;config.sample_limit=64;config.capture_enabled=0;
            if(deferred)config.version=2;
            config.expected_dispatch_module=oldBase;config.expected_context_module=Module(pid,observer.parent_path()/L"ffx_observer.dll");Require(config.expected_context_module!=0,"context observer module");
            wcscpy_s(config.log_path,log.c_str());Require(RemoteCall(process,remoteBase+rva,&config,sizeof(config))==1,"session attach rejected/uncertain");
        }else {
            FfxLiveConfigV1 config{};config.size=sizeof(config);config.version=1;config.sample_limit=16;config.expected_observer_base=oldBase;
            wcscpy_s(config.log_path,log.c_str());Require(RemoteCall(process,remoteBase+rva,&config,sizeof(config))==1,"attach rejected or uncertain; do not retry");
        }
        std::printf("LIVE_ATTACH_PASS pid=%lu; observation only until explicit session arm; no game file deployment\n",pid);
        CloseHandle(process);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s; game NOT stopped or restarted\n",e.what());if(process)CloseHandle(process);return 1;}
}
