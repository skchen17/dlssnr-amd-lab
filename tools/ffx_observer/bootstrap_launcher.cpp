#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include "ffx_observer.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace fs=std::filesystem;
void Require(bool ok,const char* what){if(!ok)throw std::runtime_error(std::string(what)+" (win32="+std::to_string(GetLastError())+")");}
void Read(HANDLE p,const void* a,void* b,SIZE_T n){SIZE_T got=0;Require(ReadProcessMemory(p,a,b,n,&got)&&got==n,"ReadProcessMemory");}
void Write(HANDLE p,void* a,const void* b,SIZE_T n){SIZE_T got=0;Require(WriteProcessMemory(p,a,b,n,&got)&&got==n,"WriteProcessMemory");}
void Instruction(HANDLE p,void* address,BYTE byte){
    DWORD previous=0,ignored=0;Require(VirtualProtectEx(p,address,1,PAGE_EXECUTE_READWRITE,&previous),"protect entry");
    SIZE_T written=0;BOOL ok=WriteProcessMemory(p,address,&byte,1,&written);
    BOOL restored=VirtualProtectEx(p,address,1,previous,&ignored);BOOL flushed=FlushInstructionCache(p,address,1);
    Require(ok&&written==1&&restored&&flushed,"restore entry protection/cache");
}
uintptr_t Module(DWORD pid,const fs::path& path) {
    HANDLE snapshot=INVALID_HANDLE_VALUE;
    for(int i=0;i<4&&snapshot==INVALID_HANDLE_VALUE;++i)snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    Require(snapshot!=INVALID_HANDLE_VALUE,"module snapshot");
    MODULEENTRY32W m{};m.dwSize=sizeof(m);uintptr_t base=0;
    for(BOOL ok=Module32FirstW(snapshot,&m);ok;ok=Module32NextW(snapshot,&m))
        if(_wcsicmp(fs::absolute(m.szExePath).c_str(),path.c_str())==0){base=reinterpret_cast<uintptr_t>(m.modBaseAddr);break;}
    CloseHandle(snapshot);Require(base!=0,"exact remote module not found");return base;
}
uintptr_t RemoteSystemFunction(DWORD pid,const char* name) {
    auto function=GetProcAddress(GetModuleHandleW(L"kernel32.dll"),name);Require(function!=nullptr,"system export");
    HMODULE owner=nullptr;Require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(function),&owner),"system export owner");
    wchar_t path[32768]{};Require(GetModuleFileNameW(owner,path,32768)>0,"system module path");
    return Module(pid,fs::absolute(path))+reinterpret_cast<uintptr_t>(function)-reinterpret_cast<uintptr_t>(owner);
}
DWORD RemoteCall(HANDLE p,uintptr_t entry,const void* data,SIZE_T size) {
    void* memory=VirtualAllocEx(p,nullptr,size,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);Require(memory!=nullptr,"remote argument allocation");
    Write(p,memory,data,size);
    HANDLE thread=CreateRemoteThread(p,nullptr,0,reinterpret_cast<LPTHREAD_START_ROUTINE>(entry),memory,0,nullptr);
    Require(thread!=nullptr,"remote call thread");
    DWORD wait=WaitForSingleObject(thread,15000),code=0;
    // Never free arguments while a remote thread may still use them. On any
    // uncertainty the outer guard terminates only this newly created child.
    if(wait!=WAIT_OBJECT_0){CloseHandle(thread);throw std::runtime_error("remote call timeout");}
    BOOL ok=GetExitCodeThread(thread,&code);CloseHandle(thread);VirtualFreeEx(p,memory,0,MEM_RELEASE);
    Require(ok,"remote call result");return code;
}
int wmain(int argc,wchar_t** argv) {
    if(argc!=4 && !(argc==5&&(std::wstring(argv[4])==L"wait-test"||std::wstring(argv[4])==L"decode-upscale"))){
        std::fprintf(stderr,"usage: ffx_bootstrap_launcher <GoWR.exe|ffx_bootstrap_host.exe> <observer.dll> <new result dir> [wait-test|decode-upscale]\n");return 2;
    }
    const bool waitTest=argc==5&&std::wstring(argv[4])==L"wait-test";
    const bool decode=argc==5&&std::wstring(argv[4])==L"decode-upscale";
    PROCESS_INFORMATION pi{};bool debugging=false,heldEvent=false,success=false,entryRestored=false,resultCreated=false;
    DEBUG_EVENT event{};std::ofstream stages;fs::path result;
    auto log=[&](const char* text){if(stages)stages<<text<<std::endl;std::printf("%s\n",text);std::fflush(stdout);};
    try {
        fs::path exe=fs::canonical(argv[1]),dll=fs::canonical(argv[2]);result=fs::absolute(argv[3]);
        const bool fixture=exe.filename()==L"ffx_bootstrap_host.exe";
        Require(fixture||exe.filename()==L"GoWR.exe","target must be explicit game or bootstrap fixture");
        Require(!waitTest||fixture,"wait-test only for fixture");Require(!decode||!fixture,"decode-upscale only for audited game");
        Require(dll.filename()==L"ffx_observer.dll"||(!fixture&&dll.filename()==L"resident_present.dll"),"observer name must match lab DLL");
        Require(!fs::exists(result)&&fs::create_directories(result),"new result directory");resultCreated=true;stages.open(result/L"stages.log");
        auto logPath=result/L"events.jsonl";Require(logPath.native().size()<1024,"log path length");
        // Validate x64 PE on disk before creating a process. No image-file writes.
        std::ifstream image(exe,std::ios::binary);IMAGE_DOS_HEADER dos{};image.read(reinterpret_cast<char*>(&dos),sizeof(dos));
        Require(image&&dos.e_magic==IMAGE_DOS_SIGNATURE&&dos.e_lfanew>0,"PE DOS header");
        image.seekg(dos.e_lfanew);IMAGE_NT_HEADERS64 nt{};image.read(reinterpret_cast<char*>(&nt),sizeof(nt));
        Require(image&&nt.Signature==IMAGE_NT_SIGNATURE&&nt.FileHeader.Machine==IMAGE_FILE_MACHINE_AMD64&&nt.OptionalHeader.Magic==IMAGE_NT_OPTIONAL_HDR64_MAGIC,"x64 target required");image.close();
        std::wstring command=L"\""+exe.native()+L"\"";
        if(fixture)command+=L" \""+(result/L"host.json").native()+L"\"";
        STARTUPINFOW startup{};startup.cb=sizeof(startup);
        DWORD flags=DEBUG_ONLY_THIS_PROCESS|(fixture?CREATE_NO_WINDOW:0);
        log("creating new debug child");
        Require(CreateProcessW(exe.c_str(),command.data(),nullptr,nullptr,FALSE,flags,nullptr,exe.parent_path().c_str(),&startup,&pi),"CreateProcess");debugging=true;
        void* entry=nullptr;BYTE original=0;bool armed=false,reached=false;
        const auto deadline=GetTickCount64()+30000;
        while(GetTickCount64()<deadline&&!reached) {
            if(!WaitForDebugEvent(&event,100)){Require(GetLastError()==ERROR_SEM_TIMEOUT,"WaitForDebugEvent");continue;}
            heldEvent=true;DWORD continuation=DBG_CONTINUE;
            if(event.dwDebugEventCode==CREATE_PROCESS_DEBUG_EVENT) {
                if(event.u.CreateProcessInfo.hFile)CloseHandle(event.u.CreateProcessInfo.hFile);
                entry=static_cast<BYTE*>(event.u.CreateProcessInfo.lpBaseOfImage)+nt.OptionalHeader.AddressOfEntryPoint;
                Read(pi.hProcess,entry,&original,1);Require(original!=0xCC,"entry already breakpointed");
            } else if(event.dwDebugEventCode==LOAD_DLL_DEBUG_EVENT) {
                if(event.u.LoadDll.hFile)CloseHandle(event.u.LoadDll.hFile);
            } else if(event.dwDebugEventCode==EXCEPTION_DEBUG_EVENT) {
                auto& exception=event.u.Exception.ExceptionRecord;
                if(exception.ExceptionCode==EXCEPTION_BREAKPOINT && !armed) {
                    Require(entry!=nullptr,"missing entry address");Instruction(pi.hProcess,entry,0xCC);armed=true;log("loader breakpoint; entry breakpoint armed");
                } else if(exception.ExceptionCode==EXCEPTION_BREAKPOINT&&exception.ExceptionAddress==entry) {
                    Require(event.dwThreadId==pi.dwThreadId,"entry not on primary thread");
                    Instruction(pi.hProcess,entry,original);entryRestored=true;
                    CONTEXT context{};context.ContextFlags=CONTEXT_CONTROL;Require(GetThreadContext(pi.hThread,&context),"entry context");
                    Require(context.Rip==reinterpret_cast<uintptr_t>(entry)+1,"entry RIP mismatch");context.Rip=reinterpret_cast<uintptr_t>(entry);
                    Require(SetThreadContext(pi.hThread,&context),"restore entry RIP");Require(SuspendThread(pi.hThread)!=DWORD(-1),"hold primary thread");
                    reached=true;log("entry restored; primary thread held");
                } else continuation=DBG_EXCEPTION_NOT_HANDLED;
            } else if(event.dwDebugEventCode==EXIT_PROCESS_DEBUG_EVENT) {
                ContinueDebugEvent(event.dwProcessId,event.dwThreadId,DBG_CONTINUE);heldEvent=false;debugging=false;
                throw std::runtime_error("child exited before entry gate (exit="+std::to_string(event.u.ExitProcess.dwExitCode)+")");
            }
            Require(ContinueDebugEvent(event.dwProcessId,event.dwThreadId,continuation),"ContinueDebugEvent");heldEvent=false;
        }
        Require(reached,"entry breakpoint deadline");
        Require(DebugActiveProcessStop(pi.dwProcessId),"detach debugger");debugging=false;
        // No executable remote allocation or custom machine-code stub. Resolve
        // system functions from their actual owning module, including forwarders.
        const auto load=RemoteSystemFunction(pi.dwProcessId,"LoadLibraryW");
        log("loading observer with primary thread held");
        RemoteCall(pi.hProcess,load,dll.c_str(),(dll.native().size()+1)*sizeof(wchar_t));
        const auto remoteBase=Module(pi.dwProcessId,dll); // Never truncate x64 HMODULE to thread exit DWORD.
        HMODULE local=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);Require(local!=nullptr,"local observer export resolution");
        auto start=GetProcAddress(local,"FfxObserver_Bootstrap");Require(start!=nullptr,"bootstrap export");
        const auto rva=reinterpret_cast<uintptr_t>(start)-reinterpret_cast<uintptr_t>(local);FreeLibrary(local);
        FfxObserverBootstrapV2 args{};args.size=sizeof(args);args.version=2;args.profile=decode?1:0;args.event_limit=decode?10000:256;wcscpy_s(args.log_path,logPath.c_str());
        Require(RemoteCall(pi.hProcess,remoteBase+rva,&args,sizeof(args))==1,"observer bootstrap rejected");
        log(decode?"observer attached metadata decode; resuming primary thread":"observer attached header-only; resuming primary thread");Require(ResumeThread(pi.hThread)==1,"resume exactly one primary suspension");
        DWORD childExit=STILL_ACTIVE;
        if(waitTest){Require(WaitForSingleObject(pi.hProcess,15000)==WAIT_OBJECT_0,"fixture completion timeout");Require(GetExitCodeProcess(pi.hProcess,&childExit)&&childExit==0,"fixture exit status");}
        std::ofstream manifest(result/L"bootstrap.json");manifest<<"{\"status\":\"BOOTSTRAP_PASS\",\"pid\":"<<pi.dwProcessId
            <<",\"entry_instruction_restored\":true,\"observer_profile\":"<<args.profile<<",\"event_limit\":"<<args.event_limit<<",\"child_exit_code\":"<<childExit
            <<",\"game_launched\":"<<(fixture?"false":"true")<<",\"texture_capture_enabled\":false,\"game_runtime_ready\":false}\n";
        manifest.close();Require(bool(manifest),"bootstrap report");success=true;log("BOOTSTRAP_PASS");
    }catch(const std::exception& e) {
        log(e.what());
        if(pi.hProcess) {
            // This handle comes only from this launcher's CreateProcess. Never
            // enumerate/terminate an already-running user's game or child tree.
            const BOOL stopped=TerminateProcess(pi.hProcess,91);
            if(heldEvent)ContinueDebugEvent(event.dwProcessId,event.dwThreadId,DBG_CONTINUE);
            if(debugging)DebugActiveProcessStop(pi.dwProcessId);
            DWORD waited=WaitForSingleObject(pi.hProcess,5000);
            log(stopped&&waited==WAIT_OBJECT_0?"failed bootstrap child terminated":"child already exited or cleanup uncertain");
        }
        if(resultCreated)std::ofstream(result/L"bootstrap.json")<<"{\"status\":\"BOOTSTRAP_FAIL\",\"pid\":"<<pi.dwProcessId<<",\"entry_instruction_restored\":"<<(entryRestored?"true":"false")<<",\"game_runtime_ready\":false}\n";
    }
    if(pi.hThread)CloseHandle(pi.hThread);if(pi.hProcess)CloseHandle(pi.hProcess);return success?0:1;
}
