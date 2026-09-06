#define wmain unused_live_attach_main
#include "live_attach.cpp"
#undef wmain
int wmain(int argc,wchar_t** argv) {
    if(argc!=5){std::fprintf(stderr,"usage: ffx_session_control <PID> <exact GoWR.exe> <loaded session.dll> <existing action|preview-network|preview-stop|preview-status|preview-input>\n");return 2;}
    HANDLE process=nullptr;
    try {
        wchar_t* end=nullptr;DWORD pid=wcstoul(argv[1],&end,10);Require(pid&&end&&!*end,"PID");
        auto exe=fs::canonical(argv[2]),dll=fs::canonical(argv[3]);std::wstring action=argv[4];
        const wchar_t* actions[]={L"arm",L"poll",L"observe",L"stop",L"output-arm",L"output-poll",L"roundtrip-arm",L"roundtrip-poll",L"patch-arm",L"patch-poll",L"filter-arm",L"filter-poll",L"network-arm",L"network-poll",L"preview-network",L"preview-stop",L"preview-status",L"preview-input",L"resident-start",L"resident-stop",L"resident-status"};
        uint32_t actionCode=0;for(uint32_t i=0;i<21;++i)if(action==actions[i])actionCode=i+1;
        Require(exe.filename()==L"GoWR.exe"&&actionCode&&
            (dll.filename()==L"ffx_capture_session.dll"||(dll.filename()==L"resident_present.dll"&&actionCode>=19)),"restricted operation");
        process=OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_VM_READ|PROCESS_VM_WRITE|PROCESS_VM_OPERATION|PROCESS_CREATE_THREAD|SYNCHRONIZE,FALSE,pid);Require(process!=nullptr,"open explicit PID");
        wchar_t path[32768]{};DWORD n=32768;Require(QueryFullProcessImageNameW(process,0,path,&n)&&!_wcsicmp(path,exe.c_str()),"exact live image");
        auto remote=Module(pid,dll);Require(remote!=0,"session not loaded");
        auto local=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);Require(local!=nullptr,"local exports");
        auto command=GetProcAddress(local,"FfxSession_Command");Require(command!=nullptr,"command export");auto rva=reinterpret_cast<uintptr_t>(command)-reinterpret_cast<uintptr_t>(local);FreeLibrary(local);
        FfxSessionCommandV1 c{sizeof(c),actionCode};auto code=RemoteCall(process,remote+rva,&c,sizeof(c));
        std::printf("SESSION_COMMAND_RESULT %lu (1=accepted/complete, 2=not complete, 3=failed, 0=rejected)\n",code);
        CloseHandle(process);return code==1||code==2?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s; game not stopped\n",e.what());if(process)CloseHandle(process);return 1;}
}
