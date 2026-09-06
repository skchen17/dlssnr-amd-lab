#pragma once
#include <tlhelp32.h>
#include <charconv>
namespace native_preview {
inline std::string scalar(const std::string& s,const char* name){
    std::string key=std::string("\"")+name+"\"";auto p=s.find(key);
    if(p==s.npos||s.find(key,p+key.size())!=s.npos)throw std::runtime_error("missing/duplicate worker field");
    p=s.find_first_not_of(" \t\r",p+key.size());if(p==s.npos||s[p++]!=':')throw std::runtime_error("worker field syntax");
    p=s.find_first_not_of(" \t\r",p);if(p==s.npos)throw std::runtime_error("worker value missing");
    if(s[p]=='\"'){auto e=s.find('\"',p+1);if(e==s.npos)throw std::runtime_error("worker string missing end");return s.substr(p,e-p+1);}
    auto e=s.find_first_of(",} \t\r\n",p);return s.substr(p,e-p);
}
inline UINT64 number(const std::string& s,const char* name){auto t=scalar(s,name);UINT64 v=0;auto r=std::from_chars(t.data(),t.data()+t.size(),v);if(r.ec!=std::errc()||r.ptr!=t.data()+t.size())throw std::runtime_error("worker integer invalid");return v;}
struct Worker {
    HANDLE input=nullptr,output=nullptr,process=nullptr,actualProcess=nullptr;DWORD launcher=0,pid=0;UINT64 luid=0;std::filesystem::path dir;
    void send(const std::string& s){if(s.size()>8190)throw std::runtime_error("IPC too large");auto line=s+"\n";DWORD written=0;if(!WriteFile(input,line.data(),DWORD(line.size()),&written,nullptr)||written!=line.size())throw std::runtime_error("worker pipe write");}
    std::string read(const char* op){
        auto until=GetTickCount64()+90000;std::string s;
        while(GetTickCount64()<until){DWORD n=0;if(!PeekNamedPipe(output,nullptr,0,nullptr,&n,nullptr))throw std::runtime_error("worker pipe closed");
            if(n){char c;DWORD got;if(!ReadFile(output,&c,1,&got,nullptr)||got!=1)throw std::runtime_error("worker read");if(c=='\n'){
                    std::ofstream(dir/L"parent_messages.jsonl",std::ios::app)<<s<<'\n';if(scalar(s,"op")!=std::string("\"")+op+"\"")throw std::runtime_error("unexpected worker message");return s;}
                s+=c;if(s.size()>8192)throw std::runtime_error("oversized worker metadata");}
            else{if(WaitForSingleObject(process,0)==WAIT_OBJECT_0)throw std::runtime_error("worker exited");Sleep(5);}}
        throw std::runtime_error("worker timeout; retain all resources; no retry/kill");
    }
    void launch(const std::filesystem::path& repo,const std::filesystem::path& run,bool network,UINT maxFrames,UINT w,UINT h){
        dir=run;SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};HANDLE childIn=nullptr,childOut=nullptr;
        if(!CreatePipe(&childIn,&input,&sa,0)||!CreatePipe(&output,&childOut,&sa,0))throw std::runtime_error("create worker pipes");
        SetHandleInformation(input,HANDLE_FLAG_INHERIT,0);SetHandleInformation(output,HANDLE_FLAG_INHERIT,0);
        HANDLE err=CreateFileW((run/L"worker_stderr.log").c_str(),GENERIC_WRITE,FILE_SHARE_READ,&sa,CREATE_NEW,0,nullptr);if(err==INVALID_HANDLE_VALUE)throw std::runtime_error("stderr creation");
        auto quote=[](const std::filesystem::path& p){return L"\""+p.wstring()+L"\"";};
        auto python=repo/L".venv-rocm/Scripts/python.exe";
        std::wstring command=quote(python)+L" "+quote(repo/L"scripts/native_gpu_worker.py")+L" --producer "+std::to_wstring(GetCurrentProcessId())+L" --dll "+quote(run/L"native_frame_bridge.dll")+L" --output "+quote(run)+L" --max-frames "+std::to_wstring(maxFrames)+L" --max-width "+std::to_wstring(w)+L" --max-height "+std::to_wstring(h)+L" --max-padded-pixels "+std::to_wstring(((w+127)/128*128ULL)*((h+127)/128*128ULL));
        if(network)command+=L" --network";
        wchar_t audit[8]{};if(GetEnvironmentVariableW(L"NR_PREVIEW_AUDIT",audit,8)==1&&audit[0]=='1')command+=L" --diagnostic-readback";
        SIZE_T bytes=0;InitializeProcThreadAttributeList(nullptr,1,0,&bytes);std::vector<uint8_t> attrs(bytes);auto* list=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attrs.data());
        if(!InitializeProcThreadAttributeList(list,1,0,&bytes))throw std::runtime_error("attribute init");HANDLE allowed[]={childIn,childOut,err};
        if(!UpdateProcThreadAttribute(list,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,allowed,sizeof(allowed),nullptr,nullptr))throw std::runtime_error("handle allowlist");
        STARTUPINFOEXW si{};si.StartupInfo.cb=sizeof(si);si.StartupInfo.dwFlags=STARTF_USESTDHANDLES;si.StartupInfo.hStdInput=childIn;si.StartupInfo.hStdOutput=childOut;si.StartupInfo.hStdError=err;si.lpAttributeList=list;PROCESS_INFORMATION pi{};
        BOOL created=CreateProcessW(python.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|EXTENDED_STARTUPINFO_PRESENT,nullptr,repo.c_str(),&si.StartupInfo,&pi);
        DeleteProcThreadAttributeList(list);CloseHandle(childIn);CloseHandle(childOut);CloseHandle(err);if(!created)throw std::runtime_error("worker launch failed");
        process=pi.hProcess;launcher=pi.dwProcessId;CloseHandle(pi.hThread);
        auto hello=read("hello");auto id=number(hello,"pid");if(!id||id>MAXDWORD||number(hello,"protocol")!=1)throw std::runtime_error("worker identity");pid=DWORD(id);
        HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);if(snap==INVALID_HANDLE_VALUE)throw std::runtime_error("process snapshot");
        PROCESSENTRY32W e{};e.dwSize=sizeof(e);bool valid=false;for(BOOL ok=Process32FirstW(snap,&e);ok;ok=Process32NextW(snap,&e))if(e.th32ProcessID==pid){valid=pid==launcher||e.th32ParentProcessID==launcher;break;}CloseHandle(snap);
        actualProcess=OpenProcess(SYNCHRONIZE|PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);
        if(!valid||!actualProcess||WaitForSingleObject(process,0)!=WAIT_TIMEOUT||WaitForSingleObject(actualProcess,0)!=WAIT_TIMEOUT)throw std::runtime_error("unverified worker child identity");
        luid=number(hello,"adapter_luid");if(!luid)throw std::runtime_error("worker LUID missing");
    }
    void prepare(Bridge& b){UINT64 d[16]{};if(nr_bridge_export(&b,pid,1,d))throw std::runtime_error(nr_bridge_error());std::ostringstream s;s<<"{\"op\":\"prepare\",\"descriptor\":[";for(int i=0;i<16;++i){if(i)s<<',';s<<d[i];}s<<"]}";send(s.str());auto m=read("ready");if(number(m,"pid")!=pid||number(m,"generation")!=1)throw std::runtime_error("ready identity");}
    std::string frame(Bridge& b){send("{\"op\":\"frame\",\"generation\":1,\"sequence\":"+std::to_string(b.seq)+",\"seed\":0}");auto m=read("done");if(number(m,"pid")!=pid||number(m,"sequence")!=b.seq||number(m,"generation")!=1||scalar(m,"finite")!="true"||scalar(m,"cpu_neural_fallback")!="false")throw std::runtime_error("wrong output identity");if(nr_bridge_remote_done(&b,b.seq))throw std::runtime_error(nr_bridge_error());return m;}
    void consumed(Bridge& b){send("{\"op\":\"consumed\",\"generation\":1,\"sequence\":"+std::to_string(b.seq)+"}");auto m=read("released");if(number(m,"sequence")!=b.seq||number(m,"generation")!=1)throw std::runtime_error("consumer identity");}
    void close(){send("{\"op\":\"close\"}");auto m=read("closed");if(number(m,"pid")!=pid||number(m,"allocated_after_release_bytes")||number(m,"reserved_after_release_bytes"))throw std::runtime_error("worker release failed");
        if(WaitForSingleObject(actualProcess,15000)!=WAIT_OBJECT_0||WaitForSingleObject(process,15000)!=WAIT_OBJECT_0)throw std::runtime_error("normal worker exit missing");DWORD code=1;GetExitCodeProcess(actualProcess,&code);if(code)throw std::runtime_error("worker abnormal exit");for(HANDLE h:{input,output,process,actualProcess})CloseHandle(h);input=output=process=actualProcess=nullptr;}
};
}
