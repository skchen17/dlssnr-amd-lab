#pragma once
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ffx_resident {
constexpr size_t HeaderBytes=256,MaxBytes=size_t(3840)*2160*8,MappingBytes=HeaderBytes+2*MaxBytes;
struct Header {char magic[8];uint64_t request,response;uint32_t width,height,bytes,status,client,version;};
static_assert(sizeof(Header)==48&&offsetof(Header,status)==36);
class Exchange {
    HANDLE mapping_=nullptr,request_=nullptr,response_=nullptr;
    uint8_t* memory_=nullptr;uint64_t sequence_=0;bool poisoned_=false;
public:
    explicit Exchange(const std::filesystem::path& nameFile){
        std::ifstream f(nameFile);std::string name,extra;
        if(!(f>>name)||(f>>extra)||name.size()!=49||name.substr(0,17)!="Local\\DLSSNRLive_")throw std::invalid_argument("IPC name");
        for(size_t i=17;i<name.size();++i)if(!((name[i]>='0'&&name[i]<='9')||(name[i]>='a'&&name[i]<='f')))throw std::invalid_argument("IPC name characters");
        std::wstring wide(name.begin(),name.end());
        try{
            mapping_=OpenFileMappingW(FILE_MAP_ALL_ACCESS,FALSE,wide.c_str());
            if(!mapping_)throw std::runtime_error("IPC mapping unavailable");
            memory_=static_cast<uint8_t*>(MapViewOfFile(mapping_,FILE_MAP_ALL_ACCESS,0,0,MappingBytes));
            request_=OpenEventW(EVENT_MODIFY_STATE,FALSE,(wide+L"_request").c_str());
            response_=OpenEventW(SYNCHRONIZE|EVENT_MODIFY_STATE,FALSE,(wide+L"_response").c_str());
            if(!memory_||!request_||!response_)throw std::runtime_error("IPC views/events unavailable");
            Header h{};std::memcpy(&h,memory_,sizeof(h));
            if(std::memcmp(h.magic,"NRLIVE01",8)||h.request||h.response||h.client||h.version!=1||h.status!=1)throw std::invalid_argument("IPC worker not fresh/ready");
        }catch(...){Close();throw;}
    }
    Exchange(const Exchange&)=delete;
    ~Exchange(){Close();}
    void Close(){if(memory_){UnmapViewOfFile(memory_);memory_=nullptr;}for(auto* h:{&response_,&request_,&mapping_})if(*h){CloseHandle(*h);*h=nullptr;}}
    uint64_t Sequence()const{return sequence_;}
    bool Poisoned()const{return poisoned_;}
    std::vector<uint8_t> Run(const std::vector<uint8_t>& input,UINT width,UINT height,DWORD timeout=22000){
        if(poisoned_||!memory_||!width||!height||width>8192||height>8192||UINT64(width)*height*8!=input.size()||input.size()>MaxBytes||!timeout||timeout>22000)throw std::invalid_argument("IPC frame precondition");
        try{
            for(size_t i=0;i<input.size();i+=2){uint16_t h;std::memcpy(&h,input.data()+i,2);if((h&0x7c00)==0x7c00)throw std::invalid_argument("IPC nonfinite input");}
            Header h{};std::memcpy(&h,memory_,sizeof(h));
            if(std::memcmp(h.magic,"NRLIVE01",8)||h.version!=1||h.status!=1||h.request!=sequence_||h.response!=sequence_||(h.client&&h.client!=GetCurrentProcessId()))throw std::invalid_argument("IPC stale worker");
            std::memcpy(memory_+HeaderBytes,input.data(),input.size());
            h.request=sequence_+1;h.width=width;h.height=height;h.bytes=UINT(input.size());h.status=2;h.client=GetCurrentProcessId();
            std::memcpy(memory_,&h,sizeof(h));MemoryBarrier();
            if(!ResetEvent(response_)||!SetEvent(request_))throw std::runtime_error("IPC request event");
            if(WaitForSingleObject(response_,timeout)!=WAIT_OBJECT_0)throw std::runtime_error("IPC timeout; no output submitted");
            MemoryBarrier();Header result{};std::memcpy(&result,memory_,sizeof(result));
            h.response=h.request;h.status=1;
            if(std::memcmp(&h,&result,sizeof(h)))throw std::runtime_error("IPC response identity/status mismatch");
            std::vector<uint8_t> output(input.size());std::memcpy(output.data(),memory_+HeaderBytes+MaxBytes,output.size());
            for(size_t i=0;i<output.size();i+=2){uint16_t v;std::memcpy(&v,output.data()+i,2);if((v&0x7c00)==0x7c00)throw std::runtime_error("IPC nonfinite output");}
            sequence_=h.request;return output;
        }catch(...){poisoned_=true;throw;}
    }
};
}
