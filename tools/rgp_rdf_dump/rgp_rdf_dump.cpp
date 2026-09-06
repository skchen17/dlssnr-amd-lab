// Enumerate the public AMD RDF container around an RGP capture.
// This does not decode proprietary RGP payloads or invent missing counters.
#include <amdrdf.h>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static std::string JsonEscape(const char* input) {
    std::string out;
    for (const unsigned char c : std::string(input)) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(char(c)); }
        else if (c >= 0x20) out.push_back(char(c));
    }
    return out;
}

static void PrintStrings(const std::vector<unsigned char>& bytes) {
    bool first=true;std::string current;std::cout << "[";
    auto flush=[&]() { if (current.size()>=3) { if (!first) std::cout << ", ";first=false;
        std::cout << "\"" << JsonEscape(current.c_str()) << "\""; } current.clear(); };
    for (const unsigned char c : bytes) { if (c>=0x20 && c<=0x7e) current.push_back(char(c)); else flush(); }
    flush();std::cout << "]";
}

template<typename T> static T Read(const unsigned char* bytes) {
    T value{};std::memcpy(&value,bytes,sizeof(value));return value;
}

struct DerivedCounter {
    std::string name;
    std::vector<double> values;
    std::uint32_t unit=0;
};

struct AsicSummary {
    std::uint64_t shader_clock_hz=0;
    std::uint64_t memory_clock_hz=0;
    std::uint64_t timestamp_frequency_hz=0;
    std::int32_t vgprs_per_simd=0;
    std::int32_t sgprs_per_simd=0;
    std::int32_t shader_engines=0;
    std::int32_t cus_per_shader_engine=0;
    std::int32_t simds_per_cu=0;
    std::int32_t waves_per_simd=0;
    std::int32_t vgpr_alloc_granularity=0;
    std::int32_t l2_bytes=0;
    std::int32_t l0_bytes_per_cu=0;
    std::int32_t lds_bytes_per_cu=0;
    std::string name;
};

static AsicSummary ReadAsicSummary(rdf::ChunkFile& file) {
    if (!file.ContainsChunk("AsicInfo")) throw std::runtime_error("capture has no ASIC information");
    std::vector<unsigned char> data(size_t(file.GetChunkDataSize("AsicInfo",0)));
    file.ReadChunkDataToBuffer("AsicInfo",0,data.data());
    // Public PAL AsicInfo v3. The uint64 fields are naturally aligned after pciId.
    if (data.size()<392) throw std::runtime_error("unsupported ASIC information payload");
    AsicSummary asic;
    asic.shader_clock_hz=Read<std::uint64_t>(&data[8]);
    asic.memory_clock_hz=Read<std::uint64_t>(&data[16]);
    asic.timestamp_frequency_hz=Read<std::uint64_t>(&data[24]);
    asic.vgprs_per_simd=Read<std::int32_t>(&data[56]);
    asic.sgprs_per_simd=Read<std::int32_t>(&data[60]);
    asic.shader_engines=Read<std::int32_t>(&data[64]);
    asic.cus_per_shader_engine=Read<std::int32_t>(&data[68]);
    asic.simds_per_cu=Read<std::int32_t>(&data[72]);
    asic.waves_per_simd=Read<std::int32_t>(&data[76]);
    asic.vgpr_alloc_granularity=Read<std::int32_t>(&data[84]);
    asic.l2_bytes=Read<std::int32_t>(&data[148]);
    asic.l0_bytes_per_cu=Read<std::int32_t>(&data[152]);
    asic.lds_bytes_per_cu=Read<std::int32_t>(&data[156]);
    const auto* name=reinterpret_cast<const char*>(&data[160]);
    asic.name.assign(name,std::find(name,name+256,'\0'));
    if (!asic.timestamp_frequency_hz || !asic.shader_clock_hz) throw std::runtime_error("invalid ASIC clock frequencies");
    return asic;
}

static int PrintSpmSummary(rdf::ChunkFile& file) {
    if (!file.ContainsChunk("SpmSession") || !file.ContainsChunk("DerivedSpmCtr"))
        throw std::runtime_error("capture has no derived SPM session");
    std::vector<unsigned char> session_header(size_t(file.GetChunkHeaderSize("SpmSession",0)));
    std::vector<unsigned char> timestamps(size_t(file.GetChunkDataSize("SpmSession",0)));
    file.ReadChunkHeaderToBuffer("SpmSession",0,session_header.data());
    file.ReadChunkDataToBuffer("SpmSession",0,timestamps.data());
    if (session_header.size()<20) throw std::runtime_error("unsupported SPM session header");
    const auto interval=Read<std::uint32_t>(&session_header[8]);
    const auto samples=Read<std::uint32_t>(&session_header[12]);
    if (!samples || timestamps.size()<size_t(samples)*8) throw std::runtime_error("invalid SPM sample count");
    const auto asic=ReadAsicSummary(file);
    std::vector<std::uint64_t> timestamp_values(samples);
    std::memcpy(timestamp_values.data(),timestamps.data(),size_t(samples)*sizeof(std::uint64_t));
    std::vector<DerivedCounter> counters;
    const auto count=file.GetChunkCount("DerivedSpmCtr");
    for (int index=0;index<count;++index) {
        std::vector<unsigned char> header(size_t(file.GetChunkHeaderSize("DerivedSpmCtr",index)));
        std::vector<unsigned char> data(size_t(file.GetChunkDataSize("DerivedSpmCtr",index)));
        file.ReadChunkHeaderToBuffer("DerivedSpmCtr",index,header.data());
        file.ReadChunkDataToBuffer("DerivedSpmCtr",index,data.data());
        if (header.size()<24) throw std::runtime_error("unsupported derived counter header");
        const auto components=Read<std::uint32_t>(&header[8]);
        const auto name_size=Read<std::uint32_t>(&header[16]);
        const auto description_size=Read<std::uint32_t>(&header[20]);
        const size_t value_bytes=size_t(samples)*sizeof(double),name_offset=value_bytes+size_t(components)*4;
        if (name_offset+name_size+description_size!=data.size()) throw std::runtime_error("derived counter payload size mismatch");
        DerivedCounter counter;counter.unit=Read<std::uint32_t>(&header[12]);
        counter.name.assign(reinterpret_cast<const char*>(data.data()+name_offset),name_size);
        counter.values.resize(samples);std::memcpy(counter.values.data(),data.data(),value_bytes);
        counters.push_back(std::move(counter));
    }
    const auto busy=std::find_if(counters.begin(),counters.end(),[](const auto& item){return item.name=="Gpu Busy Cycles";});
    if (busy==counters.end()) throw std::runtime_error("Gpu Busy Cycles counter missing");
    std::vector<size_t> active;for(size_t i=0;i<samples;++i)if(std::isfinite(busy->values[i])&&busy->values[i]>0)active.push_back(i);
    if (active.empty()) throw std::runtime_error("SPM capture has no active GPU samples");
    std::uint64_t active_ticks=0,gap_ticks=0;size_t active_segments=1;
    for(size_t i=1;i<samples;++i) {
        if(timestamp_values[i]<timestamp_values[i-1]) throw std::runtime_error("non-monotonic SPM timestamps");
        const auto delta=timestamp_values[i]-timestamp_values[i-1];
        const bool previous_active=std::isfinite(busy->values[i-1])&&busy->values[i-1]>0;
        const bool current_active=std::isfinite(busy->values[i])&&busy->values[i]>0;
        if(previous_active) active_ticks+=delta;
        if(!previous_active&&current_active) ++active_segments;
        if(!previous_active&&!current_active) gap_ticks+=delta;
    }
    const auto first_timestamp=Read<std::uint64_t>(timestamps.data());
    const auto last_timestamp=Read<std::uint64_t>(timestamps.data()+size_t(samples-1)*8);
    std::cout << "{\n  \"schema\": 1, \"sample_interval\": " << interval << ", \"sample_count\": " << samples
              << ", \"active_sample_count\": " << active.size() << ", \"first_timestamp\": " << first_timestamp
              << ", \"last_timestamp\": " << last_timestamp
              << ", \"active_segments\": " << active_segments
              << ", \"sampled_active_time_ms\": " << std::setprecision(17)
              << (double(active_ticks)*1000.0/double(asic.timestamp_frequency_hz))
              << ", \"sampled_inactive_gap_ms\": " << (double(gap_ticks)*1000.0/double(asic.timestamp_frequency_hz))
              << ",\n  \"asic\": {\"name\": \"" << JsonEscape(asic.name.c_str())
              << "\", \"shader_clock_hz\": " << asic.shader_clock_hz
              << ", \"memory_clock_hz\": " << asic.memory_clock_hz
              << ", \"timestamp_frequency_hz\": " << asic.timestamp_frequency_hz
              << ", \"vgprs_per_simd\": " << asic.vgprs_per_simd
              << ", \"sgprs_per_simd\": " << asic.sgprs_per_simd
              << ", \"shader_engines\": " << asic.shader_engines
              << ", \"cus_per_shader_engine\": " << asic.cus_per_shader_engine
              << ", \"simds_per_cu\": " << asic.simds_per_cu
              << ", \"waves_per_simd\": " << asic.waves_per_simd
              << ", \"vgpr_alloc_granularity\": " << asic.vgpr_alloc_granularity
              << ", \"l2_bytes\": " << asic.l2_bytes
              << ", \"l0_bytes_per_cu\": " << asic.l0_bytes_per_cu
              << ", \"lds_bytes_per_cu\": " << asic.lds_bytes_per_cu << "},\n  \"counters\": [\n";
    for(size_t c=0;c<counters.size();++c) {
        const auto& counter=counters[c];double sum=0,minimum=std::numeric_limits<double>::infinity(),maximum=-minimum;size_t finite=0;
        for(const size_t i:active) {const double value=counter.values[i];if(!std::isfinite(value))continue;
            sum+=value;minimum=std::min(minimum,value);maximum=std::max(maximum,value);++finite;}
        if(c)std::cout << ",\n";
        std::cout << "    {\"name\": \"" << JsonEscape(counter.name.c_str()) << "\", \"unit\": " << counter.unit
                  << ", \"active_finite_samples\": " << finite << ", \"active_sum\": " << std::setprecision(17) << sum
                  << ", \"active_mean\": ";if(finite)std::cout << sum/double(finite);else std::cout << "null";
        std::cout << ", \"active_min\": ";if(finite)std::cout << minimum;else std::cout << "null";
        std::cout << ", \"active_max\": ";if(finite)std::cout << maximum;else std::cout << "null";std::cout << "}";
    }
    std::cout << "\n  ]\n}\n";return 0;
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3 && argc != 5) {
        std::cerr << "usage: rgp_rdf_dump <capture.rgp> [--spm-summary | chunk-id chunk-index prefix-bytes]\n"; return 2;
    }
    try {
        rdf::ChunkFile file(argv[1]);
        if (argc==3) {
            if (std::string(argv[2])!="--spm-summary") throw std::invalid_argument("unknown mode");
            return PrintSpmSummary(file);
        }
        if (argc == 5) {
            const char* id=argv[2];const int index=std::stoi(argv[3]);const int limit=std::stoi(argv[4]);
            if (index<0 || limit<0 || limit>65536) throw std::invalid_argument("invalid bounded chunk request");
            const auto hs=file.GetChunkHeaderSize(id,index),ds=file.GetChunkDataSize(id,index);
            std::vector<unsigned char> header((size_t(hs)));
            std::vector<unsigned char> data((size_t(ds)));
            file.ReadChunkHeaderToBuffer(id,index,header.data());file.ReadChunkDataToBuffer(id,index,data.data());
            auto hex=[&](const std::vector<unsigned char>& bytes) {
                const auto count=std::min<std::size_t>(bytes.size(),size_t(limit));
                for (size_t i=0;i<count;++i) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
            };
            auto suffix=[&](const std::vector<unsigned char>& bytes) {
                const auto count=std::min<std::size_t>(bytes.size(),size_t(limit));
                for (size_t i=bytes.size()-count;i<bytes.size();++i) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
            };
            std::cout << "{\n  \"id\": \"" << JsonEscape(id) << "\", \"index\": " << std::dec << index
                      << ", \"header_bytes\": " << hs << ", \"data_bytes\": " << ds << ",\n  \"header_hex\": \"";
            hex(header);std::cout << "\",\n  \"data_prefix_hex\": \"";hex(data);
            std::cout << "\",\n  \"data_suffix_hex\": \"";suffix(data);
            std::cout << "\",\n  \"data_ascii_strings\": ";PrintStrings(data);std::cout << "\n}\n";return 0;
        }
        auto iterator=file.GetIterator();
        std::cout << "{\n  \"schema\": 1,\n  \"chunks\": [\n";
        bool first=true;
        while (!iterator.IsAtEnd()) {
            char id[RDF_IDENTIFIER_SIZE+1]={};iterator.GetChunkIdentifier(id);
            const int index=iterator.GetChunkIndex();
            if (!first) std::cout << ",\n";first=false;
            std::cout << "    {\"id\": \"" << JsonEscape(id) << "\", \"index\": " << index
                      << ", \"version\": " << file.GetChunkVersion(id,index)
                      << ", \"header_bytes\": " << file.GetChunkHeaderSize(id,index)
                      << ", \"data_bytes\": " << file.GetChunkDataSize(id,index) << "}";
            iterator.Advance();
        }
        std::cout << "\n  ]\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';return 1;
    }
}
