// binary_probe — Phase 3: static analysis of NVIDIA runtime binaries (PE imports,
// GPU payload discovery, weights localization). No proprietary file is required to
// build or run this tool: with no input it reports MISSING_PREREQUISITE.
//
// What it produces (never stores binary content, only metadata):
//   - file size + SHA-256 + PE header/section/import summary
//   - CUDA payload markers: fatbin container (magic 0x466243B1), ELF (e_machine,
//     EM_CUDA=190), PTX text markers (".version"/".target sm_*")
//   - entropy-based "blob" candidates (large high-entropy runs => compressed GPU
//     code, encrypted payloads, or packed weights)
//   - JSON manifest (--json) with all of the above
//
// Usage:
//   binary_probe.exe <path-to-dll> [--json out.json] [--max-mb 2048]
//   (or set DLSSNR_DLL_PATH / DLSS_DLL_PATH and use scripts/probe_dlssnr.ps1)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SHA-256 (self-contained, no crypto lib dependency)

struct Sha256 {
    uint32_t h[8];
    uint64_t len = 0;
    unsigned char buf[64];
    size_t bufLen = 0;

    Sha256() {
        h[0] = 0x6a09e667; h[1] = 0xbb67ae85; h[2] = 0x3c6ef372; h[3] = 0xa54ff53a;
        h[4] = 0x510e527f; h[5] = 0x9b05688c; h[6] = 0x1f83d9ab; h[7] = 0x5be0cd19;
    }
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
                   ((uint32_t)p[i*4+2] << 8) | p[i*4+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const unsigned char* p, size_t n) {
        len += n;
        while (n) {
            size_t take = 64 - bufLen;
            if (take > n) take = n;
            memcpy(buf + bufLen, p, take);
            bufLen += take; p += take; n -= take;
            if (bufLen == 64) { block(buf); bufLen = 0; }
        }
    }
    std::string finalHex() {
        uint64_t bits = len * 8;
        unsigned char pad = 0x80;
        update(&pad, 1);
        pad = 0;
        while (bufLen != 56) update(&pad, 1);
        unsigned char lb[8];
        for (int i = 0; i < 8; ++i) lb[i] = (unsigned char)(bits >> (56 - 8 * i));
        update(lb, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) snprintf(out + i * 8, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

// ---------------------------------------------------------------------------

struct Marker {
    std::string kind;
    uint64_t offset;
    std::string detail;
};

static std::string EscapeJson(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if ((unsigned char)ch < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", ch); o += b; }
        else o += ch;
    }
    return o;
}

static uint64_t RvaToOffset(const std::vector<IMAGE_SECTION_HEADER>& secs, uint64_t rva) {
    for (auto& s : secs) {
        if (rva >= s.VirtualAddress && rva < s.VirtualAddress + s.Misc.VirtualSize)
            return s.PointerToRawData + (rva - s.VirtualAddress);
    }
    return ~0ull;
}

int main(int argc, char** argv) {
    const char* path = nullptr;
    const char* jsonPath = nullptr;
    uint64_t maxBytes = 2048ull * 1024 * 1024;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json") && i + 1 < argc) jsonPath = argv[++i];
        else if (!strcmp(argv[i], "--max-mb") && i + 1 < argc) maxBytes = (uint64_t)atoll(argv[++i]) << 20;
        else if (argv[i][0] != '-') path = argv[i];
    }
    if (!path && getenv("DLSSNR_DLL_PATH")) path = getenv("DLSSNR_DLL_PATH");

    if (!path) {
        printf("MISSING_PREREQUISITE: no input file.\n");
        printf("Provide a legally obtained binary, e.g.:\n");
        printf("  binary_probe.exe <nvngx_dlssnr.dll> [--json out.json]\n");
        printf("  set DLSSNR_DLL_PATH=<path>   (extracted from a game you own)\n");
        if (jsonPath) {
            FILE* f = fopen(jsonPath, "w");
            if (f) { fprintf(f, "{\"status\": \"MISSING_PREREQUISITE\"}\n"); fclose(f); }
        }
        return 3;
    }

    // ---- read file ----
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long long fsz = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    if (fsz <= 0 || (uint64_t)fsz > maxBytes) {
        printf("ERROR: file size %lld out of range (max %llu bytes)\n", fsz, maxBytes);
        fclose(f);
        return 1;
    }
    std::vector<unsigned char> data((size_t)fsz);
    if (fread(data.data(), 1, (size_t)fsz, f) != (size_t)fsz) {
        printf("ERROR: short read\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    // ---- hash ----
    Sha256 sha;
    sha.update(data.data(), data.size());
    std::string hash = sha.finalHex();
    printf("file  : %s\n", path);
    printf("size  : %llu bytes\nsha256: %s\n", (unsigned long long)fsz, hash.c_str());

    std::vector<Marker> markers;
    std::string peMachine = "n/a", peKind = "n/a";
    uint32_t peTimestamp = 0;
    std::vector<std::string> sectionsOut;
    std::vector<std::string> importsOut;
    std::vector<std::string> delaysOut;

    // ---- PE parse ----
    std::vector<IMAGE_SECTION_HEADER> secs;
    if (fsz > 64 && data[0] == 'M' && data[1] == 'Z') {
        peKind = "MZ/PE";
        auto dos = (const IMAGE_DOS_HEADER*)data.data();
        uint64_t peOff = dos->e_lfanew;
        if (peOff + sizeof(IMAGE_NT_HEADERS64) < (uint64_t)fsz &&
            data[peOff] == 'P' && data[peOff+1] == 'E') {
            auto nt64 = (const IMAGE_NT_HEADERS64*)(data.data() + peOff);
            bool is64 = nt64->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64;
            char mb[64];
            snprintf(mb, sizeof(mb), "0x%04x%s", nt64->FileHeader.Machine,
                     nt64->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64 ? " (AMD64)" :
                     (nt64->FileHeader.Machine == IMAGE_FILE_MACHINE_I386 ? " (I386)" : ""));
            peMachine = mb;
            peTimestamp = nt64->FileHeader.TimeDateStamp;
            printf("PE    : machine=%s timestamp=%u sections=%u\n", peMachine.c_str(),
                   peTimestamp, nt64->FileHeader.NumberOfSections);

            const unsigned char* p = data.data() + peOff + sizeof(uint32_t) + sizeof(IMAGE_FILE_HEADER);
            uint16_t numSec = nt64->FileHeader.NumberOfSections;
            IMAGE_DATA_DIRECTORY ddir[16];
            if (is64) {
                auto oh = (const IMAGE_OPTIONAL_HEADER64*)p;
                memcpy(ddir, oh->DataDirectory, sizeof(ddir));
                p += sizeof(IMAGE_OPTIONAL_HEADER64);
            } else {
                auto oh = (const IMAGE_OPTIONAL_HEADER32*)p;
                memcpy(ddir, oh->DataDirectory, sizeof(ddir));
                p += sizeof(IMAGE_OPTIONAL_HEADER32);
            }
            for (uint16_t i = 0; i < numSec; ++i) {
                IMAGE_SECTION_HEADER sh;
                memcpy(&sh, p + i * sizeof(sh), sizeof(sh));
                secs.push_back(sh);
                char sb[160];
                char name[9] = {0};
                memcpy(name, sh.Name, 8);
                snprintf(sb, sizeof(sb), "sec %-8s va=0x%08x vsize=0x%08x raw=0x%08x rawsz=0x%08x flags=0x%08x",
                         name, sh.VirtualAddress, sh.Misc.VirtualSize, sh.PointerToRawData,
                         sh.SizeOfRawData, sh.Characteristics);
                sectionsOut.push_back(sb);
                printf("  %s\n", sb);
            }

            // ---- imports ----
            auto walkImports = [&](uint64_t rva) {
                uint64_t off = RvaToOffset(secs, (uint64_t)rva);
                if (off == ~0ull || off >= (uint64_t)fsz) return;
                for (;;) {
                    IMAGE_IMPORT_DESCRIPTOR imp;
                    if (off + sizeof(imp) > (uint64_t)fsz) break;
                    memcpy(&imp, data.data() + off, sizeof(imp));
                    if (!imp.Name && !imp.FirstThunk && !imp.OriginalFirstThunk) break;
                    uint64_t nameOff = RvaToOffset(secs, imp.Name);
                    if (nameOff == ~0ull) break;
                    std::string dll = (const char*)(data.data() + nameOff);
                    // IAT entries: 64-bit assumed (AMD64 binary)
                    uint64_t thunkRva = imp.OriginalFirstThunk ? imp.OriginalFirstThunk : imp.FirstThunk;
                    uint64_t tOff = RvaToOffset(secs, thunkRva);
                    int count = 0;
                    std::vector<std::string> interesting;
                    if (tOff != ~0ull) {
                        for (uint64_t tp = tOff; tp + 8 <= (uint64_t)fsz; tp += 8) {
                            uint64_t entry;
                            memcpy(&entry, data.data() + tp, 8);
                            if (!entry) break;
                            count++;
                            if (entry & 0x8000000000000000ull) continue;  // ordinal
                            uint64_t hintOff = RvaToOffset(secs, entry & 0x7FFFFFFF);
                            if (hintOff == ~0ull || hintOff + 2 >= (uint64_t)fsz) continue;
                            const char* fn = (const char*)(data.data() + hintOff + 2);
                            size_t maxLen = (size_t)(fsz - hintOff - 2);
                            std::string name(fn, strnlen(fn, maxLen < 512 ? maxLen : 512));
                            // keep CUDA / NVAPI / interop-related names explicitly
                            std::string lower;
                            lower.reserve(name.size());
                            for (char c : name) lower += (char)tolower((unsigned char)c);
                            bool gpuRel =
                                lower.find("cuda") != std::string::npos ||
                                lower.find("nvapi") != std::string::npos ||
                                lower.find("cubin") != std::string::npos ||
                                lower.find("fatbin") != std::string::npos ||
                                lower.find("ptx") != std::string::npos ||
                                (name.size() > 2 && name[0] == 'c' && name[1] == 'u' &&
                                 isupper((unsigned char)name[2]));
                            if (gpuRel) interesting.push_back(name);
                        }
                    }
                    char ib[320];
                    snprintf(ib, sizeof(ib), "%s: %d funcs", dll.c_str(), count);
                    importsOut.push_back(ib);
                    printf("  import %s\n", ib);
                    for (auto& nm : interesting) printf("    * %s\n", nm.c_str());
                    off += sizeof(imp);
                    if (importsOut.size() > 64) break;
                }
            };
            if (ddir[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress)
                walkImports(ddir[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress);

            // delay imports use ImgDelayDescr (not IMAGE_IMPORT_DESCRIPTOR)
            if (ddir[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT].VirtualAddress) {
                uint64_t off = RvaToOffset(secs, ddir[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT].VirtualAddress);
                if (off != ~0ull) {
                    for (;;) {
                        struct { uint32_t grAttrs, rvaDLLName, rvaHmod, rvaIAT, rvaINT,
                                         rvaBoundIAT, rvaUnloadIAT, dwTimeStamp; } d;
                        if (off + sizeof(d) > (uint64_t)fsz) break;
                        memcpy(&d, data.data() + off, sizeof(d));
                        if (!d.rvaDLLName) break;
                        uint64_t nameOff = RvaToOffset(secs, d.rvaDLLName);
                        std::string dll = nameOff != ~0ull
                            ? std::string((const char*)(data.data() + nameOff)) : "?";
                        int count = 0;
                        uint64_t tOff = RvaToOffset(secs, d.rvaINT);
                        if (tOff != ~0ull) {
                            for (uint64_t tp = tOff; tp + 8 <= (uint64_t)fsz; tp += 8) {
                                uint64_t entry; memcpy(&entry, data.data() + tp, 8);
                                if (!entry) break;
                                count++;
                            }
                        }
                        char ib[320];
                        snprintf(ib, sizeof(ib), "DELAY %s: %d funcs", dll.c_str(), count);
                        importsOut.push_back(ib);
                        delaysOut.push_back(ib);
                        printf("  %s\n", ib);
                        off += sizeof(d);
                        if (delaysOut.size() > 32) break;
                    }
                }
            }

            if (ddir[IMAGE_DIRECTORY_ENTRY_RESOURCE].Size)
                printf("  .rsrc present: size=%u\n", ddir[IMAGE_DIRECTORY_ENTRY_RESOURCE].Size);
        }
    } else {
        printf("file is not a PE image; continuing marker scan\n");
    }

    // ---- GPU payload marker scan ----
    // fatbin container magic 0x466243B1 (little-endian bytes B1 43 62 46)
    {
        int hits = 0;
        for (size_t i = 0; i + 16 <= data.size(); ++i) {
            if (data[i] == 0xB1 && data[i+1] == 0x43 && data[i+2] == 0x62 && data[i+3] == 0x46) {
                uint16_t ver; uint64_t fatSize;
                memcpy(&ver, &data[i+4], 2);
                memcpy(&fatSize, &data[i+8], 8);
                if (ver <= 3 && fatSize < (uint64_t)fsz) {
                    char b[128];
                    snprintf(b, sizeof(b), "ver=%u fatSize=%llu", ver, (unsigned long long)fatSize);
                    markers.push_back({"fatbin", (uint64_t)i, b});
                    printf("fatbin marker @0x%llx (%s)\n", (unsigned long long)i, b);
                    if (++hits >= 32) break;
                }
            }
        }
    }
    // ELF payloads (CUBIN has e_machine=190/EM_CUDA; also count generic ELF)
    {
        int hits = 0, cuda = 0;
        for (size_t i = 0; i + 0x14 <= data.size(); ++i) {
            if (data[i] == 0x7F && data[i+1] == 'E' && data[i+2] == 'L' && data[i+3] == 'F') {
                uint16_t em;
                memcpy(&em, &data[i + 0x12], 2);
                if (em == 190) cuda++;
                if (hits < 32) {
                    char b[64];
                    snprintf(b, sizeof(b), "e_machine=%u%s", em, em == 190 ? " EM_CUDA" : "");
                    markers.push_back({"elf", (uint64_t)i, b});
                }
                hits++;
                if (hits >= 4096) break;   // pathological guard
            }
        }
        printf("elf payloads: %d found (%d with EM_CUDA=190)\n", hits, cuda);
    }
    // PTX text markers
    {
        auto scanText = [&](const char* needle, const char* kind, int cap) {
            size_t nl = strlen(needle);
            int hits = 0;
            for (size_t i = 0; i + nl + 32 <= data.size(); ++i) {
                if (memcmp(&data[i], needle, nl) == 0) {
                    // capture context line
                    size_t end = i;
                    while (end < data.size() && end < i + 80 && data[end] != '\n' && data[end] != '\0') end++;
                    std::string ctx((const char*)&data[i], end - i);
                    markers.push_back({kind, (uint64_t)i, ctx});
                    if (++hits >= cap) break;
                }
            }
            if (hits) printf("%s: %d occurrence(s), e.g. \"%s\"\n", kind, hits,
                             markers.back().detail.c_str());
        };
        scanText(".target sm_", "ptx_target", 16);
        scanText(".version", "ptx_version", 8);
        scanText(".visible .entry", "ptx_entry", 8);
    }

    // ---- entropy / blob candidates (weights & compressed payload localization) ----
    {
        const size_t WIN = 1 << 16;
        std::vector<double> ent;
        ent.reserve(data.size() / WIN + 1);
        for (size_t off = 0; off < data.size(); off += WIN) {
            size_t n = std::min(WIN, data.size() - off);
            uint32_t freq[256] = {0};
            for (size_t j = 0; j < n; ++j) freq[data[off + j]]++;
            double e = 0.0;
            for (int k = 0; k < 256; ++k) {
                if (freq[k]) {
                    double p = (double)freq[k] / (double)n;
                    e -= p * log2(p);
                }
            }
            ent.push_back(e);
        }
        // find runs of high entropy (>= 7.9 bits/byte)
        std::vector<std::string> blobs;
        size_t runStart = SIZE_MAX, runWindows = 0;
        double runMax = 0;
        for (size_t w = 0; w <= ent.size(); ++w) {
            bool high = (w < ent.size()) && (ent[w] >= 7.9);
            if (high) {
                if (runStart == SIZE_MAX) { runStart = w; runWindows = 0; runMax = 0; }
                runWindows++;
                if (ent[w] > runMax) runMax = ent[w];
            } else if (runStart != SIZE_MAX) {
                uint64_t bytes = (uint64_t)runWindows * WIN;
                if (bytes >= 256 * 1024) {   // report runs >= 256 KB
                    char b[192];
                    snprintf(b, sizeof(b), "run @0x%llx size=%llu bytes maxEnt=%.3f",
                             (unsigned long long)(runStart * WIN), (unsigned long long)bytes, runMax);
                    blobs.push_back(b);
                    printf("high-entropy %s\n", b);
                }
                runStart = SIZE_MAX;
            }
        }
        if (blobs.empty()) printf("no high-entropy runs >= 256KB (weights may be low-entropy fp16/fp8 tables)\n");
        for (auto& b : blobs) markers.push_back({"entropy_run", 0, b});
    }

    // ---- JSON manifest ----
    FILE* jf = jsonPath ? fopen(jsonPath, "w") : nullptr;
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"file\": \"%s\",\n", EscapeJson(path).c_str());
        fprintf(jf, "  \"size\": %llu,\n", (unsigned long long)fsz);
        fprintf(jf, "  \"sha256\": \"%s\",\n", hash.c_str());
        fprintf(jf, "  \"pe\": {\"kind\": \"%s\", \"machine\": \"%s\", \"timestamp\": %u},\n",
                 peKind.c_str(), peMachine.c_str(), peTimestamp);
        fprintf(jf, "  \"sections\": [");
        for (size_t i = 0; i < sectionsOut.size(); ++i)
            fprintf(jf, "%s\n    \"%s\"", i ? "," : "", EscapeJson(sectionsOut[i]).c_str());
        fprintf(jf, "%s],\n", sectionsOut.empty() ? "" : "\n  ");
        fprintf(jf, "  \"imports\": [");
        for (size_t i = 0; i < importsOut.size(); ++i)
            fprintf(jf, "%s\n    \"%s\"", i ? "," : "", EscapeJson(importsOut[i]).c_str());
        fprintf(jf, "%s],\n", importsOut.empty() ? "" : "\n  ");
        fprintf(jf, "  \"markers\": [");
        for (size_t i = 0; i < markers.size(); ++i) {
            auto& m = markers[i];
            fprintf(jf, "%s\n    {\"kind\": \"%s\", \"offset\": %llu, \"detail\": \"%s\"}",
                    i ? "," : "", m.kind.c_str(), (unsigned long long)m.offset,
                    EscapeJson(m.detail).c_str());
        }
        fprintf(jf, "%s],\n", markers.empty() ? "" : "\n  ");
        fprintf(jf, "  \"status\": \"analyzed\"\n}\n");
        fclose(jf);
        printf("manifest written: %s\n", jsonPath);
    }

    if (markers.empty() && peKind == "n/a") {
        printf("\nNOTE: no GPU payload markers found; binary may store kernels elsewhere\n");
        printf("      (e.g. loaded from a sidecar file or NGX blob at runtime).\n");
    }
    return 0;
}
