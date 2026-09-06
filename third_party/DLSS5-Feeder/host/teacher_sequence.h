// Local research harness: actual FP16 color/depth/motion, explicit metadata.
// Outputs are teacher CANDIDATES until feature-18 and input identity are audited.
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>

static void TeacherCheck(HRESULT hr) {
    if (FAILED(hr)) throw std::runtime_error("teacher D3D12 operation failed");
}
static void TeacherBarrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    h.list->ResourceBarrier(1, &barrier);
}
static ID3D12Resource* TeacherBuffer(UINT64 bytes, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = type;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* resource = nullptr;
    TeacherCheck(h.dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&resource)));
    return resource;
}
static void TeacherTransfer(ID3D12Resource* texture, const std::filesystem::path& file, UINT pixelBytes, bool upload) {
    auto desc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows; UINT64 rowBytes, total;
    h.dev->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
    const size_t tightRow = size_t(desc.Width) * pixelBytes;
    if (rowBytes != tightRow || rows != desc.Height) throw std::runtime_error("teacher texture footprint mismatch");
    std::vector<char> raw(tightRow * rows);
    if (upload) {
        std::ifstream input(file, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != std::streamoff(raw.size())) throw std::runtime_error("missing/wrong-size teacher input");
        input.seekg(0); if (!input.read(raw.data(), raw.size())) throw std::runtime_error("teacher input read failed");
    } else if (std::filesystem::exists(file)) throw std::runtime_error("refusing to overwrite teacher output");
    auto buffer = TeacherBuffer(total, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK);
    char* mapped = nullptr;
    if (upload) {
        D3D12_RANGE empty{}; TeacherCheck(buffer->Map(0, &empty, reinterpret_cast<void**>(&mapped)));
        for (UINT y = 0; y < rows; ++y) memcpy(mapped + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, raw.data() + y * tightRow, tightRow);
        buffer->Unmap(0, nullptr);
    }
    if (!BeginCommands()) throw std::runtime_error("teacher begin commands failed");
    auto state = upload ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
    TeacherBarrier(texture, D3D12_RESOURCE_STATE_COMMON, state);
    D3D12_TEXTURE_COPY_LOCATION linear{}, tex{};
    linear.pResource = buffer; linear.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; linear.PlacedFootprint = footprint;
    tex.pResource = texture; tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    h.list->CopyTextureRegion(upload ? &tex : &linear, 0, 0, 0, upload ? &linear : &tex, nullptr);
    TeacherBarrier(texture, state, D3D12_RESOURCE_STATE_COMMON);
    // Retain potentially in-flight resources on failure; process teardown owns them.
    if (!WaitFenceValue(h.fence, EndCommands(), 4000)) throw std::runtime_error("teacher transfer timeout; stop, no retry");
    TeacherCheck(h.dev->GetDeviceRemovedReason());
    if (!upload) {
        D3D12_RANGE range{0, SIZE_T(total)}; TeacherCheck(buffer->Map(0, &range, reinterpret_cast<void**>(&mapped)));
        for (UINT y = 0; y < rows; ++y) memcpy(raw.data() + y * tightRow, mapped + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, tightRow);
        D3D12_RANGE empty{}; buffer->Unmap(0, &empty);
        std::ofstream out(file, std::ios::binary); out.write(raw.data(), raw.size());
        if (!out) throw std::runtime_error("teacher output write failed");
    }
    buffer->Release();
}
static int RunTeacherSequence(const char* directory) { try {
    const auto root = std::filesystem::absolute(directory);
    std::ifstream config(root / "sequence.txt");
    std::string magic; unsigned width = 0, height = 0, count = 0; int flags = 0;
    if (!(config >> magic >> width >> height >> count >> flags) || magic != "NRTEACHER1" ||
        !width || !height || uint64_t(width) * height > 3840ull * 2160 || !count || count > 32)
        throw std::runtime_error("invalid teacher sequence header");
    auto color = MakeTex(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
    auto output = MakeTex(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
    auto depth = MakeTex(width, height, DXGI_FORMAT_R32_FLOAT, false);
    auto motion = MakeTex(width, height, DXGI_FORMAT_R16G16_FLOAT, false);
    if (!color || !output || !depth || !motion) throw std::runtime_error("teacher texture allocation failed");
    for (int i = 0; i < 120; ++i) { PumpPresent(); Sleep(8); }
    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    if (!CreateFeature(width, height, flags, &result)) throw std::runtime_error("teacher feature creation failed");
    for (unsigned frame = 0; frame < count; ++frame) {
        unsigned index; int reset; float pre, exposure, mx, my, jx, jy;
        if (!(config >> index >> reset >> pre >> exposure >> mx >> my >> jx >> jy) || index != frame ||
            (reset != 0 && reset != 1) || (!frame && !reset) || !std::isfinite(pre) || pre <= 0 ||
            !std::isfinite(exposure) || exposure <= 0 || !std::isfinite(mx) || !std::isfinite(my) ||
            !std::isfinite(jx) || !std::isfinite(jy)) throw std::runtime_error("invalid explicit frame metadata");
        auto dir = root / std::to_string(frame);
        TeacherTransfer(color, dir / "color.raw", 8, true);
        TeacherTransfer(depth, dir / "depth.raw", 4, true);
        TeacherTransfer(motion, dir / "motion.raw", 4, true);
        // Verify the actual texture input before allowing a teacher output candidate.
        TeacherTransfer(color, dir / "uploaded_color.raw", 8, false);
        PumpPresent();
        if (!Evaluate(color, output, depth, motion, width, height, reset, mx, my, pre, exposure, jx, jy) ||
            !WaitFenceValue(h.fence, h.fence_value, 4000)) throw std::runtime_error("teacher evaluation failed/timed out");
        TeacherTransfer(output, dir / "candidate_output.raw", 8, false);
        Log("[teacher] candidate frame %u complete; feature18/output identity acceptance still required", frame);
    }
    output->Release(); color->Release(); depth->Release(); motion->Release();
    return 0;
} catch (const std::exception& error) { Log("[teacher] FAILED: %s", error.what()); return 1; } }
