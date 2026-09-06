#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace full_graph_dx12 {

// The captured DLSS graph produces a 640x360 image from a 640x384 working
// tensor.  Keep that graph shape as a compatibility tile while the individual
// operator families are made natively dynamic.
constexpr uint32_t CompatibilityTileWidth = 640;
constexpr uint32_t CompatibilityTileHeight = 360;
constexpr uint32_t CompatibilityWorkHeight = 384;
constexpr uint32_t CompatibilityVerticalHalo =
    (CompatibilityWorkHeight - CompatibilityTileHeight) / 2;
constexpr uint32_t HorizontalOverlap = 64;
constexpr uint32_t VerticalOverlap = 40;
constexpr uint32_t D3D12MaxTextureDimension = 16384;
constexpr uint32_t CompatibilityResidentTiles = 16;

inline uint32_t AlignUp(uint32_t value, uint32_t alignment) {
    if (!value || !alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("invalid resolution alignment");
    const uint64_t result = (uint64_t(value) + alignment - 1) & ~(uint64_t(alignment) - 1);
    if (result > UINT32_MAX) throw std::overflow_error("aligned resolution overflow");
    return uint32_t(result);
}

struct AxisTile {
    uint32_t origin = 0;
    uint32_t writeBegin = 0;
    uint32_t writeEnd = 0; // Exclusive, in frame coordinates.
};

inline std::vector<AxisTile> BuildAxisTiles(uint32_t extent, uint32_t tileExtent,
                                            uint32_t overlap) {
    if (!extent || !tileExtent || overlap >= tileExtent)
        throw std::invalid_argument("invalid tile axis");

    std::vector<uint32_t> origins{0};
    if (extent > tileExtent) {
        const uint32_t stride = tileExtent - overlap;
        while (origins.back() + tileExtent < extent) {
            const uint32_t next = std::min(origins.back() + stride, extent - tileExtent);
            if (next == origins.back()) break;
            origins.push_back(next);
        }
    }

    std::vector<uint32_t> boundaries(origins.size() + 1);
    boundaries.front() = 0;
    boundaries.back() = extent;
    for (size_t i = 0; i + 1 < origins.size(); ++i) {
        // Split the shared valid region at its midpoint.  This assigns every
        // output pixel to exactly one tile without blending or double writes.
        boundaries[i + 1] = uint32_t((uint64_t(origins[i]) + tileExtent +
                                      uint64_t(origins[i + 1])) / 2);
    }

    std::vector<AxisTile> result;
    result.reserve(origins.size());
    for (size_t i = 0; i < origins.size(); ++i)
        result.push_back({origins[i], boundaries[i], boundaries[i + 1]});
    return result;
}

// This layout is intentionally six uint32 values so it is identical to the
// StructuredBuffer<TileDescriptor> declaration in the pack/unpack shaders.
struct TileDescriptor {
    uint32_t originX;
    uint32_t originY;
    uint32_t writeBeginX;
    uint32_t writeBeginY;
    uint32_t writeEndX;
    uint32_t writeEndY;
};
static_assert(sizeof(TileDescriptor) == 24);

struct ResolutionPlan {
    uint32_t frameWidth = 0;
    uint32_t frameHeight = 0;

    // Prospective native-dynamic graph shape. Operator ports consume these
    // values instead of embedding the reference resolution in their shaders.
    uint32_t nativeWorkWidth = 0;
    uint32_t nativeWorkHeight = 0;
    uint32_t nativeCropTop = 0;
    uint32_t nativeCropBottom = 0;

    // Guaranteed compatibility route: fixed captured graph shape, arbitrarily
    // many overlapping tiles, then deterministic ownership-based assembly.
    std::vector<TileDescriptor> tiles;

    static ResolutionPlan Build(uint32_t width, uint32_t height) {
        if (!width || !height || width > D3D12MaxTextureDimension ||
            height > D3D12MaxTextureDimension)
            throw std::invalid_argument("frame resolution exceeds D3D12 texture contract");

        ResolutionPlan plan{};
        plan.frameWidth = width;
        plan.frameHeight = height;
        plan.nativeWorkWidth = AlignUp(width, 8);
        plan.nativeWorkHeight = AlignUp(height + 2 * CompatibilityVerticalHalo, 8);
        const uint32_t totalCrop = plan.nativeWorkHeight - height;
        plan.nativeCropTop = totalCrop / 2;
        plan.nativeCropBottom = totalCrop - plan.nativeCropTop;

        const auto xs = BuildAxisTiles(width, CompatibilityTileWidth, HorizontalOverlap);
        const auto ys = BuildAxisTiles(height, CompatibilityTileHeight, VerticalOverlap);
        plan.tiles.reserve(xs.size() * ys.size());
        for (const auto& y : ys)
            for (const auto& x : xs)
                plan.tiles.push_back({x.origin, y.origin, x.writeBegin, y.writeBegin,
                                      x.writeEnd, y.writeEnd});
        return plan;
    }

    uint32_t BatchCount() const {
        return uint32_t((tiles.size() + CompatibilityResidentTiles - 1) /
                        CompatibilityResidentTiles);
    }
};

} // namespace full_graph_dx12
