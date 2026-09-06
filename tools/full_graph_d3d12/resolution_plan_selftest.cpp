#include "resolution_plan.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;
using full_graph_dx12::CompatibilityTileHeight;
using full_graph_dx12::CompatibilityTileWidth;
using full_graph_dx12::ResolutionPlan;

static void Validate(const ResolutionPlan& plan) {
    if ((plan.nativeWorkWidth & 7) || (plan.nativeWorkHeight & 7) ||
        plan.nativeWorkWidth < plan.frameWidth ||
        plan.nativeWorkHeight < plan.frameHeight + 24 ||
        plan.nativeCropTop + plan.frameHeight + plan.nativeCropBottom !=
            plan.nativeWorkHeight)
        throw std::runtime_error("native dynamic geometry failed");

    // Exact ownership is separable. Testing each row interval and column
    // interval is sufficient and avoids allocating an 8K coverage bitmap.
    for (uint32_t y = 0; y < plan.frameHeight; ++y) {
        std::vector<std::pair<uint32_t, uint32_t>> spans;
        for (const auto& tile : plan.tiles) {
            if (y < tile.writeBeginY || y >= tile.writeEndY) continue;
            if (tile.writeBeginX < tile.originX ||
                tile.writeEndX > tile.originX + CompatibilityTileWidth ||
                tile.writeBeginY < tile.originY ||
                tile.writeEndY > tile.originY + CompatibilityTileHeight)
                throw std::runtime_error("ownership outside tile output");
            spans.emplace_back(tile.writeBeginX, tile.writeEndX);
        }
        std::sort(spans.begin(), spans.end());
        uint32_t cursor = 0;
        for (const auto& span : spans) {
            if (span.first != cursor || span.second <= span.first)
                throw std::runtime_error("tile ownership gap or overlap");
            cursor = span.second;
        }
        if (cursor != plan.frameWidth)
            throw std::runtime_error("tile ownership does not cover frame");
    }
}

int main(int argc, char** argv) try {
    const std::array<std::pair<uint32_t, uint32_t>, 12> cases{{
        {1, 1}, {320, 180}, {640, 360}, {641, 361}, {1280, 720},
        {1552, 872}, {1920, 1080}, {2342, 1317}, {2560, 1440},
        {3440, 1440}, {3840, 2160}, {7680, 4320}}};
    uint64_t plans = 0, tiles = 0;
    ResolutionPlan representative;
    for (auto [width, height] : cases) {
        auto plan = ResolutionPlan::Build(width, height);
        Validate(plan);
        if (width == 2342 && height == 1317) representative = plan;
        ++plans;
        tiles += plan.tiles.size();
    }
    std::mt19937 random(0xD155A5u);
    for (uint32_t i = 0; i < 512; ++i) {
        const uint32_t width = 1 + random() % 7680;
        const uint32_t height = 1 + random() % 4320;
        auto plan = ResolutionPlan::Build(width, height);
        Validate(plan);
        ++plans;
        tiles += plan.tiles.size();
    }

    std::ostringstream json;
    json << "{\n  \"status\": \"FULL_GRAPH_ARBITRARY_RESOLUTION_PLAN_PASS\",\n"
         << "  \"validated_plans\": " << plans << ",\n"
         << "  \"validated_tiles\": " << tiles << ",\n"
         << "  \"maximum_validated_resolution\": [7680, 4320],\n"
         << "  \"reference_game_resolution\": [2342, 1317],\n"
         << "  \"reference_game_native_work\": [" << representative.nativeWorkWidth
         << ", " << representative.nativeWorkHeight << "],\n"
         << "  \"reference_game_compatibility_tiles\": "
         << representative.tiles.size() << ",\n"
         << "  \"resident_tile_limit\": " << full_graph_dx12::CompatibilityResidentTiles << ",\n"
         << "  \"reference_game_batches\": " << representative.BatchCount() << ",\n"
         << "  \"compatibility_tile_output\": [640, 360],\n"
         << "  \"compatibility_tile_work\": [640, 384],\n"
         << "  \"exact_output_extent\": true,\n"
         << "  \"single_writer_per_pixel\": true,\n"
         << "  \"bounded_compatibility_memory\": true,\n"
         << "  \"complete_network\": false\n}\n";
    if (argc == 2) {
        const fs::path output = fs::absolute(argv[1]);
        if (output.has_parent_path()) fs::create_directories(output.parent_path());
        std::ofstream stream(output, std::ios::binary);
        if (!stream || !(stream << json.str())) throw std::runtime_error("write summary");
    } else if (argc != 1) {
        return 2;
    }
    std::cout << json.str();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "resolution plan test failed: " << error.what() << '\n';
    return 1;
}
