/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <archive.h>
#include <archive_entry.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <vector>

#include "core/logger.hpp"
#include "core/sogs.hpp"
#include "core/sogs_new.hpp"
#include "core/splat_data.hpp"

namespace fs = std::filesystem;

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-3f;
    constexpr float CODEBOOK_TOLERANCE = 0.1f;

    class TempDirectory {
        fs::path path_;

    public:
        TempDirectory() {
            path_ = fs::temp_directory_path() / ("sog_test_" + std::to_string(rand()));
            fs::create_directories(path_);
        }

        ~TempDirectory() {
            if (fs::exists(path_)) {
                fs::remove_all(path_);
            }
        }

        fs::path path() const { return path_; }
        operator fs::path() const { return path_; }
    };

    gs::SplatData create_test_splat_data(int num_splats, int sh_degree = 0) {
        auto means = torch::randn({num_splats, 3}, torch::kCUDA) * 10.0f;
        auto scales = torch::randn({num_splats, 3}, torch::kCUDA).abs() * 0.5f + 0.1f;
        auto rotations = torch::randn({num_splats, 4}, torch::kCUDA);
        rotations = rotations / rotations.norm(2, 1, true);
        auto opacities = torch::rand({num_splats}, torch::kCUDA);

        // sh0 is [N, 3, 1], shN is [N, 3, num_coeffs]
        auto sh0 = torch::randn({num_splats, 3, 1}, torch::kCUDA) * 0.5f;

        torch::Tensor shN;
        if (sh_degree > 0) {
            int sh_coeffs = (sh_degree == 1) ? 3 : (sh_degree == 2) ? 8
                                               : (sh_degree == 3)   ? 15
                                                                    : 0;
            shN = torch::randn({num_splats, 3, sh_coeffs}, torch::kCUDA) * 0.1f;
        } else {
            shN = torch::empty({num_splats, 3, 0}, torch::kCUDA);
        }

        // Constructor: (sh_degree, means, sh0, shN, scaling, rotation, opacity, scene_scale)
        gs::SplatData data(sh_degree, means, sh0, shN, scales, rotations, opacities, 1.0f);
        return data;
    }

    nlohmann::json load_meta_json(const fs::path& base_path) {
        auto meta_path = base_path / "meta.json";
        if (!fs::exists(meta_path)) {
            throw std::runtime_error("meta.json not found");
        }
        std::ifstream file(meta_path);
        nlohmann::json meta;
        file >> meta;
        return meta;
    }

    nlohmann::json extract_and_load_meta_from_sog(const fs::path& sog_path) {
        struct archive* a = archive_read_new();
        archive_read_support_format_zip(a);

        if (archive_read_open_filename(a, sog_path.string().c_str(), 10240) != ARCHIVE_OK) {
            archive_read_free(a);
            throw std::runtime_error("Failed to open .sog archive");
        }

        struct archive_entry* entry;
        nlohmann::json meta;
        bool found = false;

        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            std::string pathname = archive_entry_pathname(entry);
            if (pathname == "meta.json") {
                size_t size = archive_entry_size(entry);
                std::vector<char> buffer(size);
                archive_read_data(a, buffer.data(), size);
                meta = nlohmann::json::parse(buffer.begin(), buffer.end());
                found = true;
                break;
            }
            archive_read_data_skip(a);
        }

        archive_read_free(a);
        if (!found) {
            throw std::runtime_error("meta.json not found in archive");
        }
        return meta;
    }

    bool compare_codebooks(const std::vector<float>& codebook1,
                           const std::vector<float>& codebook2,
                           float tolerance = CODEBOOK_TOLERANCE) {
        if (codebook1.size() != codebook2.size()) {
            LOG_ERROR("Codebook size mismatch: {} vs {}", codebook1.size(), codebook2.size());
            return false;
        }

        auto min1 = *std::min_element(codebook1.begin(), codebook1.end());
        auto max1 = *std::max_element(codebook1.begin(), codebook1.end());
        auto min2 = *std::min_element(codebook2.begin(), codebook2.end());
        auto max2 = *std::max_element(codebook2.begin(), codebook2.end());

        float range1 = max1 - min1;
        float range2 = max2 - min2;

        float min_diff = std::abs(min1 - min2) / std::max(std::abs(min1), std::abs(min2));
        float max_diff = std::abs(max1 - max2) / std::max(std::abs(max1), std::abs(max2));
        float range_diff = std::abs(range1 - range2) / std::max(range1, range2);

        bool result = (min_diff < tolerance && max_diff < tolerance && range_diff < tolerance);

        if (!result) {
            LOG_ERROR("Codebook range mismatch: [{}, {}] vs [{}, {}]", min1, max1, min2, max2);
        }
        return result;
    }

    bool compare_meta_json(const nlohmann::json& meta1, const nlohmann::json& meta2) {
        if (meta1["version"] != meta2["version"]) {
            LOG_ERROR("Version mismatch: {} vs {}",
                      meta1["version"].get<int>(), meta2["version"].get<int>());
            return false;
        }

        if (meta1["count"] != meta2["count"]) {
            LOG_ERROR("Count mismatch: {} vs {}",
                      meta1["count"].get<int>(), meta2["count"].get<int>());
            return false;
        }

        if (meta1["width"] != meta2["width"] || meta1["height"] != meta2["height"]) {
            LOG_ERROR("Dimensions mismatch: {}x{} vs {}x{}",
                      meta1["width"].get<int>(), meta1["height"].get<int>(),
                      meta2["width"].get<int>(), meta2["height"].get<int>());
            return false;
        }

        auto mins1 = meta1["means"]["mins"].get<std::vector<float>>();
        auto mins2 = meta2["means"]["mins"].get<std::vector<float>>();
        auto maxs1 = meta1["means"]["maxs"].get<std::vector<float>>();
        auto maxs2 = meta2["means"]["maxs"].get<std::vector<float>>();

        for (size_t i = 0; i < 3; ++i) {
            float diff_min = std::abs(mins1[i] - mins2[i]) / std::max(std::abs(mins1[i]), 1e-6f);
            float diff_max = std::abs(maxs1[i] - maxs2[i]) / std::max(std::abs(maxs1[i]), 1e-6f);

            if (diff_min > FLOAT_TOLERANCE || diff_max > FLOAT_TOLERANCE) {
                LOG_ERROR("Means min/max mismatch at dim {}: min diff={}, max diff={}",
                          i, diff_min, diff_max);
                return false;
            }
        }

        auto scale_codebook1 = meta1["scales"]["codebook"].get<std::vector<float>>();
        auto scale_codebook2 = meta2["scales"]["codebook"].get<std::vector<float>>();
        if (!compare_codebooks(scale_codebook1, scale_codebook2)) {
            return false;
        }

        auto color_codebook1 = meta1["sh0"]["codebook"].get<std::vector<float>>();
        auto color_codebook2 = meta2["sh0"]["codebook"].get<std::vector<float>>();
        if (!compare_codebooks(color_codebook1, color_codebook2)) {
            return false;
        }

        if (meta1.contains("shN") && meta2.contains("shN")) {
            if (meta1["shN"]["bands"] != meta2["shN"]["bands"]) {
                LOG_ERROR("SH bands mismatch: {} vs {}",
                          meta1["shN"]["bands"].get<int>(), meta2["shN"]["bands"].get<int>());
                return false;
            }

            if (meta1["shN"]["coeffs"] != meta2["shN"]["coeffs"]) {
                LOG_ERROR("SH coeffs mismatch: {} vs {}",
                          meta1["shN"]["coeffs"].get<int>(), meta2["shN"]["coeffs"].get<int>());
                return false;
            }

            int palette1 = meta1["shN"]["palette_size"].get<int>();
            int palette2 = meta2["shN"]["palette_size"].get<int>();
            float palette_diff = std::abs(palette1 - palette2) / static_cast<float>(std::max(palette1, palette2));

            if (palette_diff > 0.2f) {
                LOG_ERROR("SH palette size mismatch: {} vs {}", palette1, palette2);
                return false;
            }

            auto sh_codebook1 = meta1["shN"]["codebook"].get<std::vector<float>>();
            auto sh_codebook2 = meta2["shN"]["codebook"].get<std::vector<float>>();
            if (!compare_codebooks(sh_codebook1, sh_codebook2, 0.3f)) {
                return false;
            }
        } else if (meta1.contains("shN") != meta2.contains("shN")) {
            LOG_ERROR("SH presence mismatch");
            return false;
        }

        return true;
    }

} // anonymous namespace

class SOGComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        gs::Tensor::manual_seed(42);
    }
};

TEST_F(SOGComparisonTest, BasicOutputStructure_Directory) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(1000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old" / "meta.json";
    options_old.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    ASSERT_TRUE(result_old) << "Old failed: " << result_old.error();

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new" / "meta.json";
    options_new.iterations = 10;

    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_new) << "New failed: " << result_new.error();

    auto old_dir = temp_dir.path() / "old";
    auto new_dir = temp_dir.path() / "new";

    EXPECT_TRUE(fs::exists(old_dir / "meta.json"));
    EXPECT_TRUE(fs::exists(new_dir / "meta.json"));
    EXPECT_TRUE(fs::exists(old_dir / "means_l.webp"));
    EXPECT_TRUE(fs::exists(new_dir / "means_l.webp"));
    EXPECT_TRUE(fs::exists(old_dir / "means_u.webp"));
    EXPECT_TRUE(fs::exists(new_dir / "means_u.webp"));
    EXPECT_TRUE(fs::exists(old_dir / "quats.webp"));
    EXPECT_TRUE(fs::exists(new_dir / "quats.webp"));
    EXPECT_TRUE(fs::exists(old_dir / "scales.webp"));
    EXPECT_TRUE(fs::exists(new_dir / "scales.webp"));
    EXPECT_TRUE(fs::exists(old_dir / "sh0.webp"));
    EXPECT_TRUE(fs::exists(new_dir / "sh0.webp"));

    auto meta_old = load_meta_json(old_dir);
    auto meta_new = load_meta_json(new_dir);
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, BasicOutputStructure_Bundle) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(1000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    ASSERT_TRUE(result_old);

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_new);

    EXPECT_TRUE(fs::exists(temp_dir.path() / "old.sog"));
    EXPECT_TRUE(fs::exists(temp_dir.path() / "new.sog"));

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, MetadataConsistency_SmallDataset) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(500, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, MetadataConsistency_MediumDataset) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(5000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 15;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 15;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, MetadataConsistency_LargeDataset) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(20000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, SphericalHarmonics_Degree1) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(2000, 1);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");

    EXPECT_TRUE(meta_old.contains("shN"));
    EXPECT_TRUE(meta_new.contains("shN"));
    EXPECT_EQ(meta_old["shN"]["bands"], 1);
    EXPECT_EQ(meta_new["shN"]["bands"], 1);
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, SphericalHarmonics_Degree2) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(3000, 2);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");

    EXPECT_TRUE(meta_old.contains("shN"));
    EXPECT_TRUE(meta_new.contains("shN"));
    EXPECT_EQ(meta_old["shN"]["bands"], 2);
    EXPECT_EQ(meta_new["shN"]["bands"], 2);
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, SphericalHarmonics_Degree3) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(4000, 3);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");

    EXPECT_TRUE(meta_old.contains("shN"));
    EXPECT_TRUE(meta_new.contains("shN"));
    EXPECT_EQ(meta_old["shN"]["bands"], 3);
    EXPECT_EQ(meta_new["shN"]["bands"], 3);
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, IterationVariation_5Iterations) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(1000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 5;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 5;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, IterationVariation_20Iterations) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(1000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 20;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 20;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, EdgeCase_VerySmallDataset) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(100, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, EdgeCase_PowerOfTwo) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(1024, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, RealWorld_TypicalScene) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(50000, 2);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));

    auto old_size = fs::file_size(temp_dir.path() / "old.sog");
    auto new_size = fs::file_size(temp_dir.path() / "new.sog");

    EXPECT_GT(old_size, 0);
    EXPECT_GT(new_size, 0);

    float size_diff = std::abs(static_cast<float>(old_size) - static_cast<float>(new_size)) /
                      std::max(old_size, new_size);
    EXPECT_LT(size_diff, 0.15f) << "Size diff: " << old_size << " vs " << new_size;
}

TEST_F(SOGComparisonTest, RealWorld_LargeComplexScene) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(100000, 3);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");
    EXPECT_TRUE(compare_meta_json(meta_old, meta_new));
}

TEST_F(SOGComparisonTest, FormatConsistency_TextureDimensions) {
    TempDirectory temp_dir;
    std::vector<int> splat_counts = {500, 1234, 5000, 10000, 32768};

    for (int count : splat_counts) {
        auto splat_data = create_test_splat_data(count, 0);

        auto old_path = temp_dir.path() / ("old_" + std::to_string(count) + ".sog");
        auto new_path = temp_dir.path() / ("new_" + std::to_string(count) + ".sog");

        gs::core::SogWriteOptions options_old;
        options_old.output_path = old_path;
        options_old.iterations = 5;

        gs::core::SogWriteOptionsNew options_new;
        options_new.output_path = new_path;
        options_new.iterations = 5;

        auto result_old = gs::core::write_sog(splat_data, options_old);
        auto result_new = gs::core::write_sog_new(splat_data, options_new);
        ASSERT_TRUE(result_old && result_new) << "Failed for count=" << count;

        auto meta_old = extract_and_load_meta_from_sog(old_path);
        auto meta_new = extract_and_load_meta_from_sog(new_path);

        EXPECT_EQ(meta_old["width"], meta_new["width"]) << "Width mismatch for count=" << count;
        EXPECT_EQ(meta_old["height"], meta_new["height"]) << "Height mismatch for count=" << count;
        EXPECT_EQ(meta_old["count"], count);
        EXPECT_EQ(meta_new["count"], count);
    }
}

TEST_F(SOGComparisonTest, FormatConsistency_CodebookSize) {
    TempDirectory temp_dir;
    auto splat_data = create_test_splat_data(10000, 0);

    gs::core::SogWriteOptions options_old;
    options_old.output_path = temp_dir.path() / "old.sog";
    options_old.iterations = 10;

    gs::core::SogWriteOptionsNew options_new;
    options_new.output_path = temp_dir.path() / "new.sog";
    options_new.iterations = 10;

    auto result_old = gs::core::write_sog(splat_data, options_old);
    auto result_new = gs::core::write_sog_new(splat_data, options_new);
    ASSERT_TRUE(result_old && result_new);

    auto meta_old = extract_and_load_meta_from_sog(temp_dir.path() / "old.sog");
    auto meta_new = extract_and_load_meta_from_sog(temp_dir.path() / "new.sog");

    EXPECT_EQ(meta_old["scales"]["codebook"].size(), 256);
    EXPECT_EQ(meta_new["scales"]["codebook"].size(), 256);
    EXPECT_EQ(meta_old["sh0"]["codebook"].size(), 256);
    EXPECT_EQ(meta_new["sh0"]["codebook"].size(), 256);
}
