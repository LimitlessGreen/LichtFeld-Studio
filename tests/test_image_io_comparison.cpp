/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/image_io.hpp"
#include "core/image_io_new.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-4f;
    constexpr float IMAGE_TOLERANCE = 2.0f / 255.0f; // Allow 2 uint8 levels of difference

    // ============================================================================
    // Helper Functions
    // ============================================================================

    torch::Tensor create_test_image_torch(int h, int w, int c) {
        auto img = torch::rand({h, w, c}, torch::kCPU);
        return img;
    }

    gs::Tensor create_test_image_tensor(int h, int w, int c) {
        auto img = gs::Tensor::rand({static_cast<size_t>(h),
                                     static_cast<size_t>(w),
                                     static_cast<size_t>(c)},
                                    gs::Device::CPU);
        return img;
    }

    torch::Tensor create_gradient_image_torch(int h, int w, int c) {
        auto img = torch::zeros({h, w, c}, torch::kCPU);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                float val = static_cast<float>(i + j) / (h + w);
                for (int k = 0; k < c; ++k) {
                    img[i][j][k] = val;
                }
            }
        }
        return img;
    }

    gs::Tensor create_gradient_image_tensor(int h, int w, int c) {
        auto img = gs::Tensor::zeros({static_cast<size_t>(h),
                                      static_cast<size_t>(w),
                                      static_cast<size_t>(c)},
                                     gs::Device::CPU);

        auto img_acc = img.accessor<float, 3>();
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                float val = static_cast<float>(i + j) / (h + w);
                for (int k = 0; k < c; ++k) {
                    img_acc(i, j, k) = val;
                }
            }
        }
        return img;
    }

    gs::Tensor torch_to_tensor(const torch::Tensor& torch_tensor) {
        auto cpu_tensor = torch_tensor.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < torch_tensor.dim(); ++i) {
            shape.push_back(torch_tensor.size(i));
        }

        if (torch_tensor.scalar_type() == torch::kFloat32) {
            std::vector<float> data(cpu_tensor.data_ptr<float>(),
                                    cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
            return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CPU);
        }

        return gs::Tensor();
    }

    torch::Tensor tensor_to_torch(const gs::Tensor& gs_tensor) {
        auto cpu_tensor = gs_tensor.cpu();
        std::vector<int64_t> shape;
        for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
            shape.push_back(cpu_tensor.shape()[i]);
        }

        if (gs_tensor.dtype() == gs::DataType::Float32) {
            auto data = cpu_tensor.to_vector();
            auto torch_tensor = torch::from_blob(data.data(), shape, torch::kFloat32).clone();
            return torch_tensor;
        }

        return torch::Tensor();
    }

    bool images_are_close(const torch::Tensor& torch_img,
                          const gs::Tensor& gs_img,
                          float tolerance = IMAGE_TOLERANCE) {
        if (torch_img.dim() != static_cast<int64_t>(gs_img.ndim())) {
            return false;
        }

        for (int i = 0; i < torch_img.dim(); ++i) {
            if (torch_img.size(i) != static_cast<int64_t>(gs_img.shape()[i])) {
                return false;
            }
        }

        auto torch_cpu = torch_img.cpu().contiguous();
        auto gs_cpu = gs_img.cpu();

        if (torch_img.scalar_type() != torch::kFloat32 ||
            gs_img.dtype() != gs::DataType::Float32) {
            return false;
        }

        auto torch_data = torch_cpu.data_ptr<float>();
        auto gs_data = gs_cpu.to_vector();

        if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
            return false;
        }

        float max_diff = 0.0f;
        int64_t diff_count = 0;

        for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
            float diff = std::abs(torch_data[i] - gs_data[i]);
            if (diff > tolerance) {
                diff_count++;
                max_diff = std::max(max_diff, diff);
            }
        }

        // Allow up to 1% of pixels to be slightly off (due to rounding differences)
        float diff_percentage = static_cast<float>(diff_count) / torch_cpu.numel() * 100.0f;

        if (diff_percentage > 1.0f) {
            std::cerr << "Images differ: " << diff_count << " pixels ("
                      << diff_percentage << "%), max diff: " << max_diff << "\n";
            return false;
        }

        return true;
    }

    bool file_exists(const fs::path& path) {
        return fs::exists(path) && fs::is_regular_file(path);
    }

    size_t get_file_size(const fs::path& path) {
        if (!file_exists(path))
            return 0;
        return fs::file_size(path);
    }

    void cleanup_test_file(const fs::path& path) {
        if (file_exists(path)) {
            fs::remove(path);
        }
    }

    class TempTestDirectory {
    public:
        TempTestDirectory() {
            path_ = fs::temp_directory_path() / "image_io_test";
            fs::create_directories(path_);
        }

        ~TempTestDirectory() {
            if (fs::exists(path_)) {
                fs::remove_all(path_);
            }
        }

        fs::path get_path() const { return path_; }

        fs::path make_path(const std::string& filename) const {
            return path_ / filename;
        }

    private:
        fs::path path_;
    };

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class ImageIOComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        gs::Tensor::manual_seed(42);
        test_dir_ = std::make_unique<TempTestDirectory>();
    }

    void TearDown() override {
        test_dir_.reset();
    }

    std::unique_ptr<TempTestDirectory> test_dir_;
};

// ============================================================================
// Basic Save/Load Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, SaveAndLoadSimpleImage_PNG) {
    const int h = 100, w = 100, c = 3;

    auto img_torch = create_gradient_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("test_old.png");
    auto path_new = test_dir_->make_path("test_new.png");

    // Save with both implementations
    save_image(path_old, img_torch);
    gs::image_io::save_image(path_new, img_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));

    // Load back with old implementation
    auto [data_old, w_old, h_old, c_old] = load_image(path_old);
    auto [data_new, w_new, h_new, c_new] = load_image(path_new);

    EXPECT_EQ(w_old, w);
    EXPECT_EQ(h_old, h);
    EXPECT_EQ(c_old, 3);

    EXPECT_EQ(w_new, w);
    EXPECT_EQ(h_new, h);
    EXPECT_EQ(c_new, 3);

    // Compare pixel data
    for (int i = 0; i < h * w * 3; ++i) {
        EXPECT_NEAR(data_old[i], data_new[i], 2);
    }

    free_image(data_old);
    free_image(data_new);
}

TEST_F(ImageIOComparisonTest, SaveAndLoadSimpleImage_JPG) {
    const int h = 100, w = 100, c = 3;

    auto img_torch = create_gradient_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("test_old.jpg");
    auto path_new = test_dir_->make_path("test_new.jpg");

    save_image(path_old, img_torch);
    gs::image_io::save_image(path_new, img_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));

    // JPG is lossy, so just check that files are reasonable size
    EXPECT_GT(get_file_size(path_old), 1000);
    EXPECT_GT(get_file_size(path_new), 1000);
}

TEST_F(ImageIOComparisonTest, SaveGrayscaleImage) {
    const int h = 50, w = 50, c = 1;

    auto img_torch = create_test_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("gray_old.png");
    auto path_new = test_dir_->make_path("gray_new.png");

    save_image(path_old, img_torch);
    gs::image_io::save_image(path_new, img_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));

    auto [data_old, w_old, h_old, c_old] = load_image(path_old);
    auto [data_new, w_new, h_new, c_new] = load_image(path_new);

    EXPECT_EQ(w_old, w);
    EXPECT_EQ(h_old, h);
    EXPECT_EQ(w_new, w);
    EXPECT_EQ(h_new, h);

    free_image(data_old);
    free_image(data_new);
}

TEST_F(ImageIOComparisonTest, SaveRGBAImage) {
    const int h = 50, w = 50, c = 4;

    auto img_torch = create_test_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("rgba_old.png");
    auto path_new = test_dir_->make_path("rgba_new.png");

    save_image(path_old, img_torch);
    gs::image_io::save_image(path_new, img_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));
}

// ============================================================================
// Layout Conversion Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, SaveCHWLayout) {
    const int h = 64, w = 64, c = 3;

    // Create [C, H, W] layout
    auto img_torch = create_test_image_torch(h, w, c).permute({2, 0, 1}).contiguous();
    auto img_tensor_hwc = create_test_image_tensor(h, w, c);
    auto img_tensor = img_tensor_hwc.permute({2, 0, 1}).contiguous();

    auto path_old = test_dir_->make_path("chw_old.png");
    auto path_new = test_dir_->make_path("chw_new.png");

    save_image(path_old, img_torch);
    gs::image_io::save_image(path_new, img_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));
}

TEST_F(ImageIOComparisonTest, SaveBatchedImage) {
    const int h = 64, w = 64, c = 3;

    // Create [1, C, H, W] layout
    auto img_torch = create_test_image_torch(h, w, c)
                         .permute({2, 0, 1})
                         .unsqueeze(0)
                         .contiguous();
    auto img_tensor_hwc = create_test_image_tensor(h, w, c);
    auto img_tensor = img_tensor_hwc.permute({2, 0, 1}).unsqueeze(0).contiguous();

    auto path_old = test_dir_->make_path("batch_old.png");
    auto path_new = test_dir_->make_path("batch_new.png");

    save_image(path_old, img_torch);
    gs::image_io::save_image(path_new, img_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));
}

// ============================================================================
// Multiple Image Concatenation Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, ConcatenateImagesHorizontal) {
    const int h = 50, w = 50, c = 3;

    std::vector<torch::Tensor> images_torch;
    std::vector<gs::Tensor> images_tensor;

    for (int i = 0; i < 3; ++i) {
        auto img_t = create_test_image_torch(h, w, c);
        images_torch.push_back(img_t);
        images_tensor.push_back(torch_to_tensor(img_t));
    }

    auto path_old = test_dir_->make_path("concat_h_old.png");
    auto path_new = test_dir_->make_path("concat_h_new.png");

    save_image(path_old, images_torch, true, 2);
    gs::image_io::save_image(path_new, images_tensor, true, 2);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));

    auto [data_old, w_old, h_old, c_old] = load_image(path_old);
    auto [data_new, w_new, h_new, c_new] = load_image(path_new);

    // Width should be 3*w + 2*separator_width
    EXPECT_EQ(w_old, 3 * w + 2 * 2);
    EXPECT_EQ(w_new, 3 * w + 2 * 2);
    EXPECT_EQ(h_old, h);
    EXPECT_EQ(h_new, h);

    free_image(data_old);
    free_image(data_new);
}

TEST_F(ImageIOComparisonTest, ConcatenateImagesVertical) {
    const int h = 50, w = 50, c = 3;

    std::vector<torch::Tensor> images_torch;
    std::vector<gs::Tensor> images_tensor;

    for (int i = 0; i < 3; ++i) {
        auto img_t = create_test_image_torch(h, w, c);
        images_torch.push_back(img_t);
        images_tensor.push_back(torch_to_tensor(img_t));
    }

    auto path_old = test_dir_->make_path("concat_v_old.png");
    auto path_new = test_dir_->make_path("concat_v_new.png");

    save_image(path_old, images_torch, false, 2);
    gs::image_io::save_image(path_new, images_tensor, false, 2);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));

    auto [data_old, w_old, h_old, c_old] = load_image(path_old);
    auto [data_new, w_new, h_new, c_new] = load_image(path_new);

    // Height should be 3*h + 2*separator_width
    EXPECT_EQ(w_old, w);
    EXPECT_EQ(w_new, w);
    EXPECT_EQ(h_old, 3 * h + 2 * 2);
    EXPECT_EQ(h_new, 3 * h + 2 * 2);

    free_image(data_old);
    free_image(data_new);
}

TEST_F(ImageIOComparisonTest, ConcatenateWithoutSeparator) {
    const int h = 40, w = 40, c = 3;

    std::vector<torch::Tensor> images_torch;
    std::vector<gs::Tensor> images_tensor;

    for (int i = 0; i < 2; ++i) {
        auto img_t = create_test_image_torch(h, w, c);
        images_torch.push_back(img_t);
        images_tensor.push_back(torch_to_tensor(img_t));
    }

    auto path_old = test_dir_->make_path("concat_nosep_old.png");
    auto path_new = test_dir_->make_path("concat_nosep_new.png");

    save_image(path_old, images_torch, true, 0);
    gs::image_io::save_image(path_new, images_tensor, true, 0);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));

    auto [data_old, w_old, h_old, c_old] = load_image(path_old);
    auto [data_new, w_new, h_new, c_new] = load_image(path_new);

    EXPECT_EQ(w_old, 2 * w);
    EXPECT_EQ(w_new, 2 * w);

    free_image(data_old);
    free_image(data_new);
}

// ============================================================================
// Image Info Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, GetImageInfo) {
    const int h = 123, w = 456, c = 3;

    auto img_tensor = create_test_image_tensor(h, w, c);
    auto path = test_dir_->make_path("info_test.png");

    gs::image_io::save_image(path, img_tensor);

    auto [info_w, info_h, info_c] = get_image_info(path);
    auto [new_w, new_h, new_c] = gs::image_io::get_image_info(path);

    EXPECT_EQ(info_w, w);
    EXPECT_EQ(info_h, h);
    EXPECT_EQ(info_c, c);

    EXPECT_EQ(new_w, w);
    EXPECT_EQ(new_h, h);
    EXPECT_EQ(new_c, c);
}

// ============================================================================
// Load with Resize Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, LoadWithResize2x) {
    const int h = 128, w = 128, c = 3;

    auto img_tensor = create_test_image_tensor(h, w, c);
    auto path = test_dir_->make_path("resize2x.png");

    gs::image_io::save_image(path, img_tensor);

    auto [data_old, w_old, h_old, c_old] = load_image(path, 2);
    auto [data_new, w_new, h_new, c_new] = gs::image_io::load_image(path, 2);

    EXPECT_EQ(w_old, w / 2);
    EXPECT_EQ(h_old, h / 2);
    EXPECT_EQ(w_new, w / 2);
    EXPECT_EQ(h_new, h / 2);

    free_image(data_old);
    free_image(data_new);
}

TEST_F(ImageIOComparisonTest, LoadWithResize4x) {
    const int h = 256, w = 256, c = 3;

    auto img_tensor = create_test_image_tensor(h, w, c);
    auto path = test_dir_->make_path("resize4x.png");

    gs::image_io::save_image(path, img_tensor);

    auto [data_old, w_old, h_old, c_old] = load_image(path, 4);
    auto [data_new, w_new, h_new, c_new] = gs::image_io::load_image(path, 4);

    EXPECT_EQ(w_old, w / 4);
    EXPECT_EQ(h_old, h / 4);
    EXPECT_EQ(w_new, w / 4);
    EXPECT_EQ(h_new, h / 4);

    free_image(data_old);
    free_image(data_new);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ImageIOComparisonTest, VerySmallImage) {
    const int h = 2, w = 2, c = 3;

    auto img_torch = create_test_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("tiny_old.png");
    auto path_new = test_dir_->make_path("tiny_new.png");

    EXPECT_NO_THROW(save_image(path_old, img_torch));
    EXPECT_NO_THROW(gs::image_io::save_image(path_new, img_tensor));

    EXPECT_TRUE(file_exists(path_old));
    EXPECT_TRUE(file_exists(path_new));
}

TEST_F(ImageIOComparisonTest, LargeImage) {
    const int h = 1024, w = 1024, c = 3;

    auto img_torch = create_test_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("large_old.png");
    auto path_new = test_dir_->make_path("large_new.png");

    EXPECT_NO_THROW(save_image(path_old, img_torch));
    EXPECT_NO_THROW(gs::image_io::save_image(path_new, img_tensor));

    EXPECT_TRUE(file_exists(path_old));
    EXPECT_TRUE(file_exists(path_new));
}

TEST_F(ImageIOComparisonTest, SaveSingleImageList) {
    const int h = 50, w = 50, c = 3;

    auto img_t = create_test_image_torch(h, w, c);
    std::vector<torch::Tensor> images_torch = {img_t};
    std::vector<gs::Tensor> images_tensor = {torch_to_tensor(img_t)};

    auto path_old = test_dir_->make_path("single_list_old.png");
    auto path_new = test_dir_->make_path("single_list_new.png");

    save_image(path_old, images_torch);
    gs::image_io::save_image(path_new, images_tensor);

    ASSERT_TRUE(file_exists(path_old));
    ASSERT_TRUE(file_exists(path_new));
}

TEST_F(ImageIOComparisonTest, ClampingOutOfRange) {
    const int h = 50, w = 50, c = 3;

    // Create image with values outside [0, 1]
    auto img_torch = torch::randn({h, w, c}, torch::kCPU) * 2.0f;
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("clamp_old.png");
    auto path_new = test_dir_->make_path("clamp_new.png");

    EXPECT_NO_THROW(save_image(path_old, img_torch));
    EXPECT_NO_THROW(gs::image_io::save_image(path_new, img_tensor));

    EXPECT_TRUE(file_exists(path_old));
    EXPECT_TRUE(file_exists(path_new));
}

// ============================================================================
// Batch Saver Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, BatchSaverSingleImage) {
    const int h = 50, w = 50, c = 3;

    auto img_torch = create_test_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    auto path_old = test_dir_->make_path("batch_single_old.png");
    auto path_new = test_dir_->make_path("batch_single_new.png");

    image_io::save_image_async(path_old, img_torch);
    gs::image_io::save_image_async(path_new, img_tensor);

    image_io::wait_for_pending_saves();
    gs::image_io::wait_for_pending_saves();

    EXPECT_TRUE(file_exists(path_old));
    EXPECT_TRUE(file_exists(path_new));
}

TEST_F(ImageIOComparisonTest, BatchSaverMultipleImages) {
    const int h = 40, w = 40, c = 3;
    const int num_images = 10;

    for (int i = 0; i < num_images; ++i) {
        auto img_torch = create_test_image_torch(h, w, c);
        auto img_tensor = torch_to_tensor(img_torch);

        auto path_old = test_dir_->make_path("batch_" + std::to_string(i) + "_old.png");
        auto path_new = test_dir_->make_path("batch_" + std::to_string(i) + "_new.png");

        image_io::save_image_async(path_old, img_torch);
        gs::image_io::save_image_async(path_new, img_tensor);
    }

    image_io::wait_for_pending_saves();
    gs::image_io::wait_for_pending_saves();

    for (int i = 0; i < num_images; ++i) {
        auto path_old = test_dir_->make_path("batch_" + std::to_string(i) + "_old.png");
        auto path_new = test_dir_->make_path("batch_" + std::to_string(i) + "_new.png");

        EXPECT_TRUE(file_exists(path_old)) << "Old image " << i << " not saved";
        EXPECT_TRUE(file_exists(path_new)) << "New image " << i << " not saved";
    }
}

TEST_F(ImageIOComparisonTest, BatchSaverPendingCount) {
    auto& saver_old = image_io::BatchImageSaver::instance();
    auto& saver_new = gs::image_io::BatchImageSaver::instance();

    const int h = 30, w = 30, c = 3;

    auto img_torch = create_test_image_torch(h, w, c);
    auto img_tensor = torch_to_tensor(img_torch);

    // Queue multiple saves
    for (int i = 0; i < 5; ++i) {
        saver_old.queue_save(test_dir_->make_path("pending_old_" + std::to_string(i) + ".png"),
                             img_torch);
        saver_new.queue_save(test_dir_->make_path("pending_new_" + std::to_string(i) + ".png"),
                             img_tensor);
    }

    // Check pending count (may be 0 if processing is fast)
    EXPECT_GE(saver_old.pending_count(), 0);
    EXPECT_GE(saver_new.pending_count(), 0);

    saver_old.wait_all();
    saver_new.wait_all();

    EXPECT_EQ(saver_old.pending_count(), 0);
    EXPECT_EQ(saver_new.pending_count(), 0);
}

TEST_F(ImageIOComparisonTest, BatchSaverDisabled) {
    auto& saver_new = gs::image_io::BatchImageSaver::instance();

    saver_new.set_enabled(false);
    EXPECT_FALSE(saver_new.is_enabled());

    const int h = 30, w = 30, c = 3;
    auto img_tensor = create_test_image_tensor(h, w, c);
    auto path = test_dir_->make_path("disabled_async.png");

    // Should save synchronously
    saver_new.queue_save(path, img_tensor);

    // Should be immediately available
    EXPECT_TRUE(file_exists(path));

    saver_new.set_enabled(true);
    EXPECT_TRUE(saver_new.is_enabled());
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

class ImageIOPerformanceTest : public ::testing::Test {
protected:
    struct BenchmarkResult {
        std::string name;
        int width;
        int height;
        int channels;
        double time_ms_old;
        double time_ms_new;
        double speedup;
    };

    void SetUp() override {
        torch::manual_seed(42);
        gs::Tensor::manual_seed(42);
        test_dir_ = std::make_unique<TempTestDirectory>();
    }

    void TearDown() override {
        test_dir_.reset();
    }

    template <typename Func>
    double benchmark(Func func, int warmup_runs = 2, int timing_runs = 5) {
        // Warm-up
        for (int i = 0; i < warmup_runs; ++i) {
            func();
        }

        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timing_runs; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (1000.0 * timing_runs);
    }

    void print_results(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n"
                  << std::string(100, '=') << "\n";
        std::cout << "IMAGE I/O PERFORMANCE COMPARISON: image_io.cpp vs image_io_new.cpp\n";
        std::cout << std::string(100, '=') << "\n\n";

        std::cout << std::left << std::setw(30) << "Benchmark"
                  << std::right << std::setw(8) << "Width"
                  << std::right << std::setw(8) << "Height"
                  << std::right << std::setw(6) << "Ch"
                  << std::right << std::setw(12) << "Old (ms)"
                  << std::right << std::setw(12) << "New (ms)"
                  << std::right << std::setw(12) << "Speedup"
                  << "\n";
        std::cout << std::string(100, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(30) << r.name
                      << std::right << std::setw(8) << r.width
                      << std::right << std::setw(8) << r.height
                      << std::right << std::setw(6) << r.channels
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms_old
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms_new
                      << std::right << std::setw(11) << std::fixed << std::setprecision(2) << r.speedup << "x"
                      << "\n";
        }

        std::cout << std::string(100, '=') << "\n";

        // Summary
        double avg_speedup = 0.0;
        for (const auto& r : results) {
            avg_speedup += r.speedup;
        }
        avg_speedup /= results.size();

        std::cout << "\nSUMMARY:\n";
        std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x\n";
        std::cout << std::string(100, '=') << "\n\n";
    }

    std::unique_ptr<TempTestDirectory> test_dir_;
};

TEST_F(ImageIOPerformanceTest, DISABLED_SavePerformance) {
    std::vector<BenchmarkResult> results;

    struct Config {
        std::string name;
        int width;
        int height;
        int channels;
    };

    std::vector<Config> configs = {
        {"Small_RGB", 256, 256, 3},
        {"Medium_RGB", 512, 512, 3},
        {"Large_RGB", 1024, 1024, 3},
        {"HD_RGB", 1920, 1080, 3},
        {"Small_RGBA", 256, 256, 4},
        {"Medium_RGBA", 512, 512, 4},
        {"Grayscale", 512, 512, 1},
    };

    for (const auto& cfg : configs) {
        auto img_torch = create_test_image_torch(cfg.height, cfg.width, cfg.channels);
        auto img_tensor = torch_to_tensor(img_torch);

        BenchmarkResult result;
        result.name = cfg.name;
        result.width = cfg.width;
        result.height = cfg.height;
        result.channels = cfg.channels;

        auto path_old = test_dir_->make_path("perf_old.png");
        auto path_new = test_dir_->make_path("perf_new.png");

        result.time_ms_old = benchmark([&]() {
            save_image(path_old, img_torch);
        });

        result.time_ms_new = benchmark([&]() {
            gs::image_io::save_image(path_new, img_tensor);
        });

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);

        cleanup_test_file(path_old);
        cleanup_test_file(path_new);
    }

    print_results(results);
}

TEST_F(ImageIOPerformanceTest, DISABLED_LoadPerformance) {
    std::vector<BenchmarkResult> results;

    struct Config {
        std::string name;
        int width;
        int height;
        int channels;
    };

    std::vector<Config> configs = {
        {"Small_RGB", 256, 256, 3},
        {"Medium_RGB", 512, 512, 3},
        {"Large_RGB", 1024, 1024, 3},
        {"HD_RGB", 1920, 1080, 3},
    };

    for (const auto& cfg : configs) {
        // Create and save test image
        auto img_tensor = create_test_image_tensor(cfg.height, cfg.width, cfg.channels);
        auto path = test_dir_->make_path("load_perf.png");
        gs::image_io::save_image(path, img_tensor);

        BenchmarkResult result;
        result.name = cfg.name;
        result.width = cfg.width;
        result.height = cfg.height;
        result.channels = cfg.channels;

        result.time_ms_old = benchmark([&]() {
            auto [data, w, h, c] = load_image(path);
            free_image(data);
        });

        result.time_ms_new = benchmark([&]() {
            auto [data, w, h, c] = gs::image_io::load_image(path);
            gs::image_io::free_image(data);
        });

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);

        cleanup_test_file(path);
    }

    print_results(results);
}

TEST_F(ImageIOPerformanceTest, DISABLED_ConcatenationPerformance) {
    std::vector<BenchmarkResult> results;

    struct Config {
        std::string name;
        int num_images;
        int width;
        int height;
    };

    std::vector<Config> configs = {
        {"Small_2Images", 2, 256, 256},
        {"Small_5Images", 5, 256, 256},
        {"Medium_2Images", 2, 512, 512},
        {"Medium_5Images", 5, 512, 512},
        {"Large_2Images", 2, 1024, 1024},
    };

    for (const auto& cfg : configs) {
        std::vector<torch::Tensor> images_torch;
        std::vector<gs::Tensor> images_tensor;

        for (int i = 0; i < cfg.num_images; ++i) {
            auto img_t = create_test_image_torch(cfg.height, cfg.width, 3);
            images_torch.push_back(img_t);
            images_tensor.push_back(torch_to_tensor(img_t));
        }

        BenchmarkResult result;
        result.name = cfg.name;
        result.width = cfg.width;
        result.height = cfg.height;
        result.channels = 3;

        auto path_old = test_dir_->make_path("concat_perf_old.png");
        auto path_new = test_dir_->make_path("concat_perf_new.png");

        result.time_ms_old = benchmark([&]() {
            save_image(path_old, images_torch, true, 2);
        },
                                       1, 3);

        result.time_ms_new = benchmark([&]() {
            gs::image_io::save_image(path_new, images_tensor, true, 2);
        },
                                       1, 3);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);

        cleanup_test_file(path_old);
        cleanup_test_file(path_new);
    }

    print_results(results);
}

TEST_F(ImageIOPerformanceTest, DISABLED_BatchSavePerformance) {
    std::vector<BenchmarkResult> results;

    struct Config {
        std::string name;
        int num_images;
        int width;
        int height;
    };

    std::vector<Config> configs = {
        {"Batch_10_Small", 10, 256, 256},
        {"Batch_50_Small", 50, 256, 256},
        {"Batch_10_Medium", 10, 512, 512},
        {"Batch_20_Medium", 20, 512, 512},
    };

    for (const auto& cfg : configs) {
        BenchmarkResult result;
        result.name = cfg.name;
        result.width = cfg.width;
        result.height = cfg.height;
        result.channels = 3;

        // Old implementation
        result.time_ms_old = benchmark([&]() {
            for (int i = 0; i < cfg.num_images; ++i) {
                auto img = create_test_image_torch(cfg.height, cfg.width, 3);
                auto path = test_dir_->make_path("batch_old_" + std::to_string(i) + ".png");
                image_io::save_image_async(path, img);
            }
            image_io::wait_for_pending_saves();
        },
                                       1, 2);

        // New implementation
        result.time_ms_new = benchmark([&]() {
            for (int i = 0; i < cfg.num_images; ++i) {
                auto img = create_test_image_tensor(cfg.height, cfg.width, 3);
                auto path = test_dir_->make_path("batch_new_" + std::to_string(i) + ".png");
                gs::image_io::save_image_async(path, img);
            }
            gs::image_io::wait_for_pending_saves();
        },
                                       1, 2);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);

        // Cleanup
        for (int i = 0; i < cfg.num_images; ++i) {
            cleanup_test_file(test_dir_->make_path("batch_old_" + std::to_string(i) + ".png"));
            cleanup_test_file(test_dir_->make_path("batch_new_" + std::to_string(i) + ".png"));
        }
    }

    print_results(results);
}

TEST_F(ImageIOPerformanceTest, DISABLED_ResizeLoadPerformance) {
    std::vector<BenchmarkResult> results;

    struct Config {
        std::string name;
        int width;
        int height;
        int resize_factor;
    };

    std::vector<Config> configs = {
        {"Medium_2x", 1024, 1024, 2},
        {"Medium_4x", 1024, 1024, 4},
        {"Large_2x", 2048, 2048, 2},
        {"Large_4x", 2048, 2048, 4},
        {"HD_2x", 1920, 1080, 2},
    };

    for (const auto& cfg : configs) {
        // Create and save test image
        auto img_tensor = create_test_image_tensor(cfg.height, cfg.width, 3);
        auto path = test_dir_->make_path("resize_perf.png");
        gs::image_io::save_image(path, img_tensor);

        BenchmarkResult result;
        result.name = cfg.name;
        result.width = cfg.width / cfg.resize_factor;
        result.height = cfg.height / cfg.resize_factor;
        result.channels = 3;

        result.time_ms_old = benchmark([&]() {
            auto [data, w, h, c] = load_image(path, cfg.resize_factor);
            free_image(data);
        });

        result.time_ms_new = benchmark([&]() {
            auto [data, w, h, c] = gs::image_io::load_image(path, cfg.resize_factor);
            gs::image_io::free_image(data);
        });

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);

        cleanup_test_file(path);
    }

    print_results(results);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, ManySmallImagesSave) {
    const int num_images = 50;
    const int h = 64, w = 64, c = 3;

    for (int i = 0; i < num_images; ++i) {
        auto img_torch = create_test_image_torch(h, w, c);
        auto img_tensor = torch_to_tensor(img_torch);

        auto path_old = test_dir_->make_path("many_old_" + std::to_string(i) + ".png");
        auto path_new = test_dir_->make_path("many_new_" + std::to_string(i) + ".png");

        EXPECT_NO_THROW(save_image(path_old, img_torch));
        EXPECT_NO_THROW(gs::image_io::save_image(path_new, img_tensor));
    }

    // Verify all saved
    for (int i = 0; i < num_images; ++i) {
        auto path_old = test_dir_->make_path("many_old_" + std::to_string(i) + ".png");
        auto path_new = test_dir_->make_path("many_new_" + std::to_string(i) + ".png");

        EXPECT_TRUE(file_exists(path_old));
        EXPECT_TRUE(file_exists(path_new));
    }
}

TEST_F(ImageIOComparisonTest, SequentialSaveAndLoad) {
    const int num_iterations = 20;
    const int h = 128, w = 128, c = 3;

    auto img_tensor = create_test_image_tensor(h, w, c);

    for (int i = 0; i < num_iterations; ++i) {
        auto path = test_dir_->make_path("sequential_" + std::to_string(i) + ".png");

        // Save
        EXPECT_NO_THROW(gs::image_io::save_image(path, img_tensor));

        // Load
        auto [data, w_loaded, h_loaded, c_loaded] = gs::image_io::load_image(path);
        EXPECT_EQ(w_loaded, w);
        EXPECT_EQ(h_loaded, h);
        EXPECT_EQ(c_loaded, c);
        gs::image_io::free_image(data);

        cleanup_test_file(path);
    }
}

TEST_F(ImageIOComparisonTest, DifferentFormats) {
    const int h = 100, w = 100, c = 3;
    auto img_tensor = create_test_image_tensor(h, w, c);

    std::vector<std::string> formats = {".png", ".jpg", ".bmp", ".tga"};

    for (const auto& ext : formats) {
        auto path = test_dir_->make_path("format_test" + ext);

        EXPECT_NO_THROW(gs::image_io::save_image(path, img_tensor))
            << "Failed to save format: " << ext;

        if (file_exists(path)) {
            auto [data, w_loaded, h_loaded, c_loaded] = gs::image_io::load_image(path);
            EXPECT_GT(w_loaded, 0) << "Format: " << ext;
            EXPECT_GT(h_loaded, 0) << "Format: " << ext;
            gs::image_io::free_image(data);
        }
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, InvalidPath) {
    const int h = 50, w = 50, c = 3;
    auto img_tensor = create_test_image_tensor(h, w, c);

    auto invalid_path = fs::path("/nonexistent/directory/image.png");

    EXPECT_THROW(gs::image_io::save_image(invalid_path, img_tensor), std::runtime_error);
}

TEST_F(ImageIOComparisonTest, LoadNonexistentFile) {
    auto nonexistent_path = test_dir_->make_path("does_not_exist.png");

    EXPECT_THROW(gs::image_io::load_image(nonexistent_path), std::runtime_error);
    EXPECT_THROW(gs::image_io::get_image_info(nonexistent_path), std::runtime_error);
}

TEST_F(ImageIOComparisonTest, SaveEmptyTensor) {
    auto empty_tensor = gs::Tensor::empty({0, 0, 3}, gs::Device::CPU);
    auto path = test_dir_->make_path("empty.png");

    EXPECT_THROW(gs::image_io::save_image(path, empty_tensor), std::runtime_error);
}

TEST_F(ImageIOComparisonTest, SaveInvalidDimensionTensor) {
    // Create 1D tensor (invalid for image)
    auto invalid_tensor = gs::Tensor::rand({100}, gs::Device::CPU);
    auto path = test_dir_->make_path("invalid.png");

    EXPECT_THROW(gs::image_io::save_image(path, invalid_tensor), std::runtime_error);
}

TEST_F(ImageIOComparisonTest, SaveTooManyChannels) {
    const int h = 50, w = 50;
    // Create tensor with 5 channels (invalid)
    auto invalid_tensor = gs::Tensor::rand({static_cast<size_t>(h),
                                            static_cast<size_t>(w),
                                            5},
                                           gs::Device::CPU);
    auto path = test_dir_->make_path("too_many_channels.png");

    EXPECT_THROW(gs::image_io::save_image(path, invalid_tensor), std::runtime_error);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(ImageIOComparisonTest, RoundTripPreservesData) {
    const int h = 128, w = 128, c = 3;

    auto original = create_gradient_image_tensor(h, w, c);
    auto path = test_dir_->make_path("roundtrip.png");

    // Save
    gs::image_io::save_image(path, original);

    // Load back
    auto [data, w_loaded, h_loaded, c_loaded] = gs::image_io::load_image(path);

    EXPECT_EQ(w_loaded, w);
    EXPECT_EQ(h_loaded, h);
    EXPECT_EQ(c_loaded, c);

    // Convert loaded data to tensor for comparison
    std::vector<float> loaded_float(w * h * c);
    for (int i = 0; i < w * h * c; ++i) {
        loaded_float[i] = static_cast<float>(data[i]) / 255.0f;
    }

    auto loaded_tensor = gs::Tensor::from_vector(
        loaded_float,
        gs::TensorShape({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(c)}),
        gs::Device::CPU);

    // Compare (allowing for uint8 quantization error)
    auto original_cpu = original.cpu();
    auto loaded_cpu = loaded_tensor.cpu();

    int mismatch_count = 0;
    float max_diff = 0.0f;
    const float threshold = 2.0f / 255.0f; // 2 uint8 levels

    auto orig_acc = original_cpu.accessor<float, 3>();
    auto load_acc = loaded_cpu.accessor<float, 3>();

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int k = 0; k < c; ++k) {
                float diff = std::abs(orig_acc(i, j, k) - load_acc(i, j, k));
                if (diff > threshold) {
                    mismatch_count++;
                    max_diff = std::max(max_diff, diff);
                }
            }
        }
    }

    // Allow up to 1% of pixels to have small errors
    float error_rate = static_cast<float>(mismatch_count) / (h * w * c) * 100.0f;
    EXPECT_LT(error_rate, 1.0f) << "Error rate: " << error_rate << "%, max diff: " << max_diff;

    gs::image_io::free_image(data);
}

TEST_F(ImageIOComparisonTest, ConcurrentAsyncSaves) {
    const int num_images = 20;
    const int h = 100, w = 100, c = 3;

    auto& saver = gs::image_io::BatchImageSaver::instance();

    std::vector<gs::Tensor> images;
    for (int i = 0; i < num_images; ++i) {
        images.push_back(create_test_image_tensor(h, w, c));
    }

    // Queue all saves
    for (int i = 0; i < num_images; ++i) {
        auto path = test_dir_->make_path("concurrent_" + std::to_string(i) + ".png");
        saver.queue_save(path, images[i]);
    }

    // Wait for completion
    saver.wait_all();

    // Verify all saved correctly
    for (int i = 0; i < num_images; ++i) {
        auto path = test_dir_->make_path("concurrent_" + std::to_string(i) + ".png");
        EXPECT_TRUE(file_exists(path)) << "Image " << i << " not saved";
        EXPECT_GT(get_file_size(path), 0) << "Image " << i << " has zero size";
    }
}

TEST_F(ImageIOComparisonTest, MixedAsyncAndSyncSaves) {
    const int h = 64, w = 64, c = 3;

    auto img1 = create_test_image_tensor(h, w, c);
    auto img2 = create_test_image_tensor(h, w, c);

    auto path_async = test_dir_->make_path("mixed_async.png");
    auto path_sync = test_dir_->make_path("mixed_sync.png");

    // Queue async save
    gs::image_io::save_image_async(path_async, img1);

    // Do sync save
    gs::image_io::save_image(path_sync, img2);

    // Wait for async
    gs::image_io::wait_for_pending_saves();

    EXPECT_TRUE(file_exists(path_async));
    EXPECT_TRUE(file_exists(path_sync));
}
