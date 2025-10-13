/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io_new.hpp"
#include "core/logger.hpp"

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <condition_variable>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// Run once: set global OIIO attributes (threading, etc.)
std::once_flag g_oiio_once;
inline void init_oiio() {
    std::call_once(g_oiio_once, [] {
        int n = (int)std::max(1u, std::thread::hardware_concurrency());
        OIIO::attribute("threads", n);
    });
}

// Downscale (resample) to (nw, nh). Returns newly malloc'd RGB buffer.
static inline unsigned char* downscale_resample_direct(const unsigned char* src_rgb,
                                                       int w, int h, int nw, int nh,
                                                       int nthreads /* 0=auto, 1=single */) {
    // Allocate destination first
    size_t outbytes = (size_t)nw * nh * 3;
    auto* out = static_cast<unsigned char*>(std::malloc(outbytes));
    if (!out)
        throw std::bad_alloc();

    // Wrap src & dst without extra allocations/copies
    OIIO::ImageBuf srcbuf(OIIO::ImageSpec(w, h, 3, OIIO::TypeDesc::UINT8),
                          const_cast<unsigned char*>(src_rgb));
    OIIO::ImageBuf dstbuf(OIIO::ImageSpec(nw, nh, 3, OIIO::TypeDesc::UINT8), out);

    OIIO::ROI roi(0, nw, 0, nh, 0, 1, 0, 3);
    if (!OIIO::ImageBufAlgo::resample(dstbuf, srcbuf, /*interpolate=*/true, roi, nthreads)) {
        std::string err = dstbuf.geterror();
        std::free(out);
        throw std::runtime_error(std::string("Resample failed: ") + (err.empty() ? "unknown" : err));
    }
    return out; // already filled
}

} // anonymous namespace

namespace gs {
namespace image_io {

std::tuple<int, int, int> get_image_info(const std::filesystem::path& path) {
    init_oiio();

    auto in = OIIO::ImageInput::open(path.string());
    if (!in) {
        throw std::runtime_error("OIIO open failed: " + path.string() + " : " + OIIO::geterror());
    }
    const OIIO::ImageSpec& spec = in->spec();
    const int w = spec.width;
    const int h = spec.height;
    const int c = spec.nchannels;
    in->close();
    return {w, h, c};
}

std::tuple<unsigned char*, int, int, int>
load_image(const std::filesystem::path& path, int resize_factor) {
    init_oiio();

    std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(path.string()));
    if (!in)
        throw std::runtime_error("Load failed: " + path.string() + " : " + OIIO::geterror());

    const OIIO::ImageSpec& spec = in->spec();
    int w = spec.width, h = spec.height, file_c = spec.nchannels;

    auto finish = [&](unsigned char* data, int W, int H, int C) {
        in->close();
        return std::make_tuple(data, W, H, C);
    };

    // Decide threading for the resample
    const int nthreads = 0; // 0 = auto, 1 = single-threaded

    // Fast path: read 3 channels directly (drop alpha if present)
    if (file_c >= 3) {
        if (resize_factor <= 1) {
            // allocate and read directly into final RGB buffer
            auto* out = static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
            if (!out) {
                in->close();
                throw std::bad_alloc();
            }

            if (!in->read_image(/*subimage*/ 0, /*miplevel*/ 0,
                                /*chbegin*/ 0, /*chend*/ 3,
                                OIIO::TypeDesc::UINT8, out)) {
                std::string e = in->geterror();
                std::free(out);
                in->close();
                throw std::runtime_error("Read failed: " + path.string() + (e.empty() ? "" : (" : " + e)));
            }
            return finish(out, w, h, 3);
        } else if (resize_factor == 2 || resize_factor == 4 || resize_factor == 8) {
            // read full, then downscale
            auto* full = static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
            if (!full) {
                in->close();
                throw std::bad_alloc();
            }

            if (!in->read_image(0, 0, 0, 3, OIIO::TypeDesc::UINT8, full)) {
                std::string e = in->geterror();
                std::free(full);
                in->close();
                throw std::runtime_error("Read failed: " + path.string() + (e.empty() ? "" : (" : " + e)));
            }
            in->close();

            const int nw = std::max(1, w / resize_factor);
            const int nh = std::max(1, h / resize_factor);
            unsigned char* out = nullptr;
            try {
                out = downscale_resample_direct(full, w, h, nw, nh, nthreads);
            } catch (...) {
                std::free(full);
                throw;
            }
            std::free(full);
            return {out, nw, nh, 3};
        } else {
            LOG_ERROR("load_image: unsupported resize factor {}", resize_factor);
            // fall through
        }
    }

    // 1-2 channel inputs -> read native, then expand to RGB
    {
        const int in_c = std::min(2, std::max(1, file_c));
        std::vector<unsigned char> tmp((size_t)w * h * in_c);
        if (!in->read_image(0, 0, 0, in_c, OIIO::TypeDesc::UINT8, tmp.data())) {
            auto e = in->geterror();
            in->close();
            throw std::runtime_error("Read failed: " + path.string() + (e.empty() ? "" : (" : " + e)));
        }
        in->close();

        auto* base = static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
        if (!base)
            throw std::bad_alloc();

        if (in_c == 1) {
            const unsigned char* g = tmp.data();
            for (size_t i = 0, N = (size_t)w * h; i < N; ++i) {
                unsigned char v = g[i];
                base[3 * i + 0] = v;
                base[3 * i + 1] = v;
                base[3 * i + 2] = v;
            }
        } else { // 2 channels -> (R,G,avg)
            const unsigned char* src = tmp.data();
            for (size_t i = 0, N = (size_t)w * h; i < N; ++i) {
                unsigned char r = src[2 * i + 0];
                unsigned char g = src[2 * i + 1];
                base[3 * i + 0] = r;
                base[3 * i + 1] = g;
                base[3 * i + 2] = (unsigned char)(((int)r + (int)g) / 2);
            }
        }

        if (resize_factor == 2 || resize_factor == 4 || resize_factor == 8) {
            const int nw = std::max(1, w / resize_factor);
            const int nh = std::max(1, h / resize_factor);
            unsigned char* out = nullptr;
            try {
                out = downscale_resample_direct(base, w, h, nw, nh, nthreads);
            } catch (...) {
                std::free(base);
                throw;
            }
            std::free(base);
            return {out, nw, nh, 3};
        }

        return {base, w, h, 3};
    }
}

void free_image(unsigned char* data) {
    std::free(data);
}

void save_image(const std::filesystem::path& path, const Tensor& image) {
    init_oiio();

    if (!image.is_valid() || image.is_empty()) {
        throw std::runtime_error("Cannot save invalid or empty tensor");
    }

    // Clone and normalize to CPU Float32
    Tensor img = image.clone();
    
    if (img.device() != Device::CPU) {
        img = img.cpu();
    }
    
    if (img.dtype() != DataType::Float32) {
        img = img.to(DataType::Float32);
    }

    // Handle batch dimension [B, C, H, W] -> [C, H, W]
    if (img.ndim() == 4) {
        if (img.size(0) != 1) {
            throw std::runtime_error("Batch dimension must be 1");
        }
        img = img.squeeze(0);
    }

    // Convert [C, H, W] -> [H, W, C] if needed
    if (img.ndim() == 3 && img.size(0) <= 4 && img.size(0) < img.size(1)) {
        img = img.permute({1, 2, 0}).contiguous();
    }

    if (img.ndim() != 3) {
        throw std::runtime_error("Image tensor must be 3D, got: " + img.shape().str());
    }

    const int height = static_cast<int>(img.size(0));
    const int width = static_cast<int>(img.size(1));
    const int channels = static_cast<int>(img.size(2));

    if (channels < 1 || channels > 4) {
        throw std::runtime_error("Channels must be in [1..4], got: " + std::to_string(channels));
    }

    LOG_DEBUG("Saving image: {} shape: [{}, {}, {}]", path.string(), height, width, channels);

    // CRITICAL: Store each step in a named variable to ensure lifetime
    Tensor img_clamped = img.clamp(0.0f, 1.0f);              // Step 1
    Tensor img_scaled = img_clamped.mul(255.0f);             // Step 2
    Tensor img_uint8_temp = img_scaled.to(DataType::UInt8); // Step 3
    Tensor img_uint8 = img_uint8_temp.contiguous();          // Step 4 - ensure contiguous
    
    // Now get the pointer - img_uint8 is guaranteed to live until end of function
    const uint8_t* data_ptr = img_uint8.ptr<uint8_t>();
    if (!data_ptr) {
        throw std::runtime_error("Failed to get data pointer from tensor");
    }

    // Prepare OIIO output
    const std::string fname = path.string();
    auto out = OIIO::ImageOutput::create(fname);
    if (!out) {
        throw std::runtime_error("ImageOutput::create failed for " + fname);
    }

    OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::UINT8);

    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), 
                  [](unsigned char c) { return std::tolower(c); });
    if (ext == ".jpg" || ext == ".jpeg")
        spec.attribute("CompressionQuality", 95);

    if (!out->open(fname, spec)) {
        throw std::runtime_error("Failed to open output file: " + fname);
    }

    if (!out->write_image(OIIO::TypeDesc::UINT8, data_ptr)) {
        out->close();
        throw std::runtime_error("Failed to write image data");
    }
    
    out->close();
}

void save_image(const std::filesystem::path& path,
                const std::vector<Tensor>& images,
                bool horizontal,
                int separator_width) {
    if (images.empty())
        throw std::runtime_error("No images provided");

    if (images.size() == 1) {
        save_image(path, images[0]);
        return;
    }

    // Prepare all to HWC float on CPU
    std::vector<Tensor> xs;
    xs.reserve(images.size());
    for (const auto& img : images) {
        Tensor processed = img.clone();
        
        if (processed.device() != Device::CPU) {
            processed = processed.cpu();
        }
        
        if (processed.dtype() != DataType::Float32) {
            processed = processed.to(DataType::Float32);
        }
        
        // Handle batch dimension
        if (processed.ndim() == 4) {
            if (processed.size(0) != 1) {
                throw std::runtime_error("Batch dimension must be 1");
            }
            processed = processed.squeeze(0);
        }
        
        // Convert [C, H, W] -> [H, W, C]
        if (processed.ndim() == 3 && processed.size(0) <= 4 && processed.size(0) < processed.size(1)) {
            processed = processed.permute({1, 2, 0}).contiguous();
        }
        
        xs.push_back(processed);
    }

    // Create separator (white)
    Tensor sep;
    if (separator_width > 0) {
        const auto& ref = xs[0];
        if (horizontal) {
            sep = Tensor::ones({ref.size(0), static_cast<size_t>(separator_width), ref.size(2)}, 
                              Device::CPU, DataType::Float32);
        } else {
            sep = Tensor::ones({static_cast<size_t>(separator_width), ref.size(1), ref.size(2)}, 
                              Device::CPU, DataType::Float32);
        }
    }

    // Concatenate
    Tensor combo = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) {
        if (separator_width > 0) {
            combo = combo.cat(sep, horizontal ? 1 : 0);
        }
        combo = combo.cat(xs[i], horizontal ? 1 : 0);
    }

    // Save
    save_image(path, combo);
}

// ============================================================================
// BatchImageSaver Implementation
// ============================================================================

BatchImageSaver::BatchImageSaver(size_t num_workers)
    : num_workers_(std::min(num_workers, 
                           std::min(size_t(8), size_t(std::thread::hardware_concurrency())))) {

    LOG_INFO("[BatchImageSaver] Starting with {} worker threads", num_workers_);
    for (size_t i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&BatchImageSaver::worker_thread, this);
    }
}

BatchImageSaver::~BatchImageSaver() {
    shutdown();
}

void BatchImageSaver::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_)
            return;
        stop_ = true;
        LOG_INFO("[BatchImageSaver] Shutting down...");
    }
    cv_.notify_all();

    for (auto& w : workers_)
        if (w.joinable())
            w.join();

    // Process remaining tasks synchronously
    while (!task_queue_.empty()) {
        process_task(task_queue_.front());
        task_queue_.pop();
    }
    LOG_INFO("[BatchImageSaver] Shutdown complete");
}

void BatchImageSaver::queue_save(const std::filesystem::path& path, Tensor image) {
    if (!enabled_) {
        save_image(path, image);
        return;
    }

    SaveTask t;
    t.path = path;
    t.image = image.clone();
    t.is_multi = false;
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            save_image(path, image);
            return;
        }
        task_queue_.push(std::move(t));
        active_tasks_++;
    }
    cv_.notify_one();
}

void BatchImageSaver::queue_save_multiple(const std::filesystem::path& path,
                                          const std::vector<Tensor>& images,
                                          bool horizontal,
                                          int separator_width) {
    if (!enabled_) {
        save_image(path, images, horizontal, separator_width);
        return;
    }

    SaveTask t;
    t.path = path;
    t.images.reserve(images.size());
    for (const auto& img : images)
        t.images.push_back(img.clone());
    t.is_multi = true;
    t.horizontal = horizontal;
    t.separator_width = separator_width;

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            save_image(path, images, horizontal, separator_width);
            return;
        }
        task_queue_.push(std::move(t));
        active_tasks_++;
    }
    cv_.notify_one();
}

void BatchImageSaver::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_finished_.wait(lock, [this] { return task_queue_.empty() && active_tasks_ == 0; });
}

size_t BatchImageSaver::pending_count() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return task_queue_.size() + active_tasks_;
}

void BatchImageSaver::worker_thread() {
    while (true) {
        SaveTask t;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });
            if (stop_ && task_queue_.empty())
                break;
            if (task_queue_.empty())
                continue;
            t = std::move(task_queue_.front());
            task_queue_.pop();
        }
        process_task(t);
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_tasks_--;
        }
        cv_finished_.notify_all();
    }
}

void BatchImageSaver::process_task(const SaveTask& t) {
    try {
        if (t.is_multi) {
            save_image(t.path, t.images, t.horizontal, t.separator_width);
        } else {
            save_image(t.path, t.image);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[BatchImageSaver] Error saving {}: {}", t.path.string(), e.what());
    }
}

} // namespace image_io
} // namespace gs
