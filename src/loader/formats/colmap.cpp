/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "colmap.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "loader/filesystem_utils.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace gs::loader {

    namespace fs = std::filesystem;

    // -----------------------------------------------------------------------------
    //  Quaternion to rotation matrix (CUDA version)
    // -----------------------------------------------------------------------------
    inline void qvec2rotmat_cuda(const float qraw[4], float R[9]) {
        // Normalize quaternion
        float norm = std::sqrt(qraw[0]*qraw[0] + qraw[1]*qraw[1] +
                              qraw[2]*qraw[2] + qraw[3]*qraw[3]);
        float q[4];
        for (int i = 0; i < 4; ++i) {
            q[i] = qraw[i] / norm;
        }

        float w = q[0], x = q[1], y = q[2], z = q[3];

        R[0] = 1 - 2 * (y * y + z * z);
        R[1] = 2 * (x * y - z * w);
        R[2] = 2 * (x * z + y * w);

        R[3] = 2 * (x * y + z * w);
        R[4] = 1 - 2 * (x * x + z * z);
        R[5] = 2 * (y * z - x * w);

        R[6] = 2 * (x * z - y * w);
        R[7] = 2 * (y * z + x * w);
        R[8] = 1 - 2 * (x * x + y * y);
    }

    class Image {
    public:
        Image() = default;
        explicit Image(uint32_t id) : _image_ID(id) {}

        uint32_t _camera_id = 0;
        std::string _name;
        float _qvec[4] = {1.f, 0.f, 0.f, 0.f};
        float _tvec[3] = {0.f, 0.f, 0.f};

    private:
        uint32_t _image_ID = 0;
    };

    // -----------------------------------------------------------------------------
    //  POD read helpers
    // -----------------------------------------------------------------------------
    static inline uint64_t read_u64(const char*& p) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    static inline uint32_t read_u32(const char*& p) {
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    static inline int32_t read_i32(const char*& p) {
        int32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    static inline double read_f64(const char*& p) {
        double v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }

    // -----------------------------------------------------------------------------
    //  COLMAP camera-model map
    // -----------------------------------------------------------------------------
    static const std::unordered_map<int, std::pair<CAMERA_MODEL, int32_t>> camera_model_ids = {
        {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
        {1, {CAMERA_MODEL::PINHOLE, 4}},
        {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
        {3, {CAMERA_MODEL::RADIAL, 5}},
        {4, {CAMERA_MODEL::OPENCV, 8}},
        {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
        {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
        {7, {CAMERA_MODEL::FOV, 5}},
        {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
        {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
        {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
        {11, {CAMERA_MODEL::UNDEFINED, -1}}
    };

    static const std::unordered_map<std::string, CAMERA_MODEL> camera_model_names = {
        {"SIMPLE_PINHOLE", CAMERA_MODEL::SIMPLE_PINHOLE},
        {"PINHOLE", CAMERA_MODEL::PINHOLE},
        {"SIMPLE_RADIAL", CAMERA_MODEL::SIMPLE_RADIAL},
        {"RADIAL", CAMERA_MODEL::RADIAL},
        {"OPENCV", CAMERA_MODEL::OPENCV},
        {"OPENCV_FISHEYE", CAMERA_MODEL::OPENCV_FISHEYE},
        {"FULL_OPENCV", CAMERA_MODEL::FULL_OPENCV},
        {"FOV", CAMERA_MODEL::FOV},
        {"SIMPLE_RADIAL_FISHEYE", CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE},
        {"RADIAL_FISHEYE", CAMERA_MODEL::RADIAL_FISHEYE},
        {"THIN_PRISM_FISHEYE", CAMERA_MODEL::THIN_PRISM_FISHEYE}
    };

    // -----------------------------------------------------------------------------
    //  Binary-file loader
    // -----------------------------------------------------------------------------
    static std::unique_ptr<std::vector<char>>
    read_binary(const std::filesystem::path& p) {
        LOG_TRACE("Reading binary file: {}", p.string());
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) {
            LOG_ERROR("Failed to open binary file: {}", p.string());
            throw std::runtime_error("Failed to open " + p.string());
        }

        auto sz = static_cast<std::streamsize>(f.tellg());
        auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(sz));

        f.seekg(0, std::ios::beg);
        f.read(buf->data(), sz);
        if (!f) {
            LOG_ERROR("Short read on binary file: {}", p.string());
            throw std::runtime_error("Short read on " + p.string());
        }
        LOG_TRACE("Read {} bytes from {}", sz, p.string());
        return buf;
    }

    // -----------------------------------------------------------------------------
    //  Helper to scale camera intrinsics based on model
    // -----------------------------------------------------------------------------
    static void scale_camera_intrinsics(CAMERA_MODEL model, std::vector<float>& params, float factor) {
        switch (model) {
        case CAMERA_MODEL::SIMPLE_PINHOLE:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;
        case CAMERA_MODEL::PINHOLE:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;
        case CAMERA_MODEL::SIMPLE_RADIAL:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;
        case CAMERA_MODEL::RADIAL:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;
        case CAMERA_MODEL::OPENCV:
        case CAMERA_MODEL::OPENCV_FISHEYE:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;
        case CAMERA_MODEL::FULL_OPENCV:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;
        case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
        case CAMERA_MODEL::RADIAL_FISHEYE:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;
        case CAMERA_MODEL::FOV:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;
        case CAMERA_MODEL::THIN_PRISM_FISHEYE:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;
        default:
            LOG_WARN("Unknown camera model for scaling: {}", static_cast<int>(model));
            if (params.size() >= 4) {
                params[2] /= factor; // cx
                params[3] /= factor; // cy
            }
            break;
        }
    }

    // -----------------------------------------------------------------------------
    //  Helper to extract scale factor from folder name
    // -----------------------------------------------------------------------------
    static float extract_scale_from_folder(const std::string& folder_name) {
        size_t underscore_pos = folder_name.rfind('_');
        if (underscore_pos != std::string::npos) {
            std::string suffix = folder_name.substr(underscore_pos + 1);
            try {
                float factor = std::stof(suffix);
                if (factor > 0 && factor <= 16) {
                    LOG_DEBUG("Extracted scale factor {} from folder name", factor);
                    return factor;
                }
            } catch (...) {
            }
        }
        return 1.0f;
    }

    // -----------------------------------------------------------------------------
    //  Helper to apply dimension correction to camera
    // -----------------------------------------------------------------------------
    static void apply_dimension_correction(internal::CudaCameraData& cam, float scale_x, float scale_y,
                                           int actual_w, int actual_h) {
        cam.width = actual_w;
        cam.height = actual_h;
        cam.focal_x *= scale_x;
        cam.focal_y *= scale_y;
        cam.center_x *= scale_x;
        cam.center_y *= scale_y;
        LOG_TRACE("Applied dimension correction to camera: scale_x={:.3f}, scale_y={:.3f}", scale_x, scale_y);
    }

    // -----------------------------------------------------------------------------
    //  images.bin (CUDA version)
    // -----------------------------------------------------------------------------
    std::vector<Image> read_images_binary_cuda(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read images.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_images = read_u64(cur);
        LOG_DEBUG("Reading {} images from binary file", n_images);
        std::vector<Image> images;
        images.reserve(n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
            uint32_t id = read_u32(cur);
            auto& img = images.emplace_back(id);

            for (int k = 0; k < 4; ++k)
                img._qvec[k] = static_cast<float>(read_f64(cur));

            for (int k = 0; k < 3; ++k)
                img._tvec[k] = static_cast<float>(read_f64(cur));

            img._camera_id = read_u32(cur);
            img._name.assign(cur);
            cur += img._name.size() + 1;

            uint64_t npts = read_u64(cur);
            cur += npts * (sizeof(double) * 2 + sizeof(uint64_t));
        }
        if (cur != end) {
            LOG_ERROR("images.bin has trailing bytes");
            throw std::runtime_error("images.bin: trailing bytes");
        }
        return images;
    }

    // -----------------------------------------------------------------------------
    //  cameras.bin (CUDA version)
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, internal::CudaCameraData>
    read_cameras_binary_cuda(const std::filesystem::path& file_path, float scale_factor = 1.0f) {
        LOG_TIMER_TRACE("Read cameras.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_cams = read_u64(cur);
        LOG_DEBUG("Reading {} cameras from binary file{}", n_cams,
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");
        std::unordered_map<uint32_t, internal::CudaCameraData> cams;
        cams.reserve(n_cams);

        for (uint64_t i = 0; i < n_cams; ++i) {
            internal::CudaCameraData cam;
            cam.camera_id = read_u32(cur);

            int32_t model_id = read_i32(cur);
            cam.width = read_u64(cur);
            cam.height = read_u64(cur);

            if (scale_factor != 1.0f) {
                cam.width = static_cast<uint64_t>(cam.width / scale_factor);
                cam.height = static_cast<uint64_t>(cam.height / scale_factor);
                LOG_TRACE("Scaled camera {} dimensions to {}x{}", cam.camera_id, cam.width, cam.height);
            }

            auto it = camera_model_ids.find(model_id);
            if (it == camera_model_ids.end() || it->second.second < 0) {
                LOG_ERROR("Unsupported camera-model id: {}", model_id);
                throw std::runtime_error("Unsupported camera-model id " + std::to_string(model_id));
            }

            cam.camera_model = static_cast<int>(it->second.first);
            int32_t param_cnt = it->second.second;

            cam.params.resize(param_cnt);
            for (int j = 0; j < param_cnt; j++) {
                cam.params[j] = static_cast<float>(read_f64(cur));
            }

            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(it->second.first, cam.params, scale_factor);
            }

            cams.emplace(cam.camera_id, std::move(cam));
        }
        if (cur != end) {
            LOG_ERROR("cameras.bin has trailing bytes");
            throw std::runtime_error("cameras.bin: trailing bytes");
        }
        return cams;
    }

    // -----------------------------------------------------------------------------
    //  points3D.bin (CUDA version)
    // -----------------------------------------------------------------------------
    internal::CudaPointCloud read_point3D_binary_cuda(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read points3D.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t N = read_u64(cur);
        LOG_DEBUG("Reading {} 3D points from binary file", N);

        // Allocate host memory
        std::vector<float> host_positions(N * 3);
        std::vector<uint8_t> host_colors(N * 3);

        for (uint64_t i = 0; i < N; ++i) {
            cur += 8; // skip point ID

            host_positions[i * 3 + 0] = static_cast<float>(read_f64(cur));
            host_positions[i * 3 + 1] = static_cast<float>(read_f64(cur));
            host_positions[i * 3 + 2] = static_cast<float>(read_f64(cur));

            host_colors[i * 3 + 0] = *cur++;
            host_colors[i * 3 + 1] = *cur++;
            host_colors[i * 3 + 2] = *cur++;

            cur += 8; // skip reprojection error
            cur += read_u64(cur) * sizeof(uint32_t) * 2; // skip track
        }

        if (cur != end) {
            LOG_ERROR("points3D.bin has trailing bytes");
            throw std::runtime_error("points3D.bin: trailing bytes");
        }

        // Upload to CUDA
        internal::CudaPointCloud result(N);
        result.positions.upload(host_positions.data(), N * 3);
        result.colors.upload(host_colors.data(), N * 3);

        return result;
    }

    // -----------------------------------------------------------------------------
    //  Text-file loader helpers
    // -----------------------------------------------------------------------------
    std::vector<std::string> read_text_file(const std::filesystem::path& file_path) {
        LOG_TRACE("Reading text file: {}", file_path.string());
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open text file: {}", file_path.string());
            throw std::runtime_error("Failed to open " + file_path.string());
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (line.starts_with("#")) {
                continue;
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            lines.push_back(line);
        }
        file.close();
        if (lines.empty()) {
            LOG_ERROR("File is empty or contains no valid lines: {}", file_path.string());
            throw std::runtime_error("File " + file_path.string() + " is empty or contains no valid lines");
        }
        if (lines.back().empty()) {
            lines.pop_back();
        }
        LOG_TRACE("Read {} lines from text file", lines.size());
        return lines;
    }

    std::vector<std::string> split_string(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        size_t start = 0;
        size_t end = s.find(delimiter);

        while (end != std::string::npos) {
            tokens.push_back(s.substr(start, end - start));
            start = end + 1;
            end = s.find(delimiter, start);
        }
        tokens.push_back(s.substr(start));

        return tokens;
    }

    // -----------------------------------------------------------------------------
    //  images.txt (CUDA version)
    // -----------------------------------------------------------------------------
    std::vector<Image> read_images_text_cuda(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read images.txt");
        auto lines = read_text_file(file_path);
        std::vector<Image> images;
        if (lines.size() % 2 != 0) {
            LOG_ERROR("images.txt should have an even number of lines");
            throw std::runtime_error("images.txt should have an even number of lines");
        }
        uint64_t n_images = lines.size() / 2;
        LOG_DEBUG("Reading {} images from text file", n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
            const auto& line = lines[i * 2];
            const auto tokens = split_string(line, ' ');
            if (tokens.size() != 10) {
                LOG_ERROR("Invalid format in images.txt line {}", i * 2 + 1);
                throw std::runtime_error("Invalid format in images.txt line " + std::to_string(i * 2 + 1));
            }

            auto& img = images.emplace_back(std::stoul(tokens[0]));
            img._qvec[0] = std::stof(tokens[1]);
            img._qvec[1] = std::stof(tokens[2]);
            img._qvec[2] = std::stof(tokens[3]);
            img._qvec[3] = std::stof(tokens[4]);

            img._tvec[0] = std::stof(tokens[5]);
            img._tvec[1] = std::stof(tokens[6]);
            img._tvec[2] = std::stof(tokens[7]);

            img._camera_id = std::stoul(tokens[8]);
            img._name = tokens[9];
        }
        return images;
    }

    // -----------------------------------------------------------------------------
    //  cameras.txt (CUDA version)
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, internal::CudaCameraData>
    read_cameras_text_cuda(const std::filesystem::path& file_path, float scale_factor = 1.0f) {
        LOG_TIMER_TRACE("Read cameras.txt");
        auto lines = read_text_file(file_path);
        std::unordered_map<uint32_t, internal::CudaCameraData> cams;
        LOG_DEBUG("Reading {} cameras from text file{}", lines.size(),
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        for (const auto& line : lines) {
            const auto tokens = split_string(line, ' ');
            if (tokens.size() < 4) {
                LOG_ERROR("Invalid format in cameras.txt: {}", line);
                throw std::runtime_error("Invalid format in cameras.txt: " + line);
            }

            internal::CudaCameraData cam;
            cam.camera_id = std::stoul(tokens[0]);
            if (!camera_model_names.contains(tokens[1])) {
                LOG_ERROR("Unknown camera model in cameras.txt: {}", tokens[1]);
                throw std::runtime_error("Invalid format in cameras.txt: " + line);
            }
            cam.camera_model = static_cast<int>(camera_model_names.at(tokens[1]));
            cam.width = std::stoi(tokens[2]);
            cam.height = std::stoi(tokens[3]);

            if (scale_factor != 1.0f) {
                cam.width = static_cast<uint64_t>(cam.width / scale_factor);
                cam.height = static_cast<uint64_t>(cam.height / scale_factor);
                LOG_TRACE("Scaled camera {} dimensions to {}x{}", cam.camera_id, cam.width, cam.height);
            }

            cam.params.resize(tokens.size() - 4);
            for (uint64_t j = 4; j < tokens.size(); ++j) {
                cam.params[j - 4] = std::stof(tokens[j]);
            }

            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(static_cast<CAMERA_MODEL>(cam.camera_model), cam.params, scale_factor);
            }

            cams.emplace(cam.camera_id, std::move(cam));
        }
        return cams;
    }

    // -----------------------------------------------------------------------------
    //  point3D.txt (CUDA version)
    // -----------------------------------------------------------------------------
    internal::CudaPointCloud read_point3D_text_cuda(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read points3D.txt");
        auto lines = read_text_file(file_path);
        uint64_t N = lines.size();
        LOG_DEBUG("Reading {} 3D points from text file", N);

        std::vector<float> host_positions(N * 3);
        std::vector<uint8_t> host_colors(N * 3);

        for (uint64_t i = 0; i < N; ++i) {
            const auto& line = lines[i];
            const auto tokens = split_string(line, ' ');

            if (tokens.size() < 8) {
                LOG_ERROR("Invalid format in points3D.txt: {}", line);
                throw std::runtime_error("Invalid format in point3D.txt: " + line);
            }

            host_positions[i * 3 + 0] = std::stof(tokens[1]);
            host_positions[i * 3 + 1] = std::stof(tokens[2]);
            host_positions[i * 3 + 2] = std::stof(tokens[3]);

            host_colors[i * 3 + 0] = std::stoi(tokens[4]);
            host_colors[i * 3 + 1] = std::stoi(tokens[5]);
            host_colors[i * 3 + 2] = std::stoi(tokens[6]);
        }

        internal::CudaPointCloud result(N);
        result.positions.upload(host_positions.data(), N * 3);
        result.colors.upload(host_colors.data(), N * 3);

        return result;
    }

    // -----------------------------------------------------------------------------
    //  Assemble per-image camera information (CUDA version)
    // -----------------------------------------------------------------------------
    ColmapCameraResult
    read_colmap_cameras_cuda(const std::filesystem::path base_path,
                            const std::unordered_map<uint32_t, internal::CudaCameraData>& cams,
                            const std::vector<Image>& images,
                            const std::string& images_folder = "images") {
        LOG_TIMER_TRACE("Assemble COLMAP cameras");
        std::vector<internal::CudaCameraData> out(images.size());

        std::filesystem::path images_path = base_path / images_folder;

        if (!std::filesystem::exists(images_path)) {
            LOG_ERROR("Images folder does not exist: {}", images_path.string());
            throw std::runtime_error("Images folder does not exist: " + images_path.string());
        }

        for (size_t i = 0; i < images.size(); ++i) {
            const Image& img = images[i];
            auto it = cams.find(img._camera_id);
            if (it == cams.end()) {
                LOG_ERROR("Camera ID {} not found", img._camera_id);
                throw std::runtime_error("Camera ID " + std::to_string(img._camera_id) + " not found");
            }

            out[i] = it->second;
            out[i].image_path = images_path / img._name;
            out[i].image_name = img._name;

            qvec2rotmat_cuda(img._qvec, out[i].R);
            std::memcpy(out[i].T, img._tvec, 3 * sizeof(float));

            // Process camera model
            CAMERA_MODEL model = static_cast<CAMERA_MODEL>(out[i].camera_model);
            switch (model) {
            case CAMERA_MODEL::SIMPLE_PINHOLE: {
                float fx = out[i].params[0];
                out[i].focal_x = fx;
                out[i].focal_y = fx;
                out[i].center_x = out[i].params[1];
                out[i].center_y = out[i].params[2];
                out[i].camera_model_type = 0; // PINHOLE
                break;
            }
            case CAMERA_MODEL::PINHOLE: {
                out[i].focal_x = out[i].params[0];
                out[i].focal_y = out[i].params[1];
                out[i].center_x = out[i].params[2];
                out[i].center_y = out[i].params[3];
                out[i].camera_model_type = 0; // PINHOLE
                break;
            }
            case CAMERA_MODEL::SIMPLE_RADIAL: {
                float fx = out[i].params[0];
                out[i].focal_x = fx;
                out[i].focal_y = fx;
                out[i].center_x = out[i].params[1];
                out[i].center_y = out[i].params[2];
                out[i].radial_distortion.push_back(out[i].params[3]);
                out[i].camera_model_type = 0; // PINHOLE
                break;
            }
            case CAMERA_MODEL::RADIAL: {
                float fx = out[i].params[0];
                out[i].focal_x = fx;
                out[i].focal_y = fx;
                out[i].center_x = out[i].params[1];
                out[i].center_y = out[i].params[2];
                out[i].radial_distortion.push_back(out[i].params[3]);
                out[i].radial_distortion.push_back(out[i].params[4]);
                out[i].camera_model_type = 0; // PINHOLE
                break;
            }
            case CAMERA_MODEL::OPENCV: {
                out[i].focal_x = out[i].params[0];
                out[i].focal_y = out[i].params[1];
                out[i].center_x = out[i].params[2];
                out[i].center_y = out[i].params[3];
                out[i].radial_distortion.push_back(out[i].params[4]);
                out[i].radial_distortion.push_back(out[i].params[5]);
                out[i].tangential_distortion.push_back(out[i].params[6]);
                out[i].tangential_distortion.push_back(out[i].params[7]);
                out[i].camera_model_type = 0; // PINHOLE
                break;
            }
            case CAMERA_MODEL::FULL_OPENCV: {
                out[i].focal_x = out[i].params[0];
                out[i].focal_y = out[i].params[1];
                out[i].center_x = out[i].params[2];
                out[i].center_y = out[i].params[3];
                for (size_t j = 4; j <= 11 && j < out[i].params.size(); ++j) {
                    if (j <= 5 || j >= 8) {
                        out[i].radial_distortion.push_back(out[i].params[j]);
                    } else {
                        out[i].tangential_distortion.push_back(out[i].params[j]);
                    }
                }
                out[i].camera_model_type = 0; // PINHOLE
                break;
            }
            case CAMERA_MODEL::OPENCV_FISHEYE: {
                out[i].focal_x = out[i].params[0];
                out[i].focal_y = out[i].params[1];
                out[i].center_x = out[i].params[2];
                out[i].center_y = out[i].params[3];
                for (size_t j = 4; j <= 7 && j < out[i].params.size(); ++j) {
                    out[i].radial_distortion.push_back(out[i].params[j]);
                }
                out[i].camera_model_type = 1; // FISHEYE
                break;
            }
            case CAMERA_MODEL::RADIAL_FISHEYE: {
                float fx = out[i].params[0];
                out[i].focal_x = fx;
                out[i].focal_y = fx;
                out[i].center_x = out[i].params[1];
                out[i].center_y = out[i].params[2];
                out[i].radial_distortion.push_back(out[i].params[3]);
                out[i].radial_distortion.push_back(out[i].params[4]);
                out[i].camera_model_type = 1; // FISHEYE
                break;
            }
            case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE: {
                float fx = out[i].params[0];
                out[i].focal_x = fx;
                out[i].focal_y = fx;
                out[i].center_x = out[i].params[1];
                out[i].center_y = out[i].params[2];
                out[i].radial_distortion.push_back(out[i].params[3]);
                out[i].camera_model_type = 1; // FISHEYE
                break;
            }
            case CAMERA_MODEL::THIN_PRISM_FISHEYE: {
                throw std::runtime_error("THIN_PRISM_FISHEYE camera model is not supported");
            }
            case CAMERA_MODEL::FOV: {
                throw std::runtime_error("FOV camera model is not supported");
            }
            default:
                LOG_ERROR("Unsupported camera model");
                throw std::runtime_error("Unsupported camera model");
            }
        }

        // Verify actual image dimensions and apply correction if needed
        if (!out.empty() && std::filesystem::exists(out[0].image_path)) {
            LOG_DEBUG("Verifying actual image dimensions against COLMAP database");

            auto [img_data, actual_w, actual_h, channels] = load_image(out[0].image_path);

            int expected_w = out[0].width;
            int expected_h = out[0].height;

            float scale_x = static_cast<float>(actual_w) / expected_w;
            float scale_y = static_cast<float>(actual_h) / expected_h;

            if (std::abs(scale_x - 1.0f) > 1e-5 || std::abs(scale_y - 1.0f) > 1e-5) {
                LOG_WARN("Image dimension mismatch detected!");
                LOG_INFO("  Expected (from COLMAP): {}x{}", expected_w, expected_h);
                LOG_INFO("  Actual (from image file): {}x{}", actual_w, actual_h);
                LOG_INFO("  Applying correction scale: {:.3f}x{:.3f}", scale_x, scale_y);

                for (auto& cam : out) {
                    apply_dimension_correction(cam, scale_x, scale_y, actual_w, actual_h);
                }
            } else {
                LOG_DEBUG("Image dimensions match COLMAP database ({}x{})", actual_w, actual_h);
            }

            free_image(img_data);
        }

        // Compute scene center
        float scene_center[3];
        internal::compute_scene_center(out, scene_center);

        ColmapCameraResult result;
        result.cameras = std::move(out);
        result.scene_center[0] = scene_center[0];
        result.scene_center[1] = scene_center[1];
        result.scene_center[2] = scene_center[2];

        LOG_INFO("Training with {} images", result.cameras.size());
        return result;
    }

    // -----------------------------------------------------------------------------
    //  Helper to get sparse file path
    // -----------------------------------------------------------------------------
    static fs::path get_sparse_file_path(const fs::path& base, const std::string& filename) {
        auto search_paths = get_colmap_search_paths(base);
        auto found = find_file_in_paths(search_paths, filename);

        if (!found.empty()) {
            LOG_TRACE("Found sparse file at: {}", found.string());
            return found;
        }

        std::string error_msg = std::format("Cannot find '{}' in any of these locations:\n", filename);
        for (const auto& dir : search_paths) {
            error_msg += std::format("  - {}\n", (dir / filename).string());
        }
        error_msg += "Searched case-insensitively for: " + filename;

        LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // -----------------------------------------------------------------------------
    //  Public API functions (CUDA versions)
    // -----------------------------------------------------------------------------
    internal::CudaPointCloud read_colmap_point_cloud_cuda(const std::filesystem::path& filepath) {
        LOG_TIMER_TRACE("Read COLMAP point cloud");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.bin");
        return read_point3D_binary_cuda(points3d_file);
    }

    ColmapCameraResult read_colmap_cameras_and_images_cuda(
        const std::filesystem::path& base,
        const std::string& images_folder) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.bin");
        fs::path images_file = get_sparse_file_path(base, "images.bin");

        auto cams = read_cameras_binary_cuda(cams_file, scale_factor);
        auto images = read_images_binary_cuda(images_file);

        LOG_INFO("Read {} cameras and {} images from COLMAP", cams.size(), images.size());

        return read_colmap_cameras_cuda(base, cams, images, images_folder);
    }

    internal::CudaPointCloud read_colmap_point_cloud_text_cuda(const std::filesystem::path& filepath) {
        LOG_TIMER_TRACE("Read COLMAP point cloud (text)");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.txt");
        return read_point3D_text_cuda(points3d_file);
    }

    ColmapCameraResult read_colmap_cameras_and_images_text_cuda(
        const std::filesystem::path& base,
        const std::string& images_folder) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images (text)");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.txt");
        fs::path images_file = get_sparse_file_path(base, "images.txt");

        auto cams = read_cameras_text_cuda(cams_file, scale_factor);
        auto images = read_images_text_cuda(images_file);

        LOG_INFO("Read {} cameras and {} images from COLMAP text files", cams.size(), images.size());

        return read_colmap_cameras_cuda(base, cams, images, images_folder);
    }

} // namespace gs::loader