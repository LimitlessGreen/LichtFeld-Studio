/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// formats/ply.hpp
#pragma once

#include "loader/cuda_data.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace gs::loader {

    // Internal torch-free version
    std::expected<internal::CudaSplatData, std::string>
    load_ply_cuda(const std::filesystem::path& filepath);

} // namespace gs::loader
