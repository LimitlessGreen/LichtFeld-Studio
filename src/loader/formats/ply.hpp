/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data_new.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace gs::loader {

    /**
     * @brief Load PLY file and return SplatDataNew
     * @param filepath Path to the PLY file
     * @return SplatDataNew on success, error string on failure
     */
    std::expected<SplatDataNew, std::string>
    load_ply(const std::filesystem::path& filepath);

} // namespace gs::loader