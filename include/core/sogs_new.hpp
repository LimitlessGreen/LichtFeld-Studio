/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <string>

namespace gs {
    // Forward declaration
    class SplatDataNew;

    namespace core {

        struct SogWriteOptionsNew {
            int iterations = 10;
            std::filesystem::path output_path;
        };

        /**
         * @brief Write Gaussian Splat data to SOG format without libtorch dependency
         *
         * Uses custom Tensor class and kmeans_new for clustering.
         * Supports both .sog bundle archives and individual file output.
         *
         * @param splat_data The Gaussian Splat data to write
         * @param options Write options including output path and iteration count
         * @return void on success, error message on failure
         */
        std::expected<void, std::string> write_sog_new(
            const SplatDataNew& splat_data,
            const SogWriteOptionsNew& options);

    } // namespace core
} // namespace gs