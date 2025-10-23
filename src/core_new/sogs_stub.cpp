/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Temporary stub for SOG writing
 * TODO: Implement this properly with morton encoding and kmeans
 */

#include "core_new/sogs.hpp"
#include "core_new/logger.hpp"
#include <expected>
#include <string>

namespace lfs::core {

    std::expected<void, std::string>
    write_sog(const SplatData& splat_data, const SogWriteOptions& options) {
        LOG_ERROR("write_sog: Stub implementation - SOG writing not yet implemented for lfs::core");
        return std::unexpected("SOG writing not yet implemented");
    }

} // namespace lfs::core
