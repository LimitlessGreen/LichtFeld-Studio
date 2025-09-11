/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"

#include <c10/cuda/CUDAAllocatorConfig.h>
#include <iostream>
#include <print>

int main(int argc, char* argv[]) {
    // Parse arguments (this automatically initializes the logger based on --log-level flag)
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        // Logger is already initialized, so we can use it for errors
        LOG_ERROR("Failed to parse arguments: {}", params_result.error());
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }

    // Logger is now ready to use
    LOG_INFO("========================================");
    LOG_INFO("LichtFeld Studio");
    LOG_INFO("========================================");

    auto params = std::move(*params_result);

    gs::Application app;
    return app.run(std::move(params));
}
