/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/events.hpp"
#include "core_new/parameters.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis {

    class DataLoadingService {
    public:
        explicit DataLoadingService(SceneManager* scene_manager);
        ~DataLoadingService();

        // Set parameters for dataset loading
        void setParameters(const lfs::core::param::TrainingParameters& params) { params_ = params; }
        const lfs::core::param::TrainingParameters& getParameters() const { return params_; }

        // Loading operations
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path);
        std::expected<void, std::string> loadSOG(const std::filesystem::path& path);
        std::expected<void, std::string> loadSplatFile(const std::filesystem::path& path);
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path);
        void clearScene();

    private:
        void setupEventHandlers();
        void handleLoadFileCommand(bool is_dataset, const std::filesystem::path& path);
        void addPLYToScene(const std::filesystem::path& path);
        void addSOGToScene(const std::filesystem::path& path);
        void addSplatFileToScene(const std::filesystem::path& path);

        // Helper to determine file type
        bool isSOGFile(const std::filesystem::path& path) const;
        bool isPLYFile(const std::filesystem::path& path) const;

        SceneManager* scene_manager_;
        lfs::core::param::TrainingParameters params_;
    };

} // namespace lfs::vis