/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file losses.hpp
 * @brief Main header for all loss functions
 *
 * All losses follow the "Option A: Structs with Static Methods" pattern:
 * - Params struct for configuration
 * - Context struct for backward pass data (if needed)
 * - Static forward() method for computing loss and context
 * - Static backward() method for computing gradients (if not done in-place)
 */

#include "photometric_loss.hpp"
#include "scale_regularization.hpp"
#include "opacity_regularization.hpp"
#include "sparsity_loss.hpp"
