/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <algorithm>
#include <execution>
#include <cstring>
#include <numeric>
#include <ranges>

#define CHECK_CUDA(call) do { \
    if (auto e = call; e != cudaSuccess) { \
        LOG_ERROR("CUDA error: {}", cudaGetErrorString(e)); \
    } \
} while(0)

namespace gs {

// ============= Masking Operations =============
Tensor Tensor::masked_select(const Tensor& mask) const {
    if (!is_valid() || !mask.is_valid() || mask.dtype() != DataType::Bool ||
        shape_ != mask.shape() || device_ != mask.device()) return {};

    size_t count = mask.count_nonzero();
    if (count == 0) return empty({0}, device_, dtype_);

    auto result = empty({count}, device_, dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_masked_select(ptr<float>(), mask.ptr<unsigned char>(),
                                        result.ptr<float>(), numel(), count, 0);
        cudaDeviceSynchronize();
    } else {
        const float* src = ptr<float>();
        const unsigned char* msk = mask.ptr<unsigned char>();
        float* dst = result.ptr<float>();

        size_t dst_idx = 0;
        for (size_t i = 0; i < numel(); ++i) {
            if (msk[i]) {
                dst[dst_idx++] = src[i];
            }
        }
    }
    return result;
}

Tensor& Tensor::masked_fill_(const Tensor& mask, float value) {
    if (!is_valid() || !mask.is_valid() || mask.dtype() != DataType::Bool ||
        shape_ != mask.shape() || device_ != mask.device()) return *this;

    if (device_ == Device::CUDA) {
        tensor_ops::launch_masked_fill(ptr<float>(), mask.ptr<unsigned char>(),
                                       value, numel(), 0);
        cudaDeviceSynchronize();
    } else {
        float* data = ptr<float>();
        const unsigned char* msk = mask.ptr<unsigned char>();

        std::for_each(std::execution::par_unseq,
                     std::views::iota(0uz, numel()).begin(),
                     std::views::iota(0uz, numel()).end(),
                     [data, msk, value](size_t i) {
                         if (msk[i]) data[i] = value;
                     });
    }
    return *this;
}

Tensor Tensor::masked_fill(const Tensor& mask, float value) const {
    auto result = clone();
    result.masked_fill_(mask, value);
    return result;
}

// ============= Indexing Operations =============
Tensor Tensor::index_select(int dim, const Tensor& indices) const {
    return index_select(dim, indices, BoundaryMode::Assert);
}

Tensor Tensor::index_select(int dim, const Tensor& indices, BoundaryMode mode) const {
    if (!is_valid() || !indices.is_valid() || indices.ndim() != 1) return {};

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return {};

    // Create result shape
    auto dims = shape_.dims();
    dims[dim] = indices.numel();
    auto result = zeros(TensorShape(dims), device_, dtype_);

    if (device_ == Device::CUDA) {
        // Ensure indices are on the same device
        const int* idx_ptr = nullptr;
        Tensor idx_temp;

        if (indices.device() == device_) {
            idx_ptr = indices.ptr<int>();
        } else {
            idx_temp = indices.to(device_);
            idx_ptr = idx_temp.ptr<int>();
        }

        tensor_ops::launch_index_select(ptr<float>(), idx_ptr,
            result.ptr<float>(), shape_.dims().data(),
            shape_.rank(), dim, indices.numel(), static_cast<int>(mode), 0);
        cudaDeviceSynchronize();
    } else {
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i) outer *= shape_[i];
        for (size_t i = dim + 1; i < shape_.rank(); ++i) inner *= shape_[i];

        const float* src = ptr<float>();
        float* dst = result.ptr<float>();

        // Get indices pointer (convert to CPU if needed)
        const int* idx = nullptr;
        Tensor idx_temp;

        if (indices.device() == Device::CPU) {
            idx = indices.ptr<int>();
        } else {
            idx_temp = indices.to(Device::CPU);
            idx = idx_temp.ptr<int>();
        }

        auto process_idx = [&](int sel) -> int {
            if (mode == BoundaryMode::Clamp) {
                return std::clamp(sel, 0, static_cast<int>(shape_[dim]) - 1);
            } else if (mode == BoundaryMode::Wrap) {
                return ((sel % static_cast<int>(shape_[dim])) + shape_[dim]) % shape_[dim];
            }
            // Assert mode - handle negative indices
            if (sel < 0) sel += shape_[dim];
            return sel;
        };

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < indices.numel(); ++i) {
                int sel = process_idx(idx[i]);
                if (sel >= 0 && sel < static_cast<int>(shape_[dim])) {
                    std::copy_n(src + (o * shape_[dim] + sel) * inner,
                               inner,
                               dst + (o * indices.numel() + i) * inner);
                }
                // If out of bounds in Assert mode, leave as zero
            }
        }
    }
    return result;
}

Tensor Tensor::gather(int dim, const Tensor& indices) const {
    return gather(dim, indices, BoundaryMode::Assert);
}

Tensor Tensor::gather(int dim, const Tensor& indices, BoundaryMode mode) const {
    if (!is_valid() || !indices.is_valid()) return {};

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return {};

    // Special case: 1D indices behave like selecting along a dimension
    if (indices.ndim() == 1) {
        // Output shape: replace dimension 'dim' with indices.numel()
        std::vector<size_t> out_dims = shape_.dims();
        out_dims[dim] = indices.numel();
        auto result = zeros(TensorShape(out_dims), device_, dtype_);

        if (device_ == Device::CUDA) {
            const int* idx_ptr = nullptr;
            Tensor idx_temp;

            if (indices.device() == device_) {
                idx_ptr = indices.ptr<int>();
            } else {
                idx_temp = indices.to(device_);
                idx_ptr = idx_temp.ptr<int>();
            }

            tensor_ops::launch_gather(ptr<float>(), idx_ptr,
                result.ptr<float>(), shape_.dims().data(),
                indices.shape().dims().data(), shape_.rank(), dim,
                result.numel(), static_cast<int>(mode), 0);
            cudaDeviceSynchronize();
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();

            const int* idx_data = nullptr;
            Tensor idx_temp;

            if (indices.device() == Device::CPU) {
                idx_data = indices.ptr<int>();
            } else {
                idx_temp = indices.to(Device::CPU);
                idx_data = idx_temp.ptr<int>();
            }

            size_t outer = 1;
            for (int i = 0; i < dim; ++i) {
                outer *= shape_[i];
            }

            size_t inner = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner *= shape_[i];
            }

            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < indices.numel(); ++i) {
                    int idx = idx_data[i];

                    // Apply boundary mode
                    if (mode == BoundaryMode::Clamp) {
                        idx = std::clamp(idx, 0, static_cast<int>(shape_[dim]) - 1);
                    } else if (mode == BoundaryMode::Wrap) {
                        idx = ((idx % static_cast<int>(shape_[dim])) + static_cast<int>(shape_[dim])) % static_cast<int>(shape_[dim]);
                    } else {  // Assert mode
                        if (idx < 0) idx += shape_[dim];
                        if (idx < 0 || idx >= static_cast<int>(shape_[dim])) {
                            continue;
                        }
                    }

                    // Copy the data
                    size_t src_base = o * shape_[dim] * inner + idx * inner;
                    size_t dst_base = o * indices.numel() * inner + i * inner;
                    for (size_t j = 0; j < inner; ++j) {
                        dst[dst_base + j] = src[src_base + j];
                    }
                }
            }
        }

        return result;
    }

    // General case: multi-dimensional indices
    // Output shape matches index shape, and indices must have same rank as input
    if (indices.ndim() != shape_.rank()) {
        LOG_ERROR("For multi-dimensional gather, indices must have same rank as input");
        return {};
    }

    auto result = zeros(indices.shape(), device_, dtype_);

    if (device_ == Device::CUDA) {
        const int* idx_ptr = nullptr;
        Tensor idx_temp;

        if (indices.device() == device_) {
            idx_ptr = indices.ptr<int>();
        } else {
            idx_temp = indices.to(device_);
            idx_ptr = idx_temp.ptr<int>();
        }

        tensor_ops::launch_gather(ptr<float>(), idx_ptr,
            result.ptr<float>(), shape_.dims().data(),
            indices.shape().dims().data(), shape_.rank(), dim,
            result.numel(), static_cast<int>(mode), 0);
        cudaDeviceSynchronize();
    } else {
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();

        const int* idx_data = nullptr;
        Tensor idx_temp;

        if (indices.device() == Device::CPU) {
            idx_data = indices.ptr<int>();
        } else {
            idx_temp = indices.to(Device::CPU);
            idx_data = idx_temp.ptr<int>();
        }

        size_t total_elements = indices.numel();

        // Calculate strides for input tensor
        std::vector<size_t> input_strides(shape_.rank());
        input_strides[shape_.rank() - 1] = 1;
        for (int i = shape_.rank() - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * shape_[i + 1];
        }

        // Calculate strides for output/index tensor
        std::vector<size_t> output_strides(indices.shape().rank());
        output_strides[indices.shape().rank() - 1] = 1;
        for (int i = indices.shape().rank() - 2; i >= 0; --i) {
            output_strides[i] = output_strides[i + 1] * indices.shape()[i + 1];
        }

        for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
            // Convert linear index to coordinates in output/index tensor
            std::vector<size_t> coords(indices.shape().rank());
            size_t temp = linear_idx;
            for (size_t d = 0; d < indices.shape().rank(); ++d) {
                coords[d] = temp / output_strides[d];
                temp %= output_strides[d];
            }

            // Get the index value at this position
            int idx = idx_data[linear_idx];

            // Apply boundary mode
            if (mode == BoundaryMode::Clamp) {
                idx = std::clamp(idx, 0, static_cast<int>(shape_[dim]) - 1);
            } else if (mode == BoundaryMode::Wrap) {
                idx = ((idx % static_cast<int>(shape_[dim])) + static_cast<int>(shape_[dim])) % static_cast<int>(shape_[dim]);
            } else {  // Assert mode
                if (idx < 0) idx += shape_[dim];
                if (idx < 0 || idx >= static_cast<int>(shape_[dim])) {
                    continue;
                }
            }

            // Build input coordinates: same as output coords except at dim
            size_t input_linear_idx = 0;
            for (size_t d = 0; d < shape_.rank(); ++d) {
                size_t coord = (d == static_cast<size_t>(dim)) ? idx : coords[d];
                input_linear_idx += coord * input_strides[d];
            }

            dst[linear_idx] = src[input_linear_idx];
        }
    }

    return result;
}

Tensor Tensor::take(const Tensor& indices) const {
    if (!is_valid() || !indices.is_valid()) return {};

    Tensor indices_same_device = (indices.device() == device_)
        ? indices.clone()
        : indices.to(device_);

    auto flat = flatten();
    auto result = empty(indices_same_device.shape(), device_, dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_take(flat.ptr<float>(), indices_same_device.ptr<int>(),
                               result.ptr<float>(), flat.numel(), indices_same_device.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        const float* src = flat.ptr<float>();
        float* dst = result.ptr<float>();
        const int* idx = indices_same_device.ptr<int>();
        size_t total = flat.numel();

        std::transform(std::execution::par_unseq,
                      idx, idx + indices_same_device.numel(), dst,
                      [src, total](int pos) {
                          if (pos < 0) pos += total;
                          return (pos >= 0 && pos < static_cast<int>(total)) ? src[pos] : 0.0f;
                      });
    }
    return result;
}
// Scatter Operations
    template<typename Op>
    static Tensor& scatter_impl(Tensor& t, int dim, const Tensor& idx,
                               const Tensor& src, Op op, int mode) {
    if (!t.is_valid() || !idx.is_valid() || !src.is_valid()) return t;

    dim = (dim < 0) ? t.shape().rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(t.shape().rank())) return t;

    // Add shape validation
    if (idx.numel() != src.shape()[dim]) {
        LOG_ERROR("Index count must match source dimension size for scatter");
        return t;
    }

    if (t.device() == Device::CUDA) {
        tensor_ops::launch_scatter(t.ptr<float>(), idx.ptr<int>(),
            src.ptr<float>(), t.shape().dims().data(), src.shape().dims().data(),
            t.shape().rank(), dim, src.numel(), mode, 0);
        cudaDeviceSynchronize();
    } else {
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i) outer *= t.shape()[i];
        for (size_t i = dim + 1; i < t.shape().rank(); ++i) inner *= t.shape()[i];

        float* dst = t.ptr<float>();
        const int* indices = idx.ptr<int>();
        const float* src_data = src.ptr<float>();

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < idx.numel(); ++i) {
                int pos = indices[i];
                if (pos < 0 || pos >= static_cast<int>(t.shape()[dim])) continue;

                for (size_t j = 0; j < inner; ++j) {
                    size_t src_idx = o * idx.numel() * inner + i * inner + j;
                    size_t dst_idx = o * t.shape()[dim] * inner + pos * inner + j;

                    // Bounds check
                    if (src_idx >= src.numel() || dst_idx >= t.numel()) {
                        LOG_ERROR("Scatter index out of bounds");
                        return t;
                    }

                    op(dst[dst_idx], src_data[src_idx]);
                }
            }
        }
    }
    return t;
}

Tensor& Tensor::scatter_(int dim, const Tensor& idx, const Tensor& src, ScatterMode mode) {
    // If this is Add mode, redirect to index_add_ for better stability
    if (mode == ScatterMode::Add) {
        return index_add_(dim, idx, src);
    }

    if (!is_valid() || !idx.is_valid() || !src.is_valid()) return *this;

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return *this;

    // For 1D case, special handling
    if (shape_.rank() == 1 && dim == 0) {
        if (idx.ndim() != 1 || src.ndim() != 1) {
            LOG_ERROR("1D scatter requires 1D indices and source");
            return *this;
        }

        if (idx.numel() != src.numel()) {
            LOG_ERROR("Index and source must have same number of elements");
            return *this;
        }

        float* dst = ptr<float>();

        // Ensure indices and source are on same device as this tensor
        const int* indices = nullptr;
        const float* src_data = nullptr;
        Tensor idx_temp, src_temp;

        if (idx.device() == device_) {
            indices = idx.ptr<int>();
        } else {
            idx_temp = idx.to(device_);
            indices = idx_temp.ptr<int>();
        }

        if (src.device() == device_) {
            src_data = src.ptr<float>();
        } else {
            src_temp = src.to(device_);
            src_data = src_temp.ptr<float>();
        }

        // Now do the scatter operation
        if (device_ == Device::CUDA) {
            // For CUDA, use the kernel
            tensor_ops::launch_scatter(dst, indices, src_data,
                shape_.dims().data(), src.shape().dims().data(),
                shape_.rank(), dim, src.numel(),
                static_cast<int>(mode), 0);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            for (size_t i = 0; i < idx.numel(); ++i) {
                int pos = indices[i];
                if (pos < 0) pos += static_cast<int>(shape_[0]);
                if (pos >= 0 && pos < static_cast<int>(shape_[0])) {
                    switch (mode) {
                        case ScatterMode::Multiply:
                            dst[pos] *= src_data[i];
                            break;
                        case ScatterMode::Max:
                            dst[pos] = std::max(dst[pos], src_data[i]);
                            break;
                        case ScatterMode::Min:
                            dst[pos] = std::min(dst[pos], src_data[i]);
                            break;
                        default:
                            dst[pos] = src_data[i];
                            break;
                    }
                }
            }
        }

        return *this;
    }

    // Multi-dimensional case
    if (idx.ndim() != 1) {
        LOG_ERROR("scatter_ currently only supports 1D index tensors");
        return *this;
    }

    // Validate source shape
    std::vector<size_t> expected_shape = shape_.dims();
    expected_shape[dim] = idx.numel();

    if (src.shape() != TensorShape(expected_shape)) {
        LOG_ERROR("Source shape mismatch: expected {}, got {}",
                  TensorShape(expected_shape).str(), src.shape().str());
        return *this;
    }

    // Convert to same device if needed
    const int* idx_ptr = nullptr;
    const float* src_ptr = nullptr;
    Tensor idx_temp, src_temp;

    if (idx.device() == device_) {
        idx_ptr = idx.ptr<int>();
    } else {
        idx_temp = idx.to(device_);
        idx_ptr = idx_temp.ptr<int>();
    }

    if (src.device() == device_) {
        src_ptr = src.ptr<float>();
    } else {
        src_temp = src.to(device_);
        src_ptr = src_temp.ptr<float>();
    }

    if (device_ == Device::CUDA) {
        tensor_ops::launch_scatter(ptr<float>(), idx_ptr,
            src_ptr, shape_.dims().data(),
            src.shape().dims().data(),
            shape_.rank(), dim, src.numel(),
            static_cast<int>(mode), 0);
        cudaDeviceSynchronize();
    } else {
        size_t outer = 1;
        for (int i = 0; i < dim; ++i) {
            outer *= shape_[i];
        }

        size_t inner = 1;
        for (size_t i = dim + 1; i < shape_.rank(); ++i) {
            inner *= shape_[i];
        }

        float* dst = ptr<float>();
        const int* indices = idx_ptr;  // Use the device-converted pointer
        const float* src_data = src_ptr;  // Use the device-converted pointer

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < idx.numel(); ++i) {
                int pos = indices[i];

                // Handle negative indices
                if (pos < 0) pos += static_cast<int>(shape_[dim]);

                // Bounds check
                if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                    continue;
                }

                // Apply operation
                size_t src_base = o * idx.numel() * inner + i * inner;
                size_t dst_base = o * shape_[dim] * inner + pos * inner;

                for (size_t j = 0; j < inner; ++j) {
                    size_t src_idx = src_base + j;
                    size_t dst_idx = dst_base + j;

                    // Safety check
                    if (src_idx >= src.numel() || dst_idx >= numel()) {
                        LOG_ERROR("Index out of bounds in scatter_");
                        return *this;
                    }

                    switch (mode) {
                        case ScatterMode::Add:
                            dst[dst_idx] += src_data[src_idx];
                            break;
                        case ScatterMode::Multiply:
                            dst[dst_idx] *= src_data[src_idx];
                            break;
                        case ScatterMode::Max:
                            dst[dst_idx] = std::max(dst[dst_idx], src_data[src_idx]);
                            break;
                        case ScatterMode::Min:
                            dst[dst_idx] = std::min(dst[dst_idx], src_data[src_idx]);
                            break;
                        default:
                            dst[dst_idx] = src_data[src_idx];
                            break;
                    }
                }
            }
        }
    }

    return *this;
}

Tensor& Tensor::scatter_(int dim, const Tensor& idx, float val, ScatterMode mode) {
    auto src = full(idx.shape(), val, device_, dtype_);
    return scatter_(dim, idx, src, mode);
}

Tensor& Tensor::index_fill_(int dim, const Tensor& idx, float val) {
    return scatter_(dim, idx, val, ScatterMode::None);
}

Tensor& Tensor::index_copy_(int dim, const Tensor& idx, const Tensor& src) {
    if (!is_valid() || !idx.is_valid() || !src.is_valid()) return *this;

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return *this;

    // Validate shapes
    if (idx.ndim() != 1) {
        LOG_ERROR("index_copy_ requires 1D index tensor");
        return *this;
    }

    // Source tensor should have the right shape
    // It should match our shape except at dimension dim where it should be idx.numel()
    std::vector<size_t> expected_src_shape = shape_.dims();
    expected_src_shape[dim] = idx.numel();

    if (src.shape() != TensorShape(expected_src_shape)) {
        LOG_ERROR("Source tensor has wrong shape for index_copy_");
        return *this;
    }

    if (device_ == Device::CUDA) {
        tensor_ops::launch_index_copy(ptr<float>(), idx.ptr<int>(),
                                      src.ptr<float>(), shape_.dims().data(),
                                      shape_.rank(), dim, idx.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i) outer *= shape_[i];
        for (size_t i = dim + 1; i < shape_.rank(); ++i) inner *= shape_[i];

        float* dst = ptr<float>();
        const int* indices = idx.ptr<int>();
        const float* src_data = src.ptr<float>();

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < idx.numel(); ++i) {
                int pos = indices[i];
                if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                    LOG_ERROR("Index {} out of bounds for dimension {} of size {}",
                              pos, dim, shape_[dim]);
                    continue;
                }

                // Copy the entire inner dimension
                for (size_t j = 0; j < inner; ++j) {
                    size_t src_idx = o * idx.numel() * inner + i * inner + j;
                    size_t dst_idx = o * shape_[dim] * inner + pos * inner + j;

                    if (src_idx < src.numel() && dst_idx < numel()) {
                        dst[dst_idx] = src_data[src_idx];
                    }
                }
            }
        }
    }

    return *this;
}

Tensor& Tensor::index_add_(int dim, const Tensor& idx, const Tensor& src) {
    if (!is_valid() || !idx.is_valid() || !src.is_valid()) return *this;

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return *this;

    // Ensure indices are 1D
    if (idx.ndim() != 1) {
        LOG_ERROR("index_add_ requires 1D index tensor");
        return *this;
    }

    // For 1D tensor, special case
    if (shape_.rank() == 1 && dim == 0) {
        if (src.ndim() != 1 || src.numel() != idx.numel()) {
            LOG_ERROR("Source must be 1D with same size as indices for 1D tensor");
            return *this;
        }

        if (device_ == Device::CUDA) {
            // Ensure indices and src are on same device
            const int* idx_ptr = nullptr;
            const float* src_ptr = nullptr;
            Tensor idx_temp, src_temp;

            if (idx.device() == device_) {
                idx_ptr = idx.ptr<int>();
            } else {
                idx_temp = idx.to(device_);
                idx_ptr = idx_temp.ptr<int>();
            }

            if (src.device() == device_) {
                src_ptr = src.ptr<float>();
            } else {
                src_temp = src.to(device_);
                src_ptr = src_temp.ptr<float>();
            }

            // Use the dedicated index_add kernel
            tensor_ops::launch_index_add(ptr<float>(), idx_ptr,
                                        src_ptr, shape_.dims().data(),
                                        shape_.rank(), dim, idx.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            float* data = ptr<float>();
            const int* indices = idx.ptr<int>();
            const float* src_data = src.ptr<float>();

            for (size_t i = 0; i < idx.numel(); ++i) {
                int pos = indices[i];
                if (pos < 0) pos += shape_[0];
                if (pos >= 0 && pos < static_cast<int>(shape_[0])) {
                    data[pos] += src_data[i];
                }
            }
        }
        return *this;
    }

    // Multi-dimensional case
    // Validate source shape
    std::vector<size_t> expected_shape = shape_.dims();
    expected_shape[dim] = idx.numel();

    if (src.shape() != TensorShape(expected_shape)) {
        LOG_ERROR("Source shape mismatch in index_add_: expected {}, got {}",
                  TensorShape(expected_shape).str(), src.shape().str());
        return *this;
    }

    if (device_ == Device::CUDA) {
        // Ensure all tensors are on same device
        const int* idx_ptr = nullptr;
        const float* src_ptr = nullptr;
        Tensor idx_temp, src_temp;

        if (idx.device() == device_) {
            idx_ptr = idx.ptr<int>();
        } else {
            idx_temp = idx.to(device_);
            idx_ptr = idx_temp.ptr<int>();
        }

        if (src.device() == device_) {
            src_ptr = src.ptr<float>();
        } else {
            src_temp = src.to(device_);
            src_ptr = src_temp.ptr<float>();
        }

        // Use the dedicated index_add kernel
        tensor_ops::launch_index_add(ptr<float>(), idx_ptr,
                                    src_ptr, shape_.dims().data(),
                                    shape_.rank(), dim, idx.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        size_t outer = 1;
        for (int i = 0; i < dim; ++i) {
            outer *= shape_[i];
        }

        size_t inner = 1;
        for (size_t i = dim + 1; i < shape_.rank(); ++i) {
            inner *= shape_[i];
        }

        float* data = ptr<float>();
        const int* indices = idx.ptr<int>();
        const float* src_data = src.ptr<float>();

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < idx.numel(); ++i) {
                int pos = indices[i];

                // Handle negative indices
                if (pos < 0) pos += static_cast<int>(shape_[dim]);

                // Bounds check
                if (pos < 0 || pos >= static_cast<int>(shape_[dim])) {
                    continue;
                }

                // Add values
                size_t src_base = o * idx.numel() * inner + i * inner;
                size_t dst_base = o * shape_[dim] * inner + pos * inner;

                for (size_t j = 0; j < inner; ++j) {
                    data[dst_base + j] += src_data[src_base + j];
                }
            }
        }
    }

    return *this;
}

Tensor& Tensor::index_put_(const Tensor& idx, const Tensor& vals) {
    if (!is_valid() || !idx.is_valid() || !vals.is_valid()) return *this;

    if (device_ == Device::CUDA) {
        tensor_ops::launch_index_put(ptr<float>(), idx.ptr<int>(),
                                     vals.ptr<float>(), numel(), idx.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        float* data = ptr<float>();
        const int* indices = idx.ptr<int>();
        const float* values = vals.ptr<float>();
        size_t num_elements = numel();

        std::for_each(std::execution::par_unseq,
                     std::views::iota(0uz, idx.numel()).begin(),
                     std::views::iota(0uz, idx.numel()).end(),
                     [data, indices, values, num_elements](size_t i) {
                         int pos = indices[i];
                         if (pos < 0) pos += num_elements;
                         if (pos >= 0 && pos < static_cast<int>(num_elements)) {
                             data[pos] = values[i];
                         }
                     });
    }
    return *this;
}

Tensor& Tensor::index_put_(const std::vector<Tensor>& indices, const Tensor& vals) {
    if (!is_valid() || indices.empty() || !vals.is_valid()) return *this;

    // Single index tensor - use existing implementation
    if (indices.size() == 1) {
        return index_put_(indices[0], vals);
    }

    // For 2D case: indices = [row_indices, col_indices]
    if (indices.size() == 2 && shape_.rank() == 2) {
        // Ensure all tensors on same device
        Tensor row_idx = (indices[0].device() == device_)
            ? indices[0].clone()
            : indices[0].to(device_);
        Tensor col_idx = (indices[1].device() == device_)
            ? indices[1].clone()
            : indices[1].to(device_);
        Tensor vals_same_device = (vals.device() == device_)
            ? vals.clone()
            : vals.to(device_);

        if (row_idx.numel() != col_idx.numel() || row_idx.numel() != vals_same_device.numel()) {
            LOG_ERROR("Index tensors and values must have same number of elements");
            return *this;
        }

        if (device_ == Device::CUDA) {
            // For CUDA, we need a custom kernel
            // For now, fall back to CPU
            auto cpu_tensor = to(Device::CPU);
            auto cpu_row = row_idx.to(Device::CPU);
            auto cpu_col = col_idx.to(Device::CPU);
            auto cpu_vals = vals_same_device.to(Device::CPU);

            const int* row_ptr = cpu_row.ptr<int>();
            const int* col_ptr = cpu_col.ptr<int>();
            const float* val_ptr = cpu_vals.ptr<float>();
            float* data_ptr = cpu_tensor.ptr<float>();

            for (size_t i = 0; i < cpu_row.numel(); ++i) {
                int r = row_ptr[i];
                int c = col_ptr[i];

                // Handle negative indices
                if (r < 0) r += shape_[0];
                if (c < 0) c += shape_[1];

                // Bounds check
                if (r >= 0 && r < static_cast<int>(shape_[0]) &&
                    c >= 0 && c < static_cast<int>(shape_[1])) {
                    data_ptr[r * shape_[1] + c] = val_ptr[i];
                }
            }

            // Copy back to original device
            *this = cpu_tensor.to(device_);
        } else {
            // CPU implementation
            const int* row_ptr = row_idx.ptr<int>();
            const int* col_ptr = col_idx.ptr<int>();
            const float* val_ptr = vals_same_device.ptr<float>();
            float* data_ptr = ptr<float>();

            for (size_t i = 0; i < row_idx.numel(); ++i) {
                int r = row_ptr[i];
                int c = col_ptr[i];

                // Handle negative indices
                if (r < 0) r += shape_[0];
                if (c < 0) c += shape_[1];

                // Bounds check
                if (r >= 0 && r < static_cast<int>(shape_[0]) &&
                    c >= 0 && c < static_cast<int>(shape_[1])) {
                    data_ptr[r * shape_[1] + c] = val_ptr[i];
                }
            }
        }
        return *this;
    }

    LOG_WARN("Multi-dimensional index_put_ not fully implemented for {} dimensions", indices.size());
    return *this;
}

// Nonzero & Count
size_t Tensor::count_nonzero() const {
    if (!is_valid() || numel() == 0) return 0;

    if (device_ == Device::CUDA) {
        size_t count;
        if (dtype_ == DataType::Bool) {
            tensor_ops::launch_count_nonzero_bool(ptr<unsigned char>(), &count, numel(), 0);
        } else {
            tensor_ops::launch_count_nonzero_float(ptr<float>(), &count, numel(), 0);
        }
        cudaDeviceSynchronize();
        return count;
    } else {
        if (dtype_ == DataType::Bool) {
            return std::count(ptr<unsigned char>(), ptr<unsigned char>() + numel(), 1);
        } else {
            return std::count_if(ptr<float>(), ptr<float>() + numel(),
                                [](float x) { return x != 0.0f; });
        }
    }
}

Tensor Tensor::nonzero() const {
    if (!is_valid()) return {};

    size_t count = count_nonzero();
    if (count == 0) return empty({0}, device_, DataType::Int64);

    auto result = empty({count}, device_, DataType::Int64);

    if (device_ == Device::CUDA) {
        if (dtype_ == DataType::Bool) {
            tensor_ops::launch_nonzero_bool(ptr<unsigned char>(),
                                           reinterpret_cast<int64_t*>(result.raw_ptr()),
                                           numel(), count, 0);
        } else {
            tensor_ops::launch_nonzero(ptr<float>(),
                                      reinterpret_cast<int64_t*>(result.raw_ptr()),
                                      numel(), count, 0);
        }
        cudaDeviceSynchronize();
    } else {
        int64_t* indices = reinterpret_cast<int64_t*>(result.raw_ptr());

        if (dtype_ == DataType::Bool) {
            const unsigned char* data = ptr<unsigned char>();
            size_t idx = 0;
            for (size_t i = 0; i < numel(); ++i) {
                if (data[i]) indices[idx++] = i;
            }
        } else {
            const float* data = ptr<float>();
            size_t idx = 0;
            for (size_t i = 0; i < numel(); ++i) {
                if (data[i] != 0.0f) indices[idx++] = i;
            }
        }
    }
    return result;
}

std::vector<Tensor> Tensor::nonzero_split() const {
    std::vector<Tensor> result;
    result.push_back(nonzero());
    return result;
}

// Pythonic Indexing
TensorIndexer Tensor::operator[](const Tensor& idx) {
    std::vector<Tensor> indices;
    indices.reserve(1);
    indices.push_back(idx.clone());
    return TensorIndexer(this, std::move(indices));
}

TensorIndexer Tensor::operator[](const std::vector<Tensor>& idx) {
    std::vector<Tensor> cloned;
    cloned.reserve(idx.size());
    std::ranges::transform(idx, std::back_inserter(cloned),
                          [](const auto& i) { return i.clone(); });
    return TensorIndexer(this, std::move(cloned));
}

MaskedTensorProxy Tensor::operator[](const Tensor& mask) const {
    return MaskedTensorProxy(this, mask.clone());
}

// Element Access
float& Tensor::at(std::initializer_list<size_t> indices) {
    static float dummy = 0;
    if (dtype_ != DataType::Float32 || indices.size() != shape_.rank()) {
        LOG_ERROR("at() type or dimension mismatch");
        return dummy;
    }

    // Convert initializer_list to vector for easier access
    std::vector<size_t> idx_vec(indices);

    // Calculate linear index in row-major order
    size_t linear_idx = 0;
    size_t stride = 1;

    // Work backwards through dimensions (row-major)
    for (int i = shape_.rank() - 1; i >= 0; --i) {
        if (idx_vec[i] >= shape_[i]) {
            LOG_ERROR("Index {} out of bounds for dimension {} with size {}",
                      idx_vec[i], i, shape_[i]);
            return dummy;
        }
        linear_idx += idx_vec[i] * stride;
        stride *= shape_[i];
    }

    if (device_ == Device::CUDA) {
        LOG_ERROR("Cannot get mutable reference to CUDA tensor element");
        return dummy;
    }
    return ptr<float>()[linear_idx];
}

float Tensor::at(std::initializer_list<size_t> indices) const {
    if (dtype_ != DataType::Float32 || indices.size() != shape_.rank()) {
        LOG_ERROR("at() type or dimension mismatch");
        return 0;
    }

    // Convert initializer_list to vector for easier access
    std::vector<size_t> idx_vec(indices);

    // Calculate linear index in row-major order
    size_t linear_idx = 0;
    size_t stride = 1;

    // Work backwards through dimensions (row-major)
    for (int i = shape_.rank() - 1; i >= 0; --i) {
        if (idx_vec[i] >= shape_[i]) {
            LOG_ERROR("Index {} out of bounds for dimension {} with size {}",
                      idx_vec[i], i, shape_[i]);
            return 0;
        }
        linear_idx += idx_vec[i] * stride;
        stride *= shape_[i];
    }

    if (device_ == Device::CUDA) {
        float value;
        cudaMemcpy(&value, ptr<float>() + linear_idx, sizeof(float), cudaMemcpyDeviceToHost);
        return value;
    }
    return ptr<float>()[linear_idx];
}
// From Vector
template<typename T>
static Tensor from_vector_impl(const std::vector<T>& data, TensorShape shape,
                               Device device, DataType dtype) {
    if (shape.elements() != data.size()) return {};
    auto t = Tensor::empty(shape, device, dtype);
    if (!t.is_valid() || t.numel() == 0) return t;

    // Only copy if there's actually data to copy
    if (t.numel() > 0 && data.data() != nullptr) {
        if (device == Device::CUDA) {
            cudaMemcpy(t.raw_ptr(), data.data(), t.bytes(), cudaMemcpyHostToDevice);
        } else {
            std::memcpy(t.raw_ptr(), data.data(), t.bytes());
        }
    }
    return t;
}

Tensor Tensor::from_vector(const std::vector<float>& data, TensorShape shape, Device device) {
    return from_vector_impl(data, shape, device, DataType::Float32);
}

Tensor Tensor::from_vector(const std::vector<int>& data, TensorShape shape, Device device) {
    return from_vector_impl(data, shape, device, DataType::Int32);
}

Tensor Tensor::from_vector(const std::vector<bool>& data, TensorShape shape, Device device) {
    if (shape.elements() != data.size()) return {};

    std::vector<unsigned char> bytes(data.size());
    std::ranges::transform(data, bytes.begin(),
                          [](bool b) { return b ? 1 : 0; });

    return from_vector_impl(bytes, shape, device, DataType::Bool);
}

// Conversion
std::vector<int> Tensor::to_vector_int() const {
    if (dtype_ != DataType::Int32 || !is_valid() || numel() == 0) return {};

    std::vector<int> result(numel());
    if (device_ == Device::CUDA) {
        cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(result.data(), data_, bytes());
    }
    return result;
}

std::vector<bool> Tensor::to_vector_bool() const {
    if (dtype_ != DataType::Bool || !is_valid() || numel() == 0) return {};

    std::vector<unsigned char> bytes(numel());
    if (device_ == Device::CUDA) {
        cudaMemcpy(bytes.data(), data_, this->bytes(), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(bytes.data(), data_, this->bytes());
    }

    std::vector<bool> result(numel());
    std::ranges::transform(bytes, result.begin(),
                          [](auto x) { return x != 0; });
    return result;
}

void Tensor::set_bool(std::initializer_list<size_t> indices, bool value) {
    if (dtype_ != DataType::Bool) return;

    size_t idx = 0, stride = 1;
    auto it = indices.end();
    for (int i = shape_.rank() - 1; i >= 0; --i) {
        --it;
        idx += (*it) * stride;
        stride *= shape_[i];
    }

    unsigned char val = value ? 1 : 0;
    if (device_ == Device::CUDA) {
        cudaMemcpy(ptr<unsigned char>() + idx, &val, 1, cudaMemcpyHostToDevice);
    } else {
        ptr<unsigned char>()[idx] = val;
    }
}

bool Tensor::get_bool(std::initializer_list<size_t> indices) const {
    if (dtype_ != DataType::Bool) return false;

    size_t idx = 0, stride = 1;
    auto it = indices.end();
    for (int i = shape_.rank() - 1; i >= 0; --i) {
        --it;
        idx += (*it) * stride;
        stride *= shape_[i];
    }

    if (device_ == Device::CUDA) {
        unsigned char val;
        cudaMemcpy(&val, ptr<unsigned char>() + idx, 1, cudaMemcpyDeviceToHost);
        return val != 0;
    }
    return ptr<unsigned char>()[idx] != 0;
}

// Proxy Implementations
void MaskedTensorProxy::operator=(float value) {
    const_cast<Tensor*>(tensor_)->masked_fill_(mask_, value);
}

void MaskedTensorProxy::operator=(const Tensor& other) {
    auto selected = tensor_->masked_select(mask_);
    if (selected.numel() != other.numel()) return;

    if (tensor_->device() == Device::CUDA) {
        tensor_ops::launch_masked_scatter(const_cast<Tensor*>(tensor_)->ptr<float>(),
                                         mask_.ptr<unsigned char>(), other.ptr<float>(),
                                         tensor_->numel(), other.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        float* data = const_cast<Tensor*>(tensor_)->ptr<float>();
        const unsigned char* mask = mask_.ptr<unsigned char>();
        const float* src = other.ptr<float>();

        size_t src_idx = 0;
        for (size_t i = 0; i < tensor_->numel() && src_idx < other.numel(); ++i) {
            if (mask[i]) data[i] = src[src_idx++];
        }
    }
}

MaskedTensorProxy::operator Tensor() const {
    return tensor_->masked_select(mask_);
}

void TensorIndexer::operator=(float value) {
    if (indices_.size() == 1) {
        if (indices_[0].dtype() == DataType::Bool) {
            tensor_->masked_fill_(indices_[0], value);
        } else {
            tensor_->scatter_(0, indices_[0], value);
        }
    }
}

void TensorIndexer::operator=(const Tensor& other) {
    if (indices_.size() == 1) {
        if (indices_[0].dtype() == DataType::Bool) {
            MaskedTensorProxy proxy(tensor_, std::move(indices_[0]));
            proxy = other;
        } else {
            tensor_->scatter_(0, indices_[0], other);
        }
    }
}

TensorIndexer::operator Tensor() const {
    if (indices_.size() == 1) {
        if (indices_[0].dtype() == DataType::Bool) {
            return tensor_->masked_select(indices_[0]);
        } else {
            return indices_[0].ndim() == 1 ?
                   tensor_->index_select(0, indices_[0]) :
                   tensor_->take(indices_[0]);
        }
    }
    return Tensor();
}

#undef CHECK_CUDA

} // namespace gs