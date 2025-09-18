/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
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

// ============= Unified Comparison Template =============
template<typename Op>
static Tensor compare_tensors(const Tensor& a, const Tensor& b, Op op,
                              void(*cuda_fn)(const float*, const float*, unsigned char*,
                                           const size_t*, const size_t*, const size_t*,
                                           size_t, size_t, size_t, size_t, cudaStream_t)) {
    if (!a.is_valid() || !b.is_valid() || a.device() != b.device()) return {};

    auto out_shape = BroadcastHelper::broadcast_shape(a.shape(), b.shape());
    if (!out_shape.is_initialized()) return {};

    auto result = Tensor::empty(out_shape, a.device(), DataType::Bool);

    if (a.device() == Device::CUDA) {
        cuda_fn(a.template ptr<float>(), b.template ptr<float>(),
                result.template ptr<unsigned char>(),
                a.shape().dims().data(), b.shape().dims().data(),
                out_shape.dims().data(), a.shape().rank(),
                b.shape().rank(), out_shape.rank(),
                out_shape.elements(), 0);
        cudaDeviceSynchronize();
    } else {
        // Manual broadcasting for mixed types (float -> unsigned char)
        const float* a_data = a.template ptr<float>();
        const float* b_data = b.template ptr<float>();
        unsigned char* c_data = result.template ptr<unsigned char>();

        size_t total = out_shape.elements();
        auto a_shape = a.shape().dims();
        auto b_shape = b.shape().dims();
        auto c_shape = out_shape.dims();

        if (a_shape == b_shape) {
            // Fast path - no broadcasting needed
            std::transform(std::execution::par_unseq,
                          a_data, a_data + total, b_data, c_data,
                          [op](float x, float y) -> unsigned char {
                              return op(x, y) ? 1 : 0;
                          });
        } else {
            // Broadcasting needed
            #ifdef __cpp_lib_parallel_algorithm
            std::for_each(std::execution::par_unseq,
                         std::views::iota(0uz, total).begin(),
                         std::views::iota(0uz, total).end(),
                         [&](size_t i) {
                size_t a_idx = BroadcastUtil::map_broadcast_index(i, c_shape, a_shape);
                size_t b_idx = BroadcastUtil::map_broadcast_index(i, c_shape, b_shape);
                c_data[i] = op(a_data[a_idx], b_data[b_idx]) ? 1 : 0;
            });
            #else
            for (size_t i = 0; i < total; ++i) {
                size_t a_idx = BroadcastUtil::map_broadcast_index(i, c_shape, a_shape);
                size_t b_idx = BroadcastUtil::map_broadcast_index(i, c_shape, b_shape);
                c_data[i] = op(a_data[a_idx], b_data[b_idx]) ? 1 : 0;
            }
            #endif
        }
    }
    return result;
}

template<typename Op>
static Tensor compare_scalar(const Tensor& tensor, float scalar, Op op,
                            void(*cuda_fn)(const float*, float, unsigned char*, size_t, cudaStream_t)) {
    if (!tensor.is_valid()) return {};

    auto result = Tensor::empty(tensor.shape(), tensor.device(), DataType::Bool);

    if (tensor.device() == Device::CUDA) {
        cuda_fn(tensor.template ptr<float>(), scalar, result.template ptr<unsigned char>(),
                tensor.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        std::transform(std::execution::par_unseq,
                      tensor.template ptr<float>(),
                      tensor.template ptr<float>() + tensor.numel(),
                      result.template ptr<unsigned char>(),
                      [op, scalar](float x) -> unsigned char {
                          return op(x, scalar) ? 1 : 0;
                      });
    }
    return result;
}

// ============= Comparison Operations (10 lines each instead of 20+) =============
Tensor Tensor::eq(const Tensor& o) const {
    return compare_tensors(*this, o, std::equal_to<float>{},
                          tensor_ops::launch_compare_eq);
}

Tensor Tensor::eq(float v) const {
    return compare_scalar(*this, v, std::equal_to<float>{},
                          tensor_ops::launch_compare_scalar_eq);
}

Tensor Tensor::lt(const Tensor& o) const {
    return compare_tensors(*this, o, std::less<float>{},
                          tensor_ops::launch_compare_lt);
}

Tensor Tensor::lt(float v) const {
    return compare_scalar(*this, v, std::less<float>{},
                          tensor_ops::launch_compare_scalar_lt);
}

Tensor Tensor::gt(const Tensor& o) const {
    return compare_tensors(*this, o, std::greater<float>{},
                          tensor_ops::launch_compare_gt);
}

Tensor Tensor::gt(float v) const {
    return compare_scalar(*this, v, std::greater<float>{},
                          tensor_ops::launch_compare_scalar_gt);
}

// Derived operations using logical_not
Tensor Tensor::ne(const Tensor& o) const { return eq(o).logical_not(); }
Tensor Tensor::ne(float v) const { return eq(v).logical_not(); }
Tensor Tensor::le(const Tensor& o) const { return gt(o).logical_not(); }
Tensor Tensor::le(float v) const { return gt(v).logical_not(); }
Tensor Tensor::ge(const Tensor& o) const { return lt(o).logical_not(); }
Tensor Tensor::ge(float v) const { return lt(v).logical_not(); }

// ============= Logical Operations - Unified Template =============
template<typename Op>
static Tensor logical_impl(const Tensor& a, const Tensor& b, Op op,
                          void(*cuda_fn)(const unsigned char*, const unsigned char*, unsigned char*,
                                       const size_t*, const size_t*, const size_t*,
                                       size_t, size_t, size_t, size_t, cudaStream_t)) {
    if (!a.is_valid() || !b.is_valid() ||
        a.dtype() != DataType::Bool || b.dtype() != DataType::Bool ||
        a.device() != b.device()) return {};

    auto out_shape = BroadcastHelper::broadcast_shape(a.shape(), b.shape());
    if (!out_shape.is_initialized()) return {};

    auto result = Tensor::empty(out_shape, a.device(), DataType::Bool);

    if (a.device() == Device::CUDA) {
        cuda_fn(a.template ptr<unsigned char>(), b.template ptr<unsigned char>(),
                result.template ptr<unsigned char>(),
                a.shape().dims().data(), b.shape().dims().data(),
                out_shape.dims().data(), a.shape().rank(),
                b.shape().rank(), out_shape.rank(),
                out_shape.elements(), 0);
        cudaDeviceSynchronize();
    } else {
        broadcast_op_cpu(a.template ptr<unsigned char>(), b.template ptr<unsigned char>(),
                        result.template ptr<unsigned char>(),
                        a.shape().dims(), b.shape().dims(),
                        out_shape.dims(), op);
    }
    return result;
}

Tensor Tensor::logical_and(const Tensor& o) const {
    auto op = [](unsigned char a, unsigned char b) -> unsigned char {
        return (a && b) ? 1 : 0;
    };
    return logical_impl(*this, o, op, tensor_ops::launch_logical_and);
}

Tensor Tensor::logical_or(const Tensor& o) const {
    auto op = [](unsigned char a, unsigned char b) -> unsigned char {
        return (a || b) ? 1 : 0;
    };
    return logical_impl(*this, o, op, tensor_ops::launch_logical_or);
}

Tensor Tensor::logical_xor(const Tensor& o) const {
    auto op = [](unsigned char a, unsigned char b) -> unsigned char {
        return ((a != 0) != (b != 0)) ? 1 : 0;
    };
    return logical_impl(*this, o, op, tensor_ops::launch_logical_xor);
}

Tensor Tensor::logical_not() const {
    if (!is_valid() || dtype_ != DataType::Bool) return {};

    auto result = empty(shape_, device_, DataType::Bool);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_logical_not(ptr<unsigned char>(),
                                       result.ptr<unsigned char>(), numel(), 0);
        cudaDeviceSynchronize();
    } else {
        std::transform(std::execution::par_unseq,
                      ptr<unsigned char>(), ptr<unsigned char>() + numel(),
                      result.ptr<unsigned char>(),
                      [](unsigned char x) { return !x; });
    }
    return result;
}

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

        // Use parallel scan for CPU too
        std::copy_if(std::execution::par_unseq,
                    std::views::iota(0uz, numel()).begin(),
                    std::views::iota(0uz, numel()).end(),
                    dst,
                    [src, msk](size_t i) { return msk[i] ? src[i] : 0.0f; });
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

Tensor Tensor::where(const Tensor& condition, const Tensor& other) const {
    return Tensor::where(condition, *this, other);
}

Tensor Tensor::where(const Tensor& cond, const Tensor& x, const Tensor& y) {
    if (!cond.is_valid() || !x.is_valid() || !y.is_valid() ||
        cond.dtype() != DataType::Bool || x.device() != y.device() ||
        x.device() != cond.device() || x.dtype() != y.dtype()) return {};

    auto shape = BroadcastHelper::broadcast_shape(
        BroadcastHelper::broadcast_shape(x.shape(), y.shape()), cond.shape());
    if (!shape.is_initialized()) return {};

    auto result = empty(shape, x.device(), x.dtype());

    if (x.device() == Device::CUDA) {
        tensor_ops::launch_where(cond.ptr<unsigned char>(),
                               x.ptr<float>(), y.ptr<float>(), result.ptr<float>(),
                               cond.shape().dims().data(), x.shape().dims().data(),
                               y.shape().dims().data(), shape.dims().data(),
                               cond.shape().rank(), x.shape().rank(),
                               y.shape().rank(), shape.rank(),
                               shape.elements(), 0);
        cudaDeviceSynchronize();
    } else {
        // Use broadcasting infrastructure
        BroadcastIterator c_it(cond.shape(), shape);
        BroadcastIterator x_it(x.shape(), shape);
        BroadcastIterator y_it(y.shape(), shape);

        const unsigned char* c_data = cond.ptr<unsigned char>();
        const float* x_data = x.ptr<float>();
        const float* y_data = y.ptr<float>();
        float* r_data = result.ptr<float>();

        for (size_t i = 0; !c_it.done(); ++i, c_it.next(), x_it.next(), y_it.next()) {
            r_data[i] = c_data[c_it.index()] ? x_data[x_it.index()] : y_data[y_it.index()];
        }
    }
    return result;
}

// ============= Indexing Operations - Simplified with Templates =============
template<typename IndexOp>
static Tensor index_op_impl(const Tensor& t, int dim, const Tensor& idx,
                            BoundaryMode mode, IndexOp op) {
    if (!t.is_valid() || !idx.is_valid() || idx.ndim() != 1) return {};

    dim = (dim < 0) ? t.shape().rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(t.shape().rank())) return {};

    return op(t, dim, idx, mode);
}

Tensor Tensor::index_select(int dim, const Tensor& indices, BoundaryMode mode) const {
    return index_op_impl(*this, dim, indices, mode,
        [](const Tensor& t, int d, const Tensor& idx, BoundaryMode m) {
            auto dims = t.shape().dims();
            dims[d] = idx.numel();
            auto result = empty(TensorShape(dims), t.device(), t.dtype());

            if (t.device() == Device::CUDA) {
                tensor_ops::launch_index_select(t.ptr<float>(), idx.ptr<int>(),
                    result.ptr<float>(), t.shape().dims().data(),
                    t.shape().rank(), d, idx.numel(), static_cast<int>(m), 0);
                cudaDeviceSynchronize();
            } else {
                // Simplified CPU implementation using ranges
                size_t outer = 1, inner = 1;
                for (int i = 0; i < d; ++i) outer *= t.shape()[i];
                for (size_t i = d + 1; i < t.shape().rank(); ++i) inner *= t.shape()[i];

                const float* src = t.ptr<float>();
                float* dst = result.ptr<float>();
                const int* indices = idx.ptr<int>();

                auto process_idx = [&](int sel) -> int {
                    if (m == BoundaryMode::Clamp) {
                        return std::clamp(sel, 0, static_cast<int>(t.shape()[d]) - 1);
                    } else if (m == BoundaryMode::Wrap) {
                        return ((sel % static_cast<int>(t.shape()[d])) + t.shape()[d]) % t.shape()[d];
                    }
                    return sel;
                };

                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < idx.numel(); ++i) {
                        int sel = process_idx(indices[i]);
                        if (sel >= 0 && sel < static_cast<int>(t.shape()[d])) {
                            std::copy_n(src + (o * t.shape()[d] + sel) * inner,
                                       inner,
                                       dst + (o * idx.numel() + i) * inner);
                        }
                    }
                }
            }
            return result;
        });
}

Tensor Tensor::index_select(int dim, const Tensor& indices) const {
    return index_select(dim, indices, BoundaryMode::Assert);
}

Tensor Tensor::gather(int dim, const Tensor& indices, BoundaryMode mode) const {
    if (!is_valid() || !indices.is_valid()) return {};

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return {};

    // The result shape should match indices shape
    auto result = empty(indices.shape(), device_, dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_gather(ptr<float>(), indices.ptr<int>(),
            result.ptr<float>(), shape_.dims().data(),
            indices.shape().dims().data(), shape_.rank(), dim,
            indices.numel(), static_cast<int>(mode), 0);
        cudaDeviceSynchronize();
    } else {
        // CPU implementation for gather
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        const int* idx_data = indices.ptr<int>();

        // For the general gather operation:
        // The output shape matches the indices shape
        // For each position in output/indices, we gather from input along dim

        // Calculate strides for all tensors
        std::vector<size_t> src_strides(shape_.rank());
        src_strides[shape_.rank() - 1] = 1;
        for (int i = shape_.rank() - 2; i >= 0; --i) {
            src_strides[i] = src_strides[i + 1] * shape_[i + 1];
        }

        std::vector<size_t> idx_strides(indices.shape().rank());
        idx_strides[indices.shape().rank() - 1] = 1;
        for (int i = indices.shape().rank() - 2; i >= 0; --i) {
            idx_strides[i] = idx_strides[i + 1] * indices.shape()[i + 1];
        }

        // Process each element in the output
        for (size_t out_idx = 0; out_idx < indices.numel(); ++out_idx) {
            // Get multi-dimensional coordinates in the indices/output tensor
            std::vector<size_t> out_coords(indices.shape().rank());
            size_t temp = out_idx;
            for (size_t d = 0; d < indices.shape().rank(); ++d) {
                out_coords[d] = temp / idx_strides[d];
                temp %= idx_strides[d];
            }

            // Get the index to gather
            int gather_idx = idx_data[out_idx];

            // Apply boundary mode
            if (mode == BoundaryMode::Clamp) {
                gather_idx = std::clamp(gather_idx, 0, static_cast<int>(shape_[dim]) - 1);
            } else if (mode == BoundaryMode::Wrap) {
                gather_idx = ((gather_idx % static_cast<int>(shape_[dim])) + shape_[dim]) % shape_[dim];
            } else if (gather_idx < 0 || gather_idx >= static_cast<int>(shape_[dim])) {
                dst[out_idx] = 0.0f;
                continue;
            }

            // Build source coordinates
            // Use output coordinates for all dimensions except dim
            // For dim, use the gathered index
            size_t src_idx = 0;
            for (size_t d = 0; d < shape_.rank(); ++d) {
                size_t coord;
                if (static_cast<int>(d) == dim) {
                    coord = gather_idx;
                } else if (d < indices.shape().rank()) {
                    coord = out_coords[d];
                } else {
                    coord = 0;
                }

                // Bounds check
                if (coord >= shape_[d]) {
                    dst[out_idx] = 0.0f;
                    break;
                }

                src_idx += coord * src_strides[d];
            }

            // Only set if we didn't break out of the loop
            if (src_idx < numel()) {
                dst[out_idx] = src[src_idx];
            }
        }
    }

    return result;
}

Tensor Tensor::gather(int dim, const Tensor& indices) const {
    return gather(dim, indices, BoundaryMode::Assert);
}

Tensor Tensor::take(const Tensor& indices) const {
    if (!is_valid() || !indices.is_valid()) return {};

    auto flat = flatten();
    auto result = empty(indices.shape(), device_, dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_take(flat.ptr<float>(), indices.ptr<int>(),
                               result.ptr<float>(), flat.numel(), indices.numel(), 0);
        cudaDeviceSynchronize();
    } else {
        const float* src = flat.ptr<float>();
        float* dst = result.ptr<float>();
        const int* idx = indices.ptr<int>();
        size_t total = flat.numel();

        std::transform(std::execution::par_unseq,
                      idx, idx + indices.numel(), dst,
                      [src, total](int pos) {
                          if (pos < 0) pos += total;
                          return (pos >= 0 && pos < static_cast<int>(total)) ? src[pos] : 0.0f;
                      });
    }
    return result;
}

// ============= Scatter Operations - Simplified =============
template<typename Op>
static Tensor& scatter_impl(Tensor& t, int dim, const Tensor& idx,
                           const Tensor& src, Op op, int mode) {
    if (!t.is_valid() || !idx.is_valid() || !src.is_valid()) return t;

    dim = (dim < 0) ? t.shape().rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(t.shape().rank())) return t;

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
                    op(dst[dst_idx], src_data[src_idx]);
                }
            }
        }
    }
    return t;
}

Tensor& Tensor::scatter_(int dim, const Tensor& idx, const Tensor& src, ScatterMode mode) {
    using OpFunc = std::function<void(float&, float)>;
    OpFunc op;

    switch (mode) {
        case ScatterMode::Add:
            op = [](float& d, float s) { d += s; };
            break;
        case ScatterMode::Multiply:
            op = [](float& d, float s) { d *= s; };
            break;
        case ScatterMode::Max:
            op = [](float& d, float s) { d = std::max(d, s); };
            break;
        case ScatterMode::Min:
            op = [](float& d, float s) { d = std::min(d, s); };
            break;
        default:
            op = [](float& d, float s) { d = s; };
            break;
    }

    return scatter_impl(*this, dim, idx, src, op, static_cast<int>(mode));
}

Tensor& Tensor::scatter_(int dim, const Tensor& idx, float val, ScatterMode mode) {
    auto src = full(idx.shape(), val, device_, dtype_);
    return scatter_(dim, idx, src, mode);
}

Tensor& Tensor::index_fill_(int dim, const Tensor& idx, float val) {
    return scatter_(dim, idx, val, ScatterMode::None);
}

Tensor& Tensor::index_copy_(int dim, const Tensor& idx, const Tensor& src) {
    return scatter_(dim, idx, src, ScatterMode::None);
}

Tensor& Tensor::index_add_(int dim, const Tensor& idx, const Tensor& src) {
    return scatter_(dim, idx, src, ScatterMode::Add);
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
    return (indices.size() == 1) ? index_put_(indices[0], vals) : *this;
}

// ============= Nonzero & Count =============
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
            auto nonzero_indices = std::views::iota(0uz, numel())
                                 | std::views::filter([data](size_t i) { return data[i]; });
            std::ranges::copy(nonzero_indices, indices);
        } else {
            const float* data = ptr<float>();
            auto nonzero_indices = std::views::iota(0uz, numel())
                                 | std::views::filter([data](size_t i) { return data[i] != 0.0f; });
            std::ranges::copy(nonzero_indices, indices);
        }
    }
    return result;
}

std::vector<Tensor> Tensor::nonzero_split() const {
    std::vector<Tensor> result;
    result.push_back(nonzero());
    return result;
}

bool Tensor::any() const {
    return dtype_ == DataType::Bool && count_nonzero() > 0;
}

bool Tensor::all() const {
    return dtype_ == DataType::Bool && count_nonzero() == numel();
}

// ============= Pythonic Indexing =============
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

// ============= Element Access =============
float& Tensor::at(std::initializer_list<size_t> indices) {
    static float dummy = 0;
    if (dtype_ != DataType::Float32 || indices.size() != shape_.rank()) return dummy;

    size_t idx = 0, stride = 1;
    auto it = indices.end();
    for (int i = shape_.rank() - 1; i >= 0; --i) {
        --it;
        if (*it >= shape_[i]) return dummy;
        idx += (*it) * stride;
        stride *= shape_[i];
    }

    if (device_ == Device::CUDA) return dummy;
    return ptr<float>()[idx];
}

float Tensor::at(std::initializer_list<size_t> indices) const {
    if (dtype_ != DataType::Float32 || indices.size() != shape_.rank()) return 0;

    size_t idx = 0, stride = 1;
    auto it = indices.end();
    for (int i = shape_.rank() - 1; i >= 0; --i) {
        --it;
        if (*it >= shape_[i]) return 0;
        idx += (*it) * stride;
        stride *= shape_[i];
    }

    if (device_ == Device::CUDA) {
        float value;
        cudaMemcpy(&value, ptr<float>() + idx, sizeof(float), cudaMemcpyDeviceToHost);
        return value;
    }
    return ptr<float>()[idx];
}

// ============= Boolean Operations =============
Tensor Tensor::full_bool(TensorShape shape, bool value, Device device) {
    auto t = empty(shape, device, DataType::Bool);
    if (t.is_valid() && t.numel() > 0) {
        unsigned char fill = value ? 1 : 0;
        if (device == Device::CUDA) {
            cudaMemset(t.data_, fill, t.bytes());
        } else {
            std::memset(t.data_, fill, t.bytes());
        }
    }
    return t;
}

Tensor Tensor::zeros_bool(TensorShape shape, Device device) {
    return full_bool(shape, false, device);
}

Tensor Tensor::ones_bool(TensorShape shape, Device device) {
    return full_bool(shape, true, device);
}

// ============= From Vector =============
template<typename T>
static Tensor from_vector_impl(const std::vector<T>& data, TensorShape shape,
                               Device device, DataType dtype) {
    if (shape.elements() != data.size()) return {};
    auto t = Tensor::empty(shape, device, dtype);
    if (!t.is_valid() || t.numel() == 0) return t;

    if (device == Device::CUDA) {
        cudaMemcpy(t.raw_ptr(), data.data(), t.bytes(), cudaMemcpyHostToDevice);
    } else {
        std::memcpy(t.raw_ptr(), data.data(), t.bytes());
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

// ============= Conversion =============
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

// ============= Proxy Implementations =============
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