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

// Logical not is a special case - not a binary operation
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

// Masking Operations
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

Tensor Tensor::where(const Tensor& condition, const Tensor& other) const {
    return Tensor::where(condition, *this, other);
}

Tensor Tensor::where(const Tensor& cond, const Tensor& x, const Tensor& y) {
    if (!cond.is_valid() || !x.is_valid() || !y.is_valid() ||
        cond.dtype() != DataType::Bool || x.device() != y.device() ||
        x.device() != cond.device() || x.dtype() != y.dtype()) return {};

    // Get broadcast shape for all three tensors
    auto xy_shape = broadcast::shape(x.shape().dims(), y.shape().dims());
    if (xy_shape.empty()) return {};

    auto final_shape = broadcast::shape(xy_shape, cond.shape().dims());
    if (final_shape.empty()) return {};

    TensorShape shape(final_shape);
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
        // Simple CPU implementation using broadcast::index
        const unsigned char* c_data = cond.ptr<unsigned char>();
        const float* x_data = x.ptr<float>();
        const float* y_data = y.ptr<float>();
        float* r_data = result.ptr<float>();

        auto c_shape = cond.shape().dims();
        auto x_shape = x.shape().dims();
        auto y_shape = y.shape().dims();
        auto r_shape = shape.dims();

        #pragma omp parallel for if(shape.elements() > 1024)
        for (size_t i = 0; i < shape.elements(); ++i) {
            size_t c_idx = broadcast::index(i, r_shape, c_shape);
            size_t x_idx = broadcast::index(i, r_shape, x_shape);
            size_t y_idx = broadcast::index(i, r_shape, y_shape);
            r_data[i] = c_data[c_idx] ? x_data[x_idx] : y_data[y_idx];
        }
    }
    return result;
}

// Indexing Operations
Tensor Tensor::index_select(int dim, const Tensor& indices) const {
    return index_select(dim, indices, BoundaryMode::Assert);
}

Tensor Tensor::index_select(int dim, const Tensor& indices, BoundaryMode mode) const {
    if (!is_valid() || !indices.is_valid() || indices.ndim() != 1) return {};

    dim = (dim < 0) ? shape_.rank() + dim : dim;
    if (dim < 0 || dim >= static_cast<int>(shape_.rank())) return {};

    auto dims = shape_.dims();
    dims[dim] = indices.numel();
    auto result = empty(TensorShape(dims), device_, dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_index_select(ptr<float>(), indices.ptr<int>(),
            result.ptr<float>(), shape_.dims().data(),
            shape_.rank(), dim, indices.numel(), static_cast<int>(mode), 0);
        cudaDeviceSynchronize();
    } else {
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i) outer *= shape_[i];
        for (size_t i = dim + 1; i < shape_.rank(); ++i) inner *= shape_[i];

        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        const int* idx = indices.ptr<int>();

        auto process_idx = [&](int sel) -> int {
            if (mode == BoundaryMode::Clamp) {
                return std::clamp(sel, 0, static_cast<int>(shape_[dim]) - 1);
            } else if (mode == BoundaryMode::Wrap) {
                return ((sel % static_cast<int>(shape_[dim])) + shape_[dim]) % shape_[dim];
            }
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

    auto result = empty(indices.shape(), device_, dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_gather(ptr<float>(), indices.ptr<int>(),
            result.ptr<float>(), shape_.dims().data(),
            indices.shape().dims().data(), shape_.rank(), dim,
            indices.numel(), static_cast<int>(mode), 0);
        cudaDeviceSynchronize();
    } else {
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        const int* idx_data = indices.ptr<int>();

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

Tensor Tensor::take(const Tensor& indices) const {
    if (!is_valid() || !indices.is_valid()) return {};

    // Use the new flatten from shape_ops!
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

// Scatter Operations
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

bool Tensor::any() const {
    return dtype_ == DataType::Bool && count_nonzero() > 0;
}

bool Tensor::all() const {
    return dtype_ == DataType::Bool && count_nonzero() == numel();
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

// Boolean Operations
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

// From Vector
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
