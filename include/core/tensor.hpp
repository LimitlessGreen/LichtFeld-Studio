/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <optional>
#include <functional>
#include <initializer_list>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <concepts>
#include <type_traits>
#include <utility>
#include <span>
#include <variant>
#include <tuple>
#include <limits>
#include <array>

namespace gs {

    // Forward declarations
    class Tensor;
    class TensorError;
    class TensorIndexer;
    class MaskedTensorProxy;

    enum class Device : uint8_t {
        CPU = 0,
        CUDA = 1
    };

    enum class DataType : uint8_t {
        Float32 = 0,
        Float16 = 1,
        Int32 = 2,
        Int64 = 3,
        UInt8 = 4,
        Bool = 5
    };

    constexpr size_t dtype_size(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
        case DataType::Bool: return 1;
        default: return 0;
        }
    }

    inline const char* dtype_name(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        case DataType::Bool: return "bool";
        default: return "unknown";
        }
    }

    inline const char* device_name(Device device) {
        return device == Device::CPU ? "cpu" : "cuda";
    }

    inline DataType promote_types(DataType a, DataType b) {
        if (a == b) return a;

        if (a == DataType::Bool) {
            if (b == DataType::Float32 || b == DataType::Float16) return b;
            if (b == DataType::Int32 || b == DataType::Int64) return b;
            return DataType::Float32;
        }
        if (b == DataType::Bool) {
            if (a == DataType::Float32 || a == DataType::Float16) return a;
            if (a == DataType::Int32 || a == DataType::Int64) return a;
            return DataType::Float32;
        }

        if ((a == DataType::Int32 || a == DataType::Int64) &&
            (b == DataType::Float32 || b == DataType::Float16)) {
            return (b == DataType::Float16) ? DataType::Float16 : DataType::Float32;
        }
        if ((b == DataType::Int32 || b == DataType::Int64) &&
            (a == DataType::Float32 || a == DataType::Float16)) {
            return (a == DataType::Float16) ? DataType::Float16 : DataType::Float32;
        }

        if ((a == DataType::Int32 && b == DataType::Int64) ||
            (a == DataType::Int64 && b == DataType::Int32)) {
            return DataType::Int64;
        }

        if ((a == DataType::Float16 && b == DataType::Float32) ||
            (a == DataType::Float32 && b == DataType::Float16)) {
            return DataType::Float32;
        }

        return DataType::Float32;
    }

    enum class BoundaryMode : uint8_t {
        Assert = 0,
        Clamp = 1,
        Wrap = 2
    };

    enum class ScatterMode : uint8_t {
        None = 0,
        Add = 1,
        Multiply = 2,
        Max = 3,
        Min = 4
    };

    enum class BinaryOp : uint8_t {
        Add = 0, Sub = 1, Mul = 2, Div = 3, Pow = 4, Mod = 5,
        Equal = 6, NotEqual = 7, Less = 8, LessEqual = 9, Greater = 10, GreaterEqual = 11,
        LogicalAnd = 12, LogicalOr = 13, LogicalXor = 14,
        Maximum = 15, Minimum = 16,
        BitwiseAnd = 17, BitwiseOr = 18, BitwiseXor = 19, LeftShift = 20, RightShift = 21
    };

    enum class UnaryOp : uint8_t {
        Neg = 0, Abs = 1, Sign = 2, Reciprocal = 3,
        Exp = 4, Exp2 = 5, Log = 6, Log2 = 7, Log10 = 8, Log1p = 9,
        Sqrt = 10, Rsqrt = 11, Square = 12,
        Sin = 13, Cos = 14, Tan = 15, Asin = 16, Acos = 17, Atan = 18,
        Sinh = 19, Cosh = 20, Tanh = 21,
        Sigmoid = 22, Relu = 23, Gelu = 24, Swish = 25,
        Floor = 26, Ceil = 27, Round = 28, Trunc = 29,
        IsNan = 30, IsInf = 31, IsFinite = 32, LogicalNot = 33,
        Normalize = 34, Logit = 35
    };

    enum class ReduceOp : uint8_t {
        Sum = 0, Mean = 1, Max = 2, Min = 3, Prod = 4,
        Any = 5, All = 6, Std = 7, Var = 8,
        Argmax = 9, Argmin = 10, CountNonzero = 11,
        Norm = 12
    };

    enum class TernaryOp : uint8_t {
        Where = 0, MulAdd = 1, Clamp = 2
    };

    enum class MovementOp : uint8_t {
        Reshape = 0, Permute = 1, Expand = 2, Pad = 3, Shrink = 4,
        Flip = 5, Transpose = 6, Squeeze = 7, Unsqueeze = 8, Flatten = 9,
        Cat = 10, Stack = 11, Slice = 12
    };

    enum class LoadOp : uint8_t {
        Empty = 0, Const = 1, Arange = 2, Random = 3, Eye = 4,
        FromCPU = 5, FromCUDA = 6, Normal = 7, Randint = 8,
        Bernoulli = 9, Multinomial = 10
    };

    class TensorShape {
    private:
        std::vector<size_t> dims_;
        size_t total_elements_ = 0;
        bool initialized_ = false;

    public:
        TensorShape() = default;
        TensorShape(std::initializer_list<size_t> dims) : dims_(dims), initialized_(true) {
            compute_total();
        }
        explicit TensorShape(const std::vector<size_t>& dims) : dims_(dims), initialized_(true) {
            compute_total();
        }
        explicit TensorShape(std::span<const size_t> dims) : dims_(dims.begin(), dims.end()), initialized_(true) {
            compute_total();
        }

        size_t rank() const { return dims_.size(); }
        size_t operator[](size_t i) const {
            if (i >= dims_.size()) {
                LOG_ERROR("Shape index {} out of range for rank {}", i, dims_.size());
                return 0;
            }
            return dims_[i];
        }
        size_t elements() const { return total_elements_; }
        const std::vector<size_t>& dims() const { return dims_; }
        bool is_initialized() const { return initialized_; }

        bool operator==(const TensorShape& other) const { return dims_ == other.dims_; }
        bool operator!=(const TensorShape& other) const { return !(*this == other); }

        std::string str() const;

    private:
        void compute_total() {
            total_elements_ = dims_.empty() ? 0 : 1;
            for (auto d : dims_) {
                total_elements_ *= d;
            }
        }
    };

    struct MovementArgs {
        std::variant<
            std::monostate,
            std::vector<int>,
            std::pair<int, int>,
            std::vector<std::pair<int, int>>,
            int,
            void*,
            std::pair<void*, int>
        > args;
    };

    struct LoadArgs {
        TensorShape shape;
        Device device = Device::CUDA;
        DataType dtype = DataType::Float32;
        std::variant<
            std::monostate,
            float,
            std::tuple<float, float, float>,
            std::pair<float, float>,
            std::pair<int, int>,
            void*,
            std::pair<void*, bool>
        > args;
    };

    struct UnaryArgs {
        std::variant<
            std::monostate,
            float,
            int
        > args;
    };

    struct ReduceArgs {
        std::vector<int> axes;
        bool keepdim = false;
        std::variant<
            std::monostate,
            float
        > args;
    };

    class RandomGenerator {
    public:
        static RandomGenerator& instance();
        void manual_seed(uint64_t seed);
        uint64_t get_seed() const { return seed_; }
        void* get_generator(Device device);

        void* get_impl() { return impl_; }
        const void* get_impl() const { return impl_; }

    private:
        RandomGenerator();
        ~RandomGenerator();
        uint64_t seed_;
        void* impl_ = nullptr;
        std::mt19937_64 cpu_generator_;
        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    };

    class Tensor {
    private:
        void* data_ = nullptr;
        std::shared_ptr<void> data_owner_;
        TensorShape shape_;
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::Float32;
        bool initialized_ = false;
        bool is_view_ = false;

        mutable size_t id_ = 0;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

        // NO TEMPLATES - Simple concrete implementations
        Tensor binary_op_impl(const Tensor& other, BinaryOp op) const;
        Tensor binary_op_scalar(float scalar, BinaryOp op) const;
        Tensor& binary_op_inplace_impl(const Tensor& other, BinaryOp op);
        Tensor& binary_op_inplace_scalar(float scalar, BinaryOp op);

        std::pair<Tensor, Tensor> _broadcasted(const Tensor& other, bool match_dtype = true) const;

        int resolve_dim(int dim) const {
            return dim < 0 ? static_cast<int>(shape_.rank()) + dim : dim;
        }

        std::vector<size_t> resolve_dims(std::span<const int> dims) const;
        bool is_contiguous_slice(const std::vector<size_t>& starts,
                                 const std::vector<size_t>& ends) const;
        size_t calculate_offset(const std::vector<size_t>& indices) const;
        Tensor copy_slice(const std::vector<size_t>& starts,
                         const std::vector<size_t>& ends,
                         const std::vector<size_t>& new_shape) const;

    public:
        Tensor() = default;
        Tensor(void* data, TensorShape shape, Device device, DataType dtype);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        ~Tensor();

        // ============= Multi-dimensional accessor for CPU tensors =============
        template<typename T, size_t N>
        class TensorAccessor {
        private:
            T* data_;
            std::array<size_t, N> sizes_;
            std::array<size_t, N> strides_;

        public:
            TensorAccessor(T* data, const std::array<size_t, N>& sizes)
                : data_(data), sizes_(sizes) {
                // Compute strides (row-major)
                strides_[N-1] = 1;
                // Use constexpr if to avoid underflow when N=1
                if constexpr (N > 1) {
                    for (size_t i = N-1; i > 0; --i) {  // ✅ No underflow
                        strides_[i-1] = strides_[i] * sizes_[i];
                    }
                }
            }

            template<typename... Indices>
            T& operator()(Indices... indices) {
                static_assert(sizeof...(Indices) == N, "Wrong number of indices");
                std::array<size_t, N> idx_array{static_cast<size_t>(indices)...};
                size_t offset = 0;
                for (size_t i = 0; i < N; ++i) {
                    offset += idx_array[i] * strides_[i];
                }
                return data_[offset];
            }

            const std::array<size_t, N>& sizes() const { return sizes_; }
        };

        template<typename T, size_t N>
        TensorAccessor<T, N> accessor() {
            if (device_ != Device::CPU) {
                LOG_ERROR("accessor() only works on CPU tensors");
                return TensorAccessor<T, N>(nullptr, std::array<size_t, N>{});
            }
            if (shape_.rank() != N) {
                LOG_ERROR("accessor() dimension mismatch: tensor has {} dims, requested {}",
                         shape_.rank(), N);
                return TensorAccessor<T, N>(nullptr, std::array<size_t, N>{});
            }

            std::array<size_t, N> sizes;
            for (size_t i = 0; i < N; ++i) {
                sizes[i] = shape_[i];
            }
            return TensorAccessor<T, N>(ptr<T>(), sizes);
        }

        // ============= CORE UNIFIED OPERATIONS =============
        static Tensor load(LoadOp op, const LoadArgs& args);
        Tensor movement(MovementOp op, const MovementArgs& args) const;
        Tensor unary(UnaryOp op, const UnaryArgs& args = {}) const;
        Tensor binary(const Tensor& other, BinaryOp op) const { return binary_op_impl(other, op); }
        Tensor binary(float scalar, BinaryOp op) const { return binary_op_scalar(scalar, op); }
        Tensor reduce(ReduceOp op, const ReduceArgs& args = {}) const;
        Tensor ternary(const Tensor& b, const Tensor& c, TernaryOp op) const;

        // ============= FACTORY METHODS =============
        static Tensor empty(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::monostate{};
            return load(LoadOp::Empty, args);
        }

        static Tensor zeros(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = 0.0f;
            return load(LoadOp::Const, args);
        }

        static Tensor ones(TensorShape shape, Device device = Device::CUDA,
                          DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = 1.0f;
            return load(LoadOp::Const, args);
        }

        static Tensor full(TensorShape shape, float value, Device device = Device::CUDA,
                          DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = value;
            return load(LoadOp::Const, args);
        }

        static Tensor full_bool(TensorShape shape, bool value, Device device = Device::CUDA) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = DataType::Bool;
            args.args = value ? 1.0f : 0.0f;
            return load(LoadOp::Const, args);
        }

        static Tensor zeros_bool(TensorShape shape, Device device = Device::CUDA) {
            return full_bool(shape, false, device);
        }

        static Tensor ones_bool(TensorShape shape, Device device = Device::CUDA) {
            return full_bool(shape, true, device);
        }

        static Tensor rand(TensorShape shape, Device device = Device::CUDA,
                          DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{0.0f, 1.0f};
            return load(LoadOp::Random, args);
        }

        static Tensor randn(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{0.0f, 1.0f};
            return load(LoadOp::Normal, args);
        }

        static Tensor uniform(TensorShape shape, float low = 0.0f, float high = 1.0f,
                             Device device = Device::CUDA, DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{low, high};
            return load(LoadOp::Random, args);
        }

        static Tensor normal(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                            Device device = Device::CUDA, DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<float, float>{mean, std};
            return load(LoadOp::Normal, args);
        }

        static Tensor randint(TensorShape shape, int low, int high,
                             Device device = Device::CUDA, DataType dtype = DataType::Int32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = std::pair<int, int>{low, high};
            return load(LoadOp::Randint, args);
        }

        static Tensor bernoulli(TensorShape shape, float p = 0.5f,
                               Device device = Device::CUDA, DataType dtype = DataType::Float32) {
            LoadArgs args;
            args.shape = shape;
            args.device = device;
            args.dtype = dtype;
            args.args = p;
            return load(LoadOp::Bernoulli, args);
        }

        static Tensor multinomial(const Tensor& weights, int num_samples,
                                 bool replacement = false);

        static Tensor arange(float end) {
            LoadArgs args;
            args.shape = TensorShape{};
            args.device = Device::CUDA;
            args.dtype = DataType::Float32;
            args.args = std::tuple<float, float, float>{0.0f, end, 1.0f};
            return load(LoadOp::Arange, args);
        }

        static Tensor arange(float start, float end, float step = 1.0f) {
            LoadArgs args;
            args.shape = TensorShape{};
            args.device = Device::CUDA;
            args.dtype = DataType::Float32;
            args.args = std::tuple<float, float, float>{start, end, step};
            return load(LoadOp::Arange, args);
        }

        static Tensor eye(size_t n, Device device = Device::CUDA) {
            LoadArgs args;
            args.shape = TensorShape{n, n};
            args.device = device;
            args.dtype = DataType::Float32;
            args.args = std::monostate{};
            return load(LoadOp::Eye, args);
        }

        static Tensor eye(size_t m, size_t n, Device device = Device::CUDA) {
            LoadArgs args;
            args.shape = TensorShape{m, n};
            args.device = device;
            args.dtype = DataType::Float32;
            args.args = std::monostate{};
            return load(LoadOp::Eye, args);
        }

        static Tensor from_blob(void* data, TensorShape shape, Device device, DataType dtype) {
            return Tensor(data, shape, device, dtype);
        }

        void set_bool(std::initializer_list<size_t> indices, bool value);
        bool get_bool(std::initializer_list<size_t> indices) const;

        static Tensor from_vector(const std::vector<float>& data, TensorShape shape,
                                 Device device = Device::CUDA);
        static Tensor from_vector(const std::vector<int>& data, TensorShape shape,
                                 Device device = Device::CUDA);
        static Tensor from_vector(const std::vector<bool>& data, TensorShape shape,
                                 Device device = Device::CUDA);

        // Data access
        template <typename T>
        T* ptr() {
            if (!data_ && shape_.elements() > 0) {
                LOG_ERROR("Tensor #{}: Attempting to access null data pointer", id_);
                return nullptr;
            }
            return static_cast<T*>(data_);
        }

        template <typename T>
        const T* ptr() const {
            if (!data_ && shape_.elements() > 0) {
                LOG_ERROR("Tensor #{}: Attempting to access null data pointer", id_);
                return nullptr;
            }
            return static_cast<const T*>(data_);
        }

        void* raw_ptr() { return data_; }
        const void* raw_ptr() const { return data_; }

        // Properties
        const TensorShape& shape() const { return shape_; }
        Device device() const { return device_; }
        DataType dtype() const { return dtype_; }
        bool owns_memory() const { return static_cast<bool>(data_owner_) && !is_view_; }
        bool is_view() const { return is_view_; }
        bool is_empty() const { return !initialized_ || shape_.elements() == 0; }
        bool is_valid() const { return initialized_; }
        size_t numel() const { return shape_.elements(); }
        size_t bytes() const { return numel() * dtype_size(dtype_); }
        size_t ndim() const { return shape_.rank(); }
        size_t size(size_t dim) const { return shape_[dim]; }

        // Memory operations
        Tensor clone() const;
        Tensor contiguous() const;
        Tensor to(Device device) const;
        Tensor to(DataType dtype) const;
        bool is_contiguous() const { return true; }

        // Convenience device conversion
        Tensor cpu() const { return to(Device::CPU); }
        Tensor cuda() const { return to(Device::CUDA); }

        // ============= SHAPE OPERATIONS =============
        Tensor reshape(std::span<const int> sizes) const {
            MovementArgs args;
            args.args = std::vector<int>(sizes.begin(), sizes.end());
            return movement(MovementOp::Reshape, args);
        }
        Tensor reshape(std::initializer_list<int> sizes) const {
            return reshape(std::span<const int>(sizes));
        }
        Tensor reshape(TensorShape new_shape) const;

        Tensor view(std::span<const int> sizes) const { return reshape(sizes); }
        Tensor view(std::initializer_list<int> sizes) const { return reshape(sizes); }
        Tensor view(TensorShape new_shape) const { return reshape(new_shape); }

        Tensor squeeze(std::optional<int> dim = std::nullopt) const {
            MovementArgs args;
            // Use INT_MIN as sentinel for "squeeze all", allowing -1 to be a valid dimension
            args.args = dim.value_or(std::numeric_limits<int>::min());
            return movement(MovementOp::Squeeze, args);
        }

        Tensor squeeze(int dim) const {
            MovementArgs args;
            args.args = dim;
            return movement(MovementOp::Squeeze, args);
        }

        Tensor unsqueeze(int dim) const {
            MovementArgs args;
            args.args = dim;
            return movement(MovementOp::Unsqueeze, args);
        }

        static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);
        Tensor expand(std::span<const int> sizes) const {
            MovementArgs args;
            args.args = std::vector<int>(sizes.begin(), sizes.end());
            return movement(MovementOp::Expand, args);
        }
        Tensor expand(std::initializer_list<int> sizes) const {
            return expand(std::span<const int>(sizes));
        }
        Tensor expand(const TensorShape& target_shape) const;

        Tensor flatten(int start_dim = 0, int end_dim = -1) const {
            MovementArgs args;
            args.args = std::pair<int, int>{start_dim, end_dim};
            return movement(MovementOp::Flatten, args);
        }

        Tensor permute(std::span<const int> axes) const;
        Tensor permute(std::initializer_list<int> axes) const {
            return permute(std::span<const int>(axes));
        }

        Tensor transpose(int dim1 = -2, int dim2 = -1) const {
            MovementArgs args;
            args.args = std::pair<int, int>{dim1, dim2};
            return movement(MovementOp::Transpose, args);
        }
        Tensor t() const;

        Tensor slice(std::span<const std::pair<int, int>> ranges) const;
        Tensor slice(std::initializer_list<std::pair<int, int>> ranges) const {
            return slice(std::span<const std::pair<int, int>>(ranges));
        }
        Tensor slice(size_t dim, size_t start, size_t end) const;

        Tensor cat(const Tensor& other, int dim = 0) const;

        static Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
        static Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

        // Broadcasting
        Tensor broadcast_to(const TensorShape& target_shape) const;
        bool can_broadcast_to(const TensorShape& target) const;
        TensorShape broadcast_shape(const TensorShape& other) const;

        // ============= UNARY OPERATIONS =============
        Tensor neg() const { return unary(UnaryOp::Neg); }
        Tensor abs() const { return unary(UnaryOp::Abs); }
        Tensor sign() const { return unary(UnaryOp::Sign); }
        Tensor reciprocal() const { return unary(UnaryOp::Reciprocal); }
        Tensor exp() const { return unary(UnaryOp::Exp); }
        Tensor exp2() const { return unary(UnaryOp::Exp2); }
        Tensor log() const { return unary(UnaryOp::Log); }
        Tensor log2() const { return unary(UnaryOp::Log2); }
        Tensor log10() const { return unary(UnaryOp::Log10); }
        Tensor log1p() const { return unary(UnaryOp::Log1p); }
        Tensor sqrt() const { return unary(UnaryOp::Sqrt); }
        Tensor rsqrt() const { return unary(UnaryOp::Rsqrt); }
        Tensor square() const { return unary(UnaryOp::Square); }
        Tensor sin() const { return unary(UnaryOp::Sin); }
        Tensor cos() const { return unary(UnaryOp::Cos); }
        Tensor tan() const { return unary(UnaryOp::Tan); }
        Tensor asin() const { return unary(UnaryOp::Asin); }
        Tensor acos() const { return unary(UnaryOp::Acos); }
        Tensor atan() const { return unary(UnaryOp::Atan); }
        Tensor sinh() const { return unary(UnaryOp::Sinh); }
        Tensor cosh() const { return unary(UnaryOp::Cosh); }
        Tensor tanh() const { return unary(UnaryOp::Tanh); }
        Tensor sigmoid() const { return unary(UnaryOp::Sigmoid); }
        Tensor relu() const { return unary(UnaryOp::Relu); }
        Tensor gelu() const { return unary(UnaryOp::Gelu); }
        Tensor swish() const { return unary(UnaryOp::Swish); }
        Tensor floor() const { return unary(UnaryOp::Floor); }
        Tensor ceil() const { return unary(UnaryOp::Ceil); }
        Tensor round() const { return unary(UnaryOp::Round); }
        Tensor trunc() const { return unary(UnaryOp::Trunc); }
        Tensor isnan() const { return unary(UnaryOp::IsNan); }
        Tensor isinf() const { return unary(UnaryOp::IsInf); }
        Tensor isfinite() const { return unary(UnaryOp::IsFinite); }
        Tensor logical_not() const { return unary(UnaryOp::LogicalNot); }

        Tensor normalize(int dim = -1, float eps = 1e-12f) const;
        Tensor logit(float eps = 1e-7f) const;

        // ============= BINARY OPERATIONS =============
        Tensor add(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Add); }
        Tensor add(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Add); }
        Tensor sub(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Sub); }
        Tensor sub(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Sub); }
        Tensor mul(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Mul); }
        Tensor mul(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Mul); }
        Tensor div(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Div); }
        Tensor div(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Div); }
        Tensor pow(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Pow); }
        Tensor pow(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Pow); }
        Tensor mod(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Mod); }
        Tensor mod(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Mod); }
        Tensor maximum(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Maximum); }
        Tensor maximum(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Maximum); }
        Tensor minimum(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Minimum); }
        Tensor minimum(float scalar) const { return binary_op_scalar(scalar, BinaryOp::Minimum); }

        // Comparison operations
        Tensor eq(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Equal); }
        Tensor ne(const Tensor& other) const { return binary_op_impl(other, BinaryOp::NotEqual); }
        Tensor lt(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Less); }
        Tensor le(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LessEqual); }
        Tensor gt(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Greater); }
        Tensor ge(const Tensor& other) const { return binary_op_impl(other, BinaryOp::GreaterEqual); }

        Tensor eq(float value) const { return binary_op_scalar(value, BinaryOp::Equal); }
        Tensor ne(float value) const { return binary_op_scalar(value, BinaryOp::NotEqual); }
        Tensor lt(float value) const { return binary_op_scalar(value, BinaryOp::Less); }
        Tensor le(float value) const { return binary_op_scalar(value, BinaryOp::LessEqual); }
        Tensor gt(float value) const { return binary_op_scalar(value, BinaryOp::Greater); }
        Tensor ge(float value) const { return binary_op_scalar(value, BinaryOp::GreaterEqual); }

        // Logical operations
        Tensor logical_and(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LogicalAnd); }
        Tensor logical_or(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LogicalOr); }
        Tensor logical_xor(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LogicalXor); }

        // ============= REDUCE OPERATIONS =============
        Tensor sum(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Sum, args);
        }
        Tensor mean(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Mean, args);
        }
        Tensor max(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Max, args);
        }
        Tensor min(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Min, args);
        }
        Tensor prod(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Prod, args);
        }
        Tensor any(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Any, args);
        }
        Tensor all(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::All, args);
        }
        Tensor std(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Std, args);
        }
        Tensor var(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Var, args);
        }
        Tensor argmax(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Argmax, args);
        }
        Tensor argmin(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Argmin, args);
        }

        // Cumulative sum
        Tensor cumsum(int dim = 0) const;

        // Scalar reduce operations
        float sum_scalar() const { return sum().item(); }
        float mean_scalar() const { return mean().item(); }
        float min_scalar() const { return min().item(); }
        float max_scalar() const { return max().item(); }
        float std_scalar(float eps = 1e-8f) const { return std().item(); }
        float var_scalar(float eps = 1e-8f) const { return var().item(); }
        std::pair<float, float> minmax() const { return {min_scalar(), max_scalar()}; }

        float norm(float p = 2.0f) const;
        float item() const;

        // Template version of item()
        template<typename T>
        T item() const {
            if (!is_valid() || numel() != 1) {
                LOG_ERROR("item<T>() requires a valid single-element tensor");
                return T{};
            }

            T value{};
            if (device_ == Device::CUDA) {
                cudaMemcpy(&value, data_, sizeof(T), cudaMemcpyDeviceToHost);
            } else {
                value = *static_cast<const T*>(data_);
            }
            return value;
        }

        size_t count_nonzero() const;

        // ============= TERNARY OPERATIONS =============
        Tensor where(const Tensor& condition, const Tensor& other) const {
            return condition.ternary(*this, other, TernaryOp::Where);
        }

        Tensor clamp(float min_val, float max_val) const {
            if (!is_valid()) {
                return Tensor();
            }

            if (numel() == 0) {
                // Return empty tensor with same shape
                return empty(shape_, device_, dtype_);
            }

            auto min_t = full(shape_, min_val, device_, dtype_);
            auto max_t = full(shape_, max_val, device_, dtype_);
            return ternary(min_t, max_t, TernaryOp::Clamp);
        }

        Tensor clamp_min(float min) const {
            return clamp(min, std::numeric_limits<float>::max());
        }

        Tensor clamp_max(float max) const {
            return clamp(std::numeric_limits<float>::lowest(), max);
        }

        // In-place clamp operations
        Tensor& clamp_(float min_val, float max_val);
        Tensor& clamp_min_(float min);
        Tensor& clamp_max_(float max);

        // In-place operations
        Tensor& add_(const Tensor& other) { return binary_op_inplace_impl(other, BinaryOp::Add); }
        Tensor& add_(float scalar) { return binary_op_inplace_scalar(scalar, BinaryOp::Add); }
        Tensor& sub_(const Tensor& other) { return binary_op_inplace_impl(other, BinaryOp::Sub); }
        Tensor& sub_(float scalar) { return binary_op_inplace_scalar(scalar, BinaryOp::Sub); }
        Tensor& mul_(const Tensor& other) { return binary_op_inplace_impl(other, BinaryOp::Mul); }
        Tensor& mul_(float scalar) { return binary_op_inplace_scalar(scalar, BinaryOp::Mul); }
        Tensor& div_(const Tensor& other) { return binary_op_inplace_impl(other, BinaryOp::Div); }
        Tensor& div_(float scalar) { return binary_op_inplace_scalar(scalar, BinaryOp::Div); }

        // Matrix operations
        Tensor mm(const Tensor& other) const;
        Tensor bmm(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor dot(const Tensor& other) const;

        // Masking operations
        Tensor masked_select(const Tensor& mask) const;
        Tensor& masked_fill_(const Tensor& mask, float value);
        Tensor masked_fill(const Tensor& mask, float value) const;

        // Indexing operations
        Tensor index_select(int dim, const Tensor& indices) const;
        Tensor gather(int dim, const Tensor& indices) const;
        Tensor take(const Tensor& indices) const;

        Tensor nonzero() const;
        std::vector<Tensor> nonzero_split() const;

        Tensor& scatter_(int dim, const Tensor& indices, const Tensor& src,
                         ScatterMode mode = ScatterMode::None);
        Tensor& scatter_(int dim, const Tensor& indices, float value,
                         ScatterMode mode = ScatterMode::None);
        Tensor& index_fill_(int dim, const Tensor& indices, float value);
        Tensor& index_copy_(int dim, const Tensor& indices, const Tensor& src);
        Tensor& index_add_(int dim, const Tensor& indices, const Tensor& src);
        Tensor& index_put_(const Tensor& indices, const Tensor& values);
        Tensor& index_put_(const std::vector<Tensor>& indices, const Tensor& values);

        Tensor index_select(int dim, const Tensor& indices, BoundaryMode mode) const;
        Tensor gather(int dim, const Tensor& indices, BoundaryMode mode) const;

        TensorIndexer operator[](const Tensor& indices);
        TensorIndexer operator[](const std::vector<Tensor>& indices);
        MaskedTensorProxy operator[](const Tensor& mask) const;

        float& at(std::initializer_list<size_t> indices);
        float at(std::initializer_list<size_t> indices) const;

        // Operator overloads - NO TEMPLATES, just concrete overloads
        Tensor operator+(const Tensor& other) const { return add(other); }
        Tensor operator+(float scalar) const { return add(scalar); }
        Tensor operator+(int scalar) const { return add(static_cast<float>(scalar)); }
        Tensor operator+(double scalar) const { return add(static_cast<float>(scalar)); }

        Tensor operator-(const Tensor& other) const { return sub(other); }
        Tensor operator-(float scalar) const { return sub(scalar); }
        Tensor operator-(int scalar) const { return sub(static_cast<float>(scalar)); }
        Tensor operator-(double scalar) const { return sub(static_cast<float>(scalar)); }

        Tensor operator*(const Tensor& other) const { return mul(other); }
        Tensor operator*(float scalar) const { return mul(scalar); }
        Tensor operator*(int scalar) const { return mul(static_cast<float>(scalar)); }
        Tensor operator*(double scalar) const { return mul(static_cast<float>(scalar)); }

        Tensor operator/(const Tensor& other) const { return div(other); }
        Tensor operator/(float scalar) const { return div(scalar); }
        Tensor operator/(int scalar) const { return div(static_cast<float>(scalar)); }
        Tensor operator/(double scalar) const { return div(static_cast<float>(scalar)); }

        Tensor operator%(const Tensor& other) const { return mod(other); }
        Tensor operator%(float scalar) const { return mod(scalar); }
        Tensor operator%(int scalar) const { return mod(static_cast<float>(scalar)); }

        Tensor operator-() const { return neg(); }

        Tensor operator==(const Tensor& other) const { return eq(other); }
        Tensor operator==(float value) const { return eq(value); }
        Tensor operator!=(const Tensor& other) const { return ne(other); }
        Tensor operator!=(float value) const { return ne(value); }
        Tensor operator<(const Tensor& other) const { return lt(other); }
        Tensor operator<(float value) const { return lt(value); }
        Tensor operator<=(const Tensor& other) const { return le(other); }
        Tensor operator<=(float value) const { return le(value); }
        Tensor operator>(const Tensor& other) const { return gt(other); }
        Tensor operator>(float value) const { return gt(value); }
        Tensor operator>=(const Tensor& other) const { return ge(other); }
        Tensor operator>=(float value) const { return ge(value); }

        Tensor operator&&(const Tensor& other) const { return logical_and(other); }
        Tensor operator||(const Tensor& other) const { return logical_or(other); }
        Tensor operator!() const { return logical_not(); }

        // Bitwise NOT for boolean tensors
        Tensor operator~() const;

        // Bitwise OR for boolean tensors
        Tensor operator|(const Tensor& other) const;

        // Other in-place operations
        Tensor& zero_();
        Tensor& fill_(float value);
        Tensor& copy_from(const Tensor& other);
        Tensor& uniform_(float low = 0.0f, float high = 1.0f);
        Tensor& normal_(float mean = 0.0f, float std = 1.0f);

        std::optional<Tensor> try_reshape(TensorShape shape) const;

        static std::vector<Tensor> split_batch(const Tensor& tensor, size_t batch_size);
        static void enable_profiling(bool enable) { profiling_enabled_ = enable; }

        // Utility template methods - these are OK as they're just helpers
        template <typename Func>
        Tensor& inplace(Func&& func) {
            func(*this);
            return *this;
        }

        template <typename Func>
        Tensor apply(Func&& func) const {
            return func(*this);
        }

        template <typename Func>
        Tensor timed(const std::string& name, Func&& func) const {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = func(*this);
            auto end = std::chrono::high_resolution_clock::now();
            if (profiling_enabled_) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                LOG_INFO("{}: {} μs", name, duration.count());
            }
            return result;
        }

        // Validation & assertions
        Tensor& assert_shape(TensorShape expected, const std::string& msg = "");
        Tensor& assert_device(Device expected);
        Tensor& assert_dtype(DataType expected);
        Tensor& assert_finite();

        // Comparison operations
        bool has_nan() const;
        bool has_inf() const;
        bool all_close(const Tensor& other, float rtol = 1e-5f, float atol = 1e-8f) const;

        // Utility functions
        std::string str() const;
        std::vector<float> to_vector() const;
        std::vector<int> to_vector_int() const;
        std::vector<bool> to_vector_bool() const;
        std::vector<float> debug_values(size_t max_values = 100) const;

        void dump_diagnostic(const std::string& filename) const;
        void log_info(const std::string& name = "") const;
        void print_formatted(const std::string& name = "", size_t max_per_dim = 10) const;

    private:
        void print_1d(size_t max_elem = 10) const;
        void print_2d(size_t max_per_dim = 10) const;

        friend class TensorIndexer;
        friend class MaskedTensorProxy;
    };

    // Helper classes
    class MaskedTensorProxy {
    private:
        const Tensor* tensor_;
        Tensor mask_;

    public:
        MaskedTensorProxy(const Tensor* tensor, Tensor mask)
            : tensor_(tensor), mask_(std::move(mask)) {}

        void operator=(float value);
        void operator=(const Tensor& other);
        operator Tensor() const;
    };

    class TensorIndexer {
    private:
        Tensor* tensor_;
        std::vector<Tensor> indices_;

    public:
        TensorIndexer(Tensor* tensor, std::vector<Tensor> indices)
            : tensor_(tensor), indices_(std::move(indices)) {}

        void operator=(float value);
        void operator=(const Tensor& other);
        operator Tensor() const;
    };

    class TensorBuilder {
    private:
        TensorShape shape_;
        Device device_ = Device::CUDA;
        DataType dtype_ = DataType::Float32;
        std::optional<float> fill_value_;
        bool check_finite_ = false;

    public:
        TensorBuilder& with_shape(TensorShape shape);
        TensorBuilder& on_device(Device device);
        TensorBuilder& with_dtype(DataType dtype);
        TensorBuilder& filled_with(float value);
        TensorBuilder& ensure_finite();
        Tensor build();
    };

    namespace tensor {
        inline void manual_seed(uint64_t seed) {
            RandomGenerator::instance().manual_seed(seed);
        }

        Tensor zeros_like(const Tensor& other);
        Tensor ones_like(const Tensor& other);
        Tensor ones_like(const Tensor& other, DataType dtype);
        Tensor rand_like(const Tensor& other);
        Tensor randn_like(const Tensor& other);

        Tensor diag(const Tensor& diagonal);
        Tensor linspace(float start, float end, size_t steps);

        Tensor stack(std::vector<Tensor>&& tensors, int dim = 0);
        Tensor cat(std::vector<Tensor>&& tensors, int dim = 0);

        bool check_valid(const Tensor& t, const std::string& name);
        void assert_same_shape(const Tensor& a, const Tensor& b);
        void assert_same_device(const Tensor& a, const Tensor& b);
    }

    class TensorError : public std::runtime_error {
    public:
        TensorError(const std::string& msg, const Tensor* t = nullptr);
        const std::string& tensor_info() const { return tensor_info_; }

    private:
        std::string tensor_info_;
    };

    namespace SafeOps {
        using Tensor = gs::Tensor;
        Tensor divide(const Tensor& a, const Tensor& b, float epsilon = 1e-6f);
        Tensor log(const Tensor& input, float epsilon = 1e-6f);
        Tensor sqrt(const Tensor& input, float epsilon = 0.0f);
    }

    class MemoryInfo {
    public:
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        size_t allocated_bytes = 0;
        int device_id = -1;

        static MemoryInfo cuda();
        static MemoryInfo cpu();

        void log() const;
    };

    namespace functional {
        Tensor map(const Tensor& input, std::function<float(float)> func);
        float reduce(const Tensor& input, float init, std::function<float(float, float)> func);
        Tensor filter(const Tensor& input, std::function<bool(float)> predicate);

        template <typename... Funcs>
        auto pipe(Funcs... funcs) {
            return [=](const Tensor& input) -> Tensor {
                Tensor result = input.clone();
                ((result = funcs(std::move(result))), ...);
                return result;
            };
        }
    }

} // namespace gs