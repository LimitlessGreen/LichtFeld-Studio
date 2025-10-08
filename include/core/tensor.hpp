/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cuda_runtime.h>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace gs {

    // Forward declarations
    class Tensor;
    class TensorError;
    class TensorIndexer;
    class MaskedTensorProxy;
    class TensorRowProxy;

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
        if (a == b)
            return a;

        if (a == DataType::Bool) {
            if (b == DataType::Float32 || b == DataType::Float16)
                return b;
            if (b == DataType::Int32 || b == DataType::Int64)
                return b;
            return DataType::Float32;
        }
        if (b == DataType::Bool) {
            if (a == DataType::Float32 || a == DataType::Float16)
                return a;
            if (a == DataType::Int32 || a == DataType::Int64)
                return a;
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
        Add = 0,
        Sub = 1,
        Mul = 2,
        Div = 3,
        Pow = 4,
        Mod = 5,
        Equal = 6,
        NotEqual = 7,
        Less = 8,
        LessEqual = 9,
        Greater = 10,
        GreaterEqual = 11,
        LogicalAnd = 12,
        LogicalOr = 13,
        LogicalXor = 14,
        Maximum = 15,
        Minimum = 16,
        BitwiseAnd = 17,
        BitwiseOr = 18,
        BitwiseXor = 19,
        LeftShift = 20,
        RightShift = 21
    };

    enum class UnaryOp : uint8_t {
        Neg = 0,
        Abs = 1,
        Sign = 2,
        Reciprocal = 3,
        Exp = 4,
        Exp2 = 5,
        Log = 6,
        Log2 = 7,
        Log10 = 8,
        Log1p = 9,
        Sqrt = 10,
        Rsqrt = 11,
        Square = 12,
        Sin = 13,
        Cos = 14,
        Tan = 15,
        Asin = 16,
        Acos = 17,
        Atan = 18,
        Sinh = 19,
        Cosh = 20,
        Tanh = 21,
        Sigmoid = 22,
        Relu = 23,
        Gelu = 24,
        Swish = 25,
        Floor = 26,
        Ceil = 27,
        Round = 28,
        Trunc = 29,
        IsNan = 30,
        IsInf = 31,
        IsFinite = 32,
        LogicalNot = 33,
        Normalize = 34,
        Logit = 35
    };

    enum class ReduceOp : uint8_t {
        Sum = 0,
        Mean = 1,
        Max = 2,
        Min = 3,
        Prod = 4,
        Any = 5,
        All = 6,
        Std = 7,
        Var = 8,
        Argmax = 9,
        Argmin = 10,
        CountNonzero = 11,
        Norm = 12
    };

    enum class TernaryOp : uint8_t {
        Where = 0,
        MulAdd = 1,
        Clamp = 2
    };

    enum class MovementOp : uint8_t {
        Reshape = 0,
        Permute = 1,
        Expand = 2,
        Pad = 3,
        Shrink = 4,
        Flip = 5,
        Transpose = 6,
        Squeeze = 7,
        Unsqueeze = 8,
        Flatten = 9,
        Cat = 10,
        Stack = 11,
        Slice = 12
    };

    enum class LoadOp : uint8_t {
        Empty = 0,
        Const = 1,
        Arange = 2,
        Random = 3,
        Eye = 4,
        FromCPU = 5,
        FromCUDA = 6,
        Normal = 7,
        Randint = 8,
        Bernoulli = 9,
        Multinomial = 10
    };

    // ============= Compile-Time Operation Tables =============
    namespace op_tables {
        template<typename T>
        using UnaryFunc = T(*)(T);

        template<typename T>
        using BinaryFunc = T(*)(T, T);

        template<typename T>
        using CompareFunc = bool(*)(T, T);

        template<typename T>
        consteval auto make_unary_ops() {
            std::array<UnaryFunc<T>, 36> ops{};

            ops[int(UnaryOp::Neg)] = [](T x) { return -x; };
            ops[int(UnaryOp::Abs)] = [](T x) { return std::abs(x); };
            ops[int(UnaryOp::Sign)] = [](T x) { return T((x > T(0)) - (x < T(0))); };
            ops[int(UnaryOp::Reciprocal)] = [](T x) { return T(1) / x; };
            ops[int(UnaryOp::Exp)] = [](T x) { return std::exp(x); };
            ops[int(UnaryOp::Exp2)] = [](T x) { return std::exp2(x); };
            ops[int(UnaryOp::Log)] = [](T x) { return std::log(std::max(x, T(1e-45))); };
            ops[int(UnaryOp::Log2)] = [](T x) { return std::log2(std::max(x, T(1e-45))); };
            ops[int(UnaryOp::Log10)] = [](T x) { return std::log10(std::max(x, T(1e-45))); };
            ops[int(UnaryOp::Log1p)] = [](T x) { return std::log1p(x); };
            ops[int(UnaryOp::Sqrt)] = [](T x) { return std::sqrt(std::max(x, T(0))); };
            ops[int(UnaryOp::Rsqrt)] = [](T x) { return T(1) / std::sqrt(std::max(x, T(1e-45))); };
            ops[int(UnaryOp::Square)] = [](T x) { return x * x; };
            ops[int(UnaryOp::Sin)] = [](T x) { return std::sin(x); };
            ops[int(UnaryOp::Cos)] = [](T x) { return std::cos(x); };
            ops[int(UnaryOp::Tan)] = [](T x) { return std::tan(x); };
            ops[int(UnaryOp::Asin)] = [](T x) { return std::asin(x); };
            ops[int(UnaryOp::Acos)] = [](T x) { return std::acos(x); };
            ops[int(UnaryOp::Atan)] = [](T x) { return std::atan(x); };
            ops[int(UnaryOp::Sinh)] = [](T x) { return std::sinh(x); };
            ops[int(UnaryOp::Cosh)] = [](T x) { return std::cosh(x); };
            ops[int(UnaryOp::Tanh)] = [](T x) { return std::tanh(x); };
            ops[int(UnaryOp::Sigmoid)] = [](T x) { return T(1) / (T(1) + std::exp(-x)); };
            ops[int(UnaryOp::Relu)] = [](T x) { return std::max(x, T(0)); };
            ops[int(UnaryOp::Floor)] = [](T x) { return std::floor(x); };
            ops[int(UnaryOp::Ceil)] = [](T x) { return std::ceil(x); };
            ops[int(UnaryOp::Round)] = [](T x) { return std::round(x); };
            ops[int(UnaryOp::Trunc)] = [](T x) { return std::trunc(x); };

            return ops;
        }

        template<typename T>
        consteval auto make_binary_ops() {
            std::array<BinaryFunc<T>, 22> ops{};

            ops[int(BinaryOp::Add)] = [](T a, T b) { return a + b; };
            ops[int(BinaryOp::Sub)] = [](T a, T b) { return a - b; };
            ops[int(BinaryOp::Mul)] = [](T a, T b) { return a * b; };
            ops[int(BinaryOp::Div)] = [](T a, T b) { return a / b; };
            ops[int(BinaryOp::Pow)] = [](T a, T b) { return static_cast<T>(std::pow(a, b)); };
            ops[int(BinaryOp::Maximum)] = [](T a, T b) { return std::max(a, b); };
            ops[int(BinaryOp::Minimum)] = [](T a, T b) { return std::min(a, b); };

            return ops;
        }

        template<typename T>
        consteval auto make_compare_ops() {
            std::array<CompareFunc<T>, 22> ops{};

            ops[int(BinaryOp::Equal)] = [](T a, T b) { return a == b; };
            ops[int(BinaryOp::NotEqual)] = [](T a, T b) { return a != b; };
            ops[int(BinaryOp::Less)] = [](T a, T b) { return a < b; };
            ops[int(BinaryOp::LessEqual)] = [](T a, T b) { return a <= b; };
            ops[int(BinaryOp::Greater)] = [](T a, T b) { return a > b; };
            ops[int(BinaryOp::GreaterEqual)] = [](T a, T b) { return a >= b; };

            return ops;
        }

        template<typename T>
        consteval auto make_logical_ops() {
            std::array<CompareFunc<T>, 22> ops{};

            ops[int(BinaryOp::LogicalAnd)] = [](T a, T b) { return (a != T(0)) && (b != T(0)); };
            ops[int(BinaryOp::LogicalOr)] = [](T a, T b) { return (a != T(0)) || (b != T(0)); };
            ops[int(BinaryOp::LogicalXor)] = [](T a, T b) { return (a != T(0)) != (b != T(0)); };
            ops[int(BinaryOp::BitwiseOr)] = [](T a, T b) { return (a != T(0)) || (b != T(0)); };

            return ops;
        }

        // Compile-time constants
        inline constexpr auto float_unary_ops = make_unary_ops<float>();
        inline constexpr auto float_binary_ops = make_binary_ops<float>();
        inline constexpr auto float_compare_ops = make_compare_ops<float>();
        inline constexpr auto float_logical_ops = make_logical_ops<float>();

        inline constexpr auto int_binary_ops = make_binary_ops<int>();

        inline constexpr auto bool_logical_ops = make_logical_ops<unsigned char>();
    } // namespace op_tables

    class TensorShape {
    private:
        std::vector<size_t> dims_;
        size_t total_elements_ = 0;
        bool initialized_ = false;

    public:
        TensorShape() = default;
        TensorShape(std::initializer_list<size_t> dims) : dims_(dims),
                                                          initialized_(true) {
            compute_total();
        }
        explicit TensorShape(const std::vector<size_t>& dims) : dims_(dims),
                                                                initialized_(true) {
            compute_total();
        }
        explicit TensorShape(std::span<const size_t> dims) : dims_(dims.begin(), dims.end()),
                                                             initialized_(true) {
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

        // Calculate strides for row-major layout
        std::vector<size_t> strides() const {
            if (dims_.empty()) return {};

            std::vector<size_t> result(dims_.size());
            result.back() = 1;
            for (int i = static_cast<int>(dims_.size()) - 2; i >= 0; --i) {
                result[i] = result[i + 1] * dims_[i + 1];
            }
            return result;
        }

        bool operator==(const TensorShape& other) const { return dims_ == other.dims_; }
        bool operator!=(const TensorShape& other) const { return !(*this == other); }

        std::string str() const;

    private:
        void compute_total() {
            if (dims_.empty()) {
                total_elements_ = 1;
            } else {
                total_elements_ = 1;
                for (auto d : dims_) {
                    total_elements_ *= d;
                }
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
            std::pair<void*, int>>
            args;
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
            std::pair<void*, bool>>
            args;
    };

    struct UnaryArgs {
        std::variant<
            std::monostate,
            float,
            int>
            args;
    };

    struct ReduceArgs {
        std::vector<int> axes;
        bool keepdim = false;
        std::variant<
            std::monostate,
            float>
            args;
    };

    class RandomGenerator {
    public:
        static RandomGenerator& instance();
        void manual_seed(uint64_t seed);
        uint64_t get_seed() const { return seed_; }
        void* get_generator(Device device);
        uint64_t get_next_cuda_seed();
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

        Tensor binary_op_impl(const Tensor& other, BinaryOp op) const;
        Tensor binary_op_scalar(float scalar, BinaryOp op) const;
        Tensor& binary_op_inplace_impl(const Tensor& other, BinaryOp op);
        Tensor& binary_op_inplace_scalar(float scalar, BinaryOp op);

        std::pair<Tensor, Tensor> _broadcasted(const Tensor& other, bool match_dtype = true) const;

        int resolve_dim(int dim) const {
            return dim < 0 ? static_cast<int>(shape_.rank()) + dim : dim;
        }

        // Validation helpers
        bool validate_binary_op(const Tensor& other, bool require_same_shape = false, bool require_same_device = false) const {
            if (!is_valid() || !other.is_valid()) {
                LOG_ERROR("Operation on invalid tensor");
                return false;
            }
            if (require_same_device && device_ != other.device()) {
                LOG_ERROR("Tensors must be on same device");
                return false;
            }
            if (require_same_shape && shape_ != other.shape()) {
                LOG_ERROR("Shape mismatch: {} vs {}", shape_.str(), other.shape_.str());
                return false;
            }
            return true;
        }

        bool validate_unary_op() const {
            if (!is_valid()) {
                LOG_ERROR("Unary operation on invalid tensor");
                return false;
            }
            return true;
        }

        bool validate_ternary_op(const Tensor& b, const Tensor& c) const {
            if (!is_valid() || !b.is_valid() || !c.is_valid()) {
                LOG_ERROR("Ternary operation on invalid tensor");
                return false;
            }
            if (device_ != b.device() || device_ != c.device()) {
                LOG_ERROR("All tensors must be on same device");
                return false;
            }
            return true;
        }

        // Helper to ensure tensor is on same device
        Tensor ensure_same_device(const Tensor& other) const {
            return (other.device() == device_) ? other.clone() : other.to(device_);
        }

        // Helper to create view with shared ownership
        Tensor create_view(const TensorShape& new_shape) const {
            Tensor view(data_, new_shape, device_, dtype_);
            view.data_owner_ = data_owner_;
            view.is_view_ = true;
            return view;
        }

        std::vector<size_t> resolve_dims(std::span<const int> dims) const;
        bool is_contiguous_slice(const std::vector<size_t>& starts,
                                 const std::vector<size_t>& ends) const;
        size_t calculate_offset(const std::vector<size_t>& indices) const;
        Tensor copy_slice(const std::vector<size_t>& starts,
                          const std::vector<size_t>& ends,
                          const std::vector<size_t>& new_shape) const;

        // Unified template for binary operations
        template<typename T>
        Tensor binary_op(const T& rhs, BinaryOp op) const {
            if constexpr (std::is_same_v<T, Tensor>) {
                return binary_op_impl(rhs, op);
            } else if constexpr (std::is_arithmetic_v<T>) {
                return binary_op_scalar(static_cast<float>(rhs), op);
            } else {
                static_assert(std::is_same_v<T, Tensor> || std::is_arithmetic_v<T>,
                             "Binary operations only support Tensor or arithmetic types");
                return Tensor();
            }
        }

        template<typename T>
        Tensor& binary_op_inplace(const T& rhs, BinaryOp op) {
            if constexpr (std::is_same_v<T, Tensor>) {
                return binary_op_inplace_impl(rhs, op);
            } else if constexpr (std::is_arithmetic_v<T>) {
                return binary_op_inplace_scalar(static_cast<float>(rhs), op);
            } else {
                static_assert(std::is_same_v<T, Tensor> || std::is_arithmetic_v<T>,
                             "Binary operations only support Tensor or arithmetic types");
                return *this;
            }
        }

    public:
        Tensor() = default;
        Tensor(void* data, TensorShape shape, Device device, DataType dtype);

        // Copy constructor and assignment
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);

        // Move constructor and assignment
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        ~Tensor();

        // ============= Multi-dimensional accessor =============
        template <typename T, size_t N>
        class TensorAccessor {
        private:
            T* data_;
            std::array<size_t, N> sizes_;
            std::array<size_t, N> strides_;

        public:
            TensorAccessor(T* data, const std::array<size_t, N>& sizes)
                : data_(data),
                  sizes_(sizes) {
                strides_[N - 1] = 1;
                if constexpr (N > 1) {
                    for (size_t i = N - 1; i > 0; --i) {
                        strides_[i - 1] = strides_[i] * sizes_[i];
                    }
                }
            }

            template <typename... Indices>
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

        template <typename T, size_t N>
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

        // ============= Array-like indexing operator[] =============
        TensorRowProxy operator[](size_t index);
        const TensorRowProxy operator[](size_t index) const;

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

        static Tensor linspace(float start, float end, size_t steps, Device device = Device::CUDA);

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

        static Tensor diag(const Tensor& diagonal);

        static Tensor from_blob(void* data, TensorShape shape, Device device, DataType dtype) {
            return Tensor(data, shape, device, dtype);
        }

        static Tensor from_vector(const std::vector<float>& data, TensorShape shape,
                                  Device device = Device::CUDA);
        static Tensor from_vector(const std::vector<int>& data, TensorShape shape,
                                  Device device = Device::CUDA);
        static Tensor from_vector(const std::vector<bool>& data, TensorShape shape,
                                  Device device = Device::CUDA);

        // Initializer list overloads for convenience
        static Tensor from_vector(std::initializer_list<float> data, TensorShape shape,
                                  Device device = Device::CUDA) {
            return from_vector(std::vector<float>(data), shape, device);
        }

        static Tensor from_vector(std::initializer_list<int> data, TensorShape shape,
                                  Device device = Device::CUDA) {
            return from_vector(std::vector<int>(data), shape, device);
        }

        static Tensor from_vector(std::initializer_list<bool> data, TensorShape shape,
                                  Device device = Device::CUDA) {
            return from_vector(std::vector<bool>(data), shape, device);
        }

        // ============= LIKE OPERATIONS =============
        static Tensor zeros_like(const Tensor& other) {
            return zeros(other.shape(), other.device(), other.dtype());
        }

        static Tensor ones_like(const Tensor& other) {
            return ones(other.shape(), other.device(), other.dtype());
        }

        static Tensor ones_like(const Tensor& other, DataType dtype) {
            return ones(other.shape(), other.device(), dtype);
        }

        static Tensor rand_like(const Tensor& other) {
            return rand(other.shape(), other.device(), other.dtype());
        }

        static Tensor randn_like(const Tensor& other) {
            return randn(other.shape(), other.device(), other.dtype());
        }

        static Tensor empty_like(const Tensor& other) {
            return empty(other.shape(), other.device(), other.dtype());
        }

        static Tensor full_like(const Tensor& other, float value) {
            return full(other.shape(), value, other.device(), other.dtype());
        }

        // ============= COMBINING TENSORS =============
        static Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
        static Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

        // ============= CONDITIONAL =============
        static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

        // ============= GLOBAL CONFIGURATION =============
        static void manual_seed(uint64_t seed) {
            RandomGenerator::instance().manual_seed(seed);
        }

        static void enable_profiling(bool enable) { profiling_enabled_ = enable; }

        void set_bool(std::initializer_list<size_t> indices, bool value);
        bool get_bool(std::initializer_list<size_t> indices) const;
        void set_bool(std::span<const size_t> indices, bool value);
        bool get_bool(std::span<const size_t> indices) const;

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

        // ============= BINARY OPERATIONS (Template-based) =============

        // Arithmetic operations
        template<typename T>
        Tensor add(const T& other) const { return binary_op(other, BinaryOp::Add); }

        template<typename T>
        Tensor sub(const T& other) const { return binary_op(other, BinaryOp::Sub); }

        template<typename T>
        Tensor mul(const T& other) const { return binary_op(other, BinaryOp::Mul); }

        template<typename T>
        Tensor div(const T& other) const { return binary_op(other, BinaryOp::Div); }

        template<typename T>
        Tensor pow(const T& other) const { return binary_op(other, BinaryOp::Pow); }

        template<typename T>
        Tensor mod(const T& other) const { return binary_op(other, BinaryOp::Mod); }

        template<typename T>
        Tensor maximum(const T& other) const { return binary_op(other, BinaryOp::Maximum); }

        template<typename T>
        Tensor minimum(const T& other) const { return binary_op(other, BinaryOp::Minimum); }

        // Comparison operations
        template<typename T>
        Tensor eq(const T& other) const { return binary_op(other, BinaryOp::Equal); }

        template<typename T>
        Tensor ne(const T& other) const { return binary_op(other, BinaryOp::NotEqual); }

        template<typename T>
        Tensor lt(const T& other) const { return binary_op(other, BinaryOp::Less); }

        template<typename T>
        Tensor le(const T& other) const { return binary_op(other, BinaryOp::LessEqual); }

        template<typename T>
        Tensor gt(const T& other) const { return binary_op(other, BinaryOp::Greater); }

        template<typename T>
        Tensor ge(const T& other) const { return binary_op(other, BinaryOp::GreaterEqual); }

        // Logical operations (Tensor only)
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

        Tensor sum(std::initializer_list<int> axes, bool keepdim = false) const {
            return sum(std::span<const int>(axes), keepdim);
        }

        Tensor sum(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return sum(std::span<const int>(axes), keepdim);
        }

        Tensor mean(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Mean, args);
        }

        Tensor mean(std::initializer_list<int> axes, bool keepdim = false) const {
            return mean(std::span<const int>(axes), keepdim);
        }

        Tensor mean(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return mean(std::span<const int>(axes), keepdim);
        }

        Tensor max(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Max, args);
        }

        Tensor max(std::initializer_list<int> axes, bool keepdim = false) const {
            return max(std::span<const int>(axes), keepdim);
        }

        Tensor max(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return max(std::span<const int>(axes), keepdim);
        }

        Tensor min(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Min, args);
        }

        Tensor min(std::initializer_list<int> axes, bool keepdim = false) const {
            return min(std::span<const int>(axes), keepdim);
        }

        Tensor min(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return min(std::span<const int>(axes), keepdim);
        }

        Tensor prod(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Prod, args);
        }

        Tensor prod(std::initializer_list<int> axes, bool keepdim = false) const {
            return prod(std::span<const int>(axes), keepdim);
        }

        Tensor prod(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return prod(std::span<const int>(axes), keepdim);
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

        Tensor std(std::initializer_list<int> axes, bool keepdim = false) const {
            return std(std::span<const int>(axes), keepdim);
        }

        Tensor std(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return std(std::span<const int>(axes), keepdim);
        }

        Tensor var(std::span<const int> axes = {}, bool keepdim = false) const {
            ReduceArgs args;
            args.axes = std::vector<int>(axes.begin(), axes.end());
            args.keepdim = keepdim;
            return reduce(ReduceOp::Var, args);
        }

        Tensor var(std::initializer_list<int> axes, bool keepdim = false) const {
            return var(std::span<const int>(axes), keepdim);
        }

        Tensor var(int dim, bool keepdim = false) const {
            std::vector<int> axes = {dim};
            return var(std::span<const int>(axes), keepdim);
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
        Tensor norm(float p, std::span<const int> dims, bool keepdim = false) const;
        Tensor norm(float p, std::initializer_list<int> dims, bool keepdim = false) const {
            return norm(p, std::span<const int>(dims), keepdim);
        }

        // Convenience methods
        Tensor norm(float p, int dim, bool keepdim = false) const {
            std::vector<int> dims_vec = {dim};
            return norm(p, std::span<const int>(dims_vec), keepdim);
        }
        float item() const;

        template <typename T>
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

        Tensor& clamp_(float min_val, float max_val);
        Tensor& clamp_min_(float min);
        Tensor& clamp_max_(float max);

        // In-place operations (Template-based)
        template<typename T>
        Tensor& add_(const T& other) { return binary_op_inplace(other, BinaryOp::Add); }

        template<typename T>
        Tensor& sub_(const T& other) { return binary_op_inplace(other, BinaryOp::Sub); }

        template<typename T>
        Tensor& mul_(const T& other) { return binary_op_inplace(other, BinaryOp::Mul); }

        template<typename T>
        Tensor& div_(const T& other) { return binary_op_inplace(other, BinaryOp::Div); }

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

        // ============= ADVANCED OPERATIONS =============

        // Pairwise distance
        Tensor cdist(const Tensor& other, float p = 2.0f) const;

        // Min/max with indices
        std::pair<Tensor, Tensor> min_with_indices(int dim = -1, bool keepdim = false) const;
        std::pair<Tensor, Tensor> max_with_indices(int dim = -1, bool keepdim = false) const;

        /**
         * Sort the tensor along a given dimension.
         *
         * Returns a pair of tensors:
         * - values: Sorted values (same dtype as input)
         * - indices: Int64 tensor containing the indices that would sort the input
         *
         * Example:
         *   auto t = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, Device::CPU);
         *   auto [sorted_vals, sorted_idx] = t.sort(0, false);
         *   // sorted_vals: [1.0, 2.0, 3.0] (Float32)
         *   // sorted_idx:  [1, 0, 2]       (Int64)
         *
         * @param dim Dimension to sort along (default: -1, last dimension)
         * @param descending If true, sort in descending order (default: false)
         * @return Pair of (sorted_values, indices). Indices are always Int64 dtype.
         */
        std::pair<Tensor, Tensor> sort(int dim = -1, bool descending = false) const;

        // Scalar boolean reductions
        bool any_scalar() const;
        bool all_scalar() const;

        // ============= OPERATOR OVERLOADS (Template-based) =============

        // Addition
        template<typename T>
        Tensor operator+(const T& other) const { return add(other); }

        // Subtraction
        template<typename T>
        Tensor operator-(const T& other) const { return sub(other); }

        // Multiplication
        template<typename T>
        Tensor operator*(const T& other) const { return mul(other); }

        // Division
        template<typename T>
        Tensor operator/(const T& other) const { return div(other); }

        // Modulo
        template<typename T>
        Tensor operator%(const T& other) const { return mod(other); }

        // Negation
        Tensor operator-() const { return neg(); }

        // Comparison operators
        template<typename T>
        Tensor operator==(const T& other) const { return eq(other); }

        template<typename T>
        Tensor operator!=(const T& other) const { return ne(other); }

        template<typename T>
        Tensor operator<(const T& other) const { return lt(other); }

        template<typename T>
        Tensor operator<=(const T& other) const { return le(other); }

        template<typename T>
        Tensor operator>(const T& other) const { return gt(other); }

        template<typename T>
        Tensor operator>=(const T& other) const { return ge(other); }

        // Logical operators (Tensor only)
        Tensor operator&&(const Tensor& other) const { return logical_and(other); }
        Tensor operator||(const Tensor& other) const { return logical_or(other); }
        Tensor operator!() const { return logical_not(); }

        Tensor operator~() const;
        Tensor operator|(const Tensor& other) const;

        // Other in-place operations
        Tensor& zero_();
        Tensor& fill_(float value);
        Tensor& copy_from(const Tensor& other);
        Tensor& copy_(const Tensor& src) { return copy_from(src); }
        Tensor& uniform_(float low = 0.0f, float high = 1.0f);
        Tensor& normal_(float mean = 0.0f, float std = 1.0f);

        std::optional<Tensor> try_reshape(TensorShape shape) const;

        static std::vector<Tensor> split_batch(const Tensor& tensor, size_t batch_size);

        // Utility template methods
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

        std::vector<int64_t> to_vector_int64() const;

        std::vector<int> to_vector_int() const;
        std::vector<bool> to_vector_bool() const;
        std::vector<float> debug_values(size_t max_values = 100) const;

        void dump_diagnostic(const std::string& filename) const;
        void log_info(const std::string& name = "") const;
        void print_formatted(const std::string& name = "", size_t max_per_dim = 10) const;

        // ============= TENSOR OPTIONS =============
        struct TensorOptions {
            Device device = Device::CUDA;
            DataType dtype = DataType::Float32;

            TensorOptions() = default;
            TensorOptions(Device dev) : device(dev) {}
            TensorOptions(DataType dt) : dtype(dt) {}
            TensorOptions(Device dev, DataType dt) : device(dev), dtype(dt) {}
        };

        TensorOptions options() const {
            return TensorOptions{device_, dtype_};
        }

    private:
        void print_1d(size_t max_elem = 10) const;
        void print_2d(size_t max_per_dim = 10) const;
        friend class TensorIndexer;
        friend class MaskedTensorProxy;
        friend class TensorRowProxy;
    };

    // ============= TensorRowProxy for operator[] =============
    class TensorRowProxy {
    private:
        Tensor* tensor_;
        size_t row_index_;

    public:
        TensorRowProxy(Tensor* tensor, size_t row_index)
            : tensor_(tensor), row_index_(row_index) {
            if (tensor_ && row_index_ >= tensor_->shape()[0]) {
                LOG_ERROR("Row index {} out of bounds for dimension 0 with size {}",
                          row_index_, tensor_->shape()[0]);
            }
        }

        // For 2D tensors: tensor[i][j]
        float& operator[](size_t col_index) {
            if (!tensor_) {
                LOG_ERROR("TensorRowProxy: null tensor pointer");
                static float dummy = 0.0f;
                return dummy;
            }

            if (tensor_->shape().rank() < 2) {
                LOG_ERROR("TensorRowProxy: tensor rank {} < 2", tensor_->shape().rank());
                static float dummy = 0.0f;
                return dummy;
            }

            if (col_index >= tensor_->shape()[1]) {
                LOG_ERROR("Column index {} out of bounds for dimension 1 with size {}",
                          col_index, tensor_->shape()[1]);
                static float dummy = 0.0f;
                return dummy;
            }

            return tensor_->at({row_index_, col_index});
        }

        float operator[](size_t col_index) const {
            if (!tensor_) {
                LOG_ERROR("TensorRowProxy: null tensor pointer");
                return 0.0f;
            }

            if (tensor_->shape().rank() < 2) {
                LOG_ERROR("TensorRowProxy: tensor rank {} < 2", tensor_->shape().rank());
                return 0.0f;
            }

            if (col_index >= tensor_->shape()[1]) {
                LOG_ERROR("Column index {} out of bounds for dimension 1 with size {}",
                          col_index, tensor_->shape()[1]);
                return 0.0f;
            }

            return tensor_->at({row_index_, col_index});
        }

        // For 1D tensors: tensor[i] - convert proxy to float
        operator float() const {
            if (!tensor_) {
                LOG_ERROR("TensorRowProxy: null tensor pointer");
                return 0.0f;
            }

            if (tensor_->shape().rank() != 1) {
                LOG_ERROR("Implicit conversion to float only valid for 1D tensors, got rank {}",
                          tensor_->shape().rank());
                return 0.0f;
            }

            return tensor_->at({row_index_});
        }

        operator float&() {
            if (!tensor_) {
                LOG_ERROR("TensorRowProxy: null tensor pointer");
                static float dummy = 0.0f;
                return dummy;
            }

            if (tensor_->shape().rank() != 1) {
                LOG_ERROR("Implicit conversion to float& only valid for 1D tensors, got rank {}",
                          tensor_->shape().rank());
                static float dummy = 0.0f;
                return dummy;
            }

            return tensor_->at({row_index_});
        }

        // Convert to Tensor (for operations on slices)
        operator Tensor() const {
            if (!tensor_) {
                LOG_ERROR("TensorRowProxy: null tensor pointer");
                return Tensor();
            }

            // For nD tensors where n > 1, return a slice
            if (tensor_->shape().rank() > 1) {
                return tensor_->slice(0, row_index_, row_index_ + 1).squeeze(0);
            }

            // For 1D tensors, return a scalar tensor
            float val = tensor_->at({row_index_});
            auto result = Tensor::empty({1}, tensor_->device(), tensor_->dtype());
            if (tensor_->device() == Device::CUDA) {
                cudaMemcpy(result.raw_ptr(), &val, sizeof(float), cudaMemcpyHostToDevice);
            } else {
                *result.ptr<float>() = val;
            }
            return result.squeeze();
        }

        // Extract scalar value with type specification
        template<typename T = float>
        T item() const {
            if (!tensor_) {
                LOG_ERROR("TensorRowProxy::item(): null tensor pointer");
                return T{};
            }

            // Convert to Tensor and call its item() method
            return Tensor(*this).item<T>();
        }

        // CRITICAL FIX #1: Enhanced assignment from Tensor
        TensorRowProxy& operator=(const Tensor& other) {
            if (!tensor_) {
                return *this;
            }

            if (tensor_->shape().rank() > 1) {
                // Multi-dimensional: assign entire row slice
                // Calculate the shape of one row (all dims except first)
                std::vector<size_t> slice_shape;
                for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                    slice_shape.push_back(tensor_->shape()[i]);
                }
                TensorShape expected_shape(slice_shape);

                // Validate shape match
                if (other.shape() != expected_shape) {
                    LOG_ERROR("Shape mismatch in row assignment: expected {}, got {}",
                             expected_shape.str(), other.shape().str());
                    return *this;
                }

                // Calculate number of elements per row
                size_t row_elements = 1;
                for (size_t i = 1; i < tensor_->shape().rank(); ++i) {
                    row_elements *= tensor_->shape()[i];
                }

                // Calculate byte offset for this row
                size_t byte_offset = row_index_ * row_elements * dtype_size(tensor_->dtype());
                size_t copy_bytes = row_elements * dtype_size(tensor_->dtype());

                // Ensure tensors are on same device
                auto other_same_device = (other.device() == tensor_->device())
                    ? other.clone()
                    : other.to(tensor_->device());

                // Copy data based on device
                if (tensor_->device() == Device::CUDA) {
                    cudaError_t err = cudaMemcpy(
                        static_cast<char*>(tensor_->raw_ptr()) + byte_offset,
                        other_same_device.raw_ptr(),
                        copy_bytes,
                        cudaMemcpyDeviceToDevice
                    );
                    if (err != cudaSuccess) {
                        LOG_ERROR("CUDA memcpy failed in row assignment: {}",
                                 cudaGetErrorString(err));
                    }
                } else {
                    std::memcpy(
                        static_cast<char*>(tensor_->raw_ptr()) + byte_offset,
                        other_same_device.raw_ptr(),
                        copy_bytes
                    );
                }
            } else {
                // 1D: assign single element
                if (other.numel() != 1) {
                    LOG_ERROR("Cannot assign tensor with {} elements to single position",
                             other.numel());
                    return *this;
                }

                float val = other.item();
                if (tensor_->device() == Device::CUDA) {
                    cudaMemcpy(
                        tensor_->ptr<float>() + row_index_,
                        &val,
                        sizeof(float),
                        cudaMemcpyHostToDevice
                    );
                } else {
                    tensor_->ptr<float>()[row_index_] = val;
                }
            }
            return *this;
        }

        // Assignment from float (for 1D tensors)
        TensorRowProxy& operator=(float value) {
            if (!tensor_) {
                return *this;
            }

            if (tensor_->shape().rank() != 1) {
                LOG_ERROR("Float assignment only valid for 1D tensors");
                return *this;
            }

            if (tensor_->device() == Device::CUDA) {
                cudaMemcpy(tensor_->ptr<float>() + row_index_, &value, sizeof(float),
                          cudaMemcpyHostToDevice);
            } else {
                tensor_->ptr<float>()[row_index_] = value;
            }
            return *this;
        }

        // ============= Arithmetic operations with TensorRowProxy =============

        Tensor operator-(const TensorRowProxy& other) const {
            return Tensor(*this).sub(Tensor(other));
        }

        Tensor operator+(const TensorRowProxy& other) const {
            return Tensor(*this).add(Tensor(other));
        }

        Tensor operator*(const TensorRowProxy& other) const {
            return Tensor(*this).mul(Tensor(other));
        }

        Tensor operator/(const TensorRowProxy& other) const {
            return Tensor(*this).div(Tensor(other));
        }

        // Arithmetic operations with scalars
        Tensor operator-(float scalar) const {
            return Tensor(*this).sub(scalar);
        }

        Tensor operator+(float scalar) const {
            return Tensor(*this).add(scalar);
        }

        Tensor operator*(float scalar) const {
            return Tensor(*this).mul(scalar);
        }

        Tensor operator/(float scalar) const {
            return Tensor(*this).div(scalar);
        }

        // Unary operations returning Tensor
        Tensor pow(float exponent) const {
            return Tensor(*this).pow(exponent);
        }

        Tensor sqrt() const {
            return Tensor(*this).sqrt();
        }

        Tensor abs() const {
            return Tensor(*this).abs();
        }

        Tensor sum() const {
            return Tensor(*this).sum();
        }

        Tensor mean() const {
            return Tensor(*this).mean();
        }

        Tensor square() const {
            return Tensor(*this).square();
        }

        Tensor neg() const {
            return Tensor(*this).neg();
        }

        Tensor operator-() const {
            return neg();
        }
    };

    // Implementation of Tensor::operator[]
    inline TensorRowProxy Tensor::operator[](size_t index) {
        if (index >= shape_[0]) {
            LOG_ERROR("Index {} out of bounds for dimension 0 with size {}", index, shape_[0]);
        }
        return TensorRowProxy(this, index);
    }

    inline const TensorRowProxy Tensor::operator[](size_t index) const {
        if (index >= shape_[0]) {
            LOG_ERROR("Index {} out of bounds for dimension 0 with size {}", index, shape_[0]);
        }
        return TensorRowProxy(const_cast<Tensor*>(this), index);
    }

    // Helper classes
    class MaskedTensorProxy {
    private:
        const Tensor* tensor_;
        Tensor mask_;

    public:
        MaskedTensorProxy(const Tensor* tensor, Tensor mask)
            : tensor_(tensor),
              mask_(std::move(mask)) {}

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
            : tensor_(tensor),
              indices_(std::move(indices)) {}

        void operator=(float value);
        void operator=(const Tensor& other);
        operator Tensor() const;
    };

    class TensorError : public std::runtime_error {
    public:
        TensorError(const std::string& msg, const Tensor* t = nullptr);
        const std::string& tensor_info() const { return tensor_info_; }

    private:
        std::string tensor_info_;
    };

    // ============= UTILITY NAMESPACES =============

    // Safe operations namespace
    namespace SafeOps {
        using Tensor = gs::Tensor;
        Tensor divide(const Tensor& a, const Tensor& b, float epsilon = 1e-6f);
        Tensor log(const Tensor& input, float epsilon = 1e-6f);
        Tensor sqrt(const Tensor& input, float epsilon = 0.0f);
    } // namespace SafeOps

    // Memory info
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

    // Functional operations namespace
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
    } // namespace functional

} // namespace gs