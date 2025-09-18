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

namespace gs {

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

    // Type promotion for mixed dtype operations
    inline DataType promote_types(DataType a, DataType b) {
        if (a == b) return a;

        // Promotion hierarchy: Bool -> Int32 -> Float32
        // Bool -> Float32 is also allowed

        // Handle Bool promotion
        if (a == DataType::Bool) {
            if (b == DataType::Float32 || b == DataType::Float16) return b;
            if (b == DataType::Int32 || b == DataType::Int64) return b;
            return DataType::Float32; // Default
        }
        if (b == DataType::Bool) {
            if (a == DataType::Float32 || a == DataType::Float16) return a;
            if (a == DataType::Int32 || a == DataType::Int64) return a;
            return DataType::Float32; // Default
        }

        // Int -> Float promotion
        if ((a == DataType::Int32 || a == DataType::Int64) &&
            (b == DataType::Float32 || b == DataType::Float16)) {
            return (b == DataType::Float16) ? DataType::Float16 : DataType::Float32;
        }
        if ((b == DataType::Int32 || b == DataType::Int64) &&
            (a == DataType::Float32 || a == DataType::Float16)) {
            return (a == DataType::Float16) ? DataType::Float16 : DataType::Float32;
        }

        // Int32 -> Int64 promotion
        if ((a == DataType::Int32 && b == DataType::Int64) ||
            (a == DataType::Int64 && b == DataType::Int32)) {
            return DataType::Int64;
        }

        // Float16 -> Float32 promotion
        if ((a == DataType::Float16 && b == DataType::Float32) ||
            (a == DataType::Float32 && b == DataType::Float16)) {
            return DataType::Float32;
        }

        // Default to Float32 for safety
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

    // Comprehensive Binary operation types for unified dispatch
    enum class BinaryOp : uint8_t {
        // Arithmetic
        Add = 0,
        Sub = 1,
        Mul = 2,
        Div = 3,
        Pow = 4,
        Mod = 5,

        // Comparison
        Equal = 6,
        NotEqual = 7,
        Less = 8,
        LessEqual = 9,
        Greater = 10,
        GreaterEqual = 11,

        // Logical (for bool tensors)
        LogicalAnd = 12,
        LogicalOr = 13,
        LogicalXor = 14,

        // Min/Max
        Maximum = 15,
        Minimum = 16,

        // Bitwise (for integer tensors)
        BitwiseAnd = 17,
        BitwiseOr = 18,
        BitwiseXor = 19,
        LeftShift = 20,
        RightShift = 21
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

    class TensorError;
    class TensorIndexer;
    class MaskedTensorProxy;

    class RandomGenerator {
    public:
        static RandomGenerator& instance();
        void manual_seed(uint64_t seed);
        uint64_t get_seed() const { return seed_; }
        void* get_generator(Device device);

    private:
        RandomGenerator();
        ~RandomGenerator();
        uint64_t seed_;
        void* cuda_generator_ = nullptr;
        std::mt19937_64 cpu_generator_;
        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    };

    class Tensor {
    private:
        void* data_ = nullptr;
        TensorShape shape_;
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::Float32;
        bool owns_memory_ = false;
        bool initialized_ = false;

        mutable size_t id_ = 0;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

        // Concepts for arithmetic operations
        template<typename T>
        static constexpr bool is_scalar_v = std::is_arithmetic_v<T>;

        template<typename T>
        static constexpr bool is_tensor_v = std::is_same_v<std::remove_cvref_t<T>, Tensor>;

        // Unified binary operation implementation
        template<typename T>
        Tensor binary_op_impl(const T& other, BinaryOp op) const;

        template<typename T>
        Tensor& binary_op_inplace_impl(const T& other, BinaryOp op);

        // Broadcasting and type promotion helper
        std::pair<Tensor, Tensor> _broadcasted(const Tensor& other, bool match_dtype = true) const;

        template<typename T>
        std::pair<Tensor, Tensor> _broadcasted(const T& scalar) const;

    public:
        // ============= Constructors & Destructor =============
        Tensor() = default;
        Tensor(void* data, TensorShape shape, Device device, DataType dtype);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        ~Tensor();

        // ============= Factory Methods =============
        static Tensor empty(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32);
        static Tensor zeros(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32);
        static Tensor ones(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32);
        static Tensor full(TensorShape shape, float value, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32);

        static Tensor full_bool(TensorShape shape, bool value, Device device = Device::CUDA);
        static Tensor zeros_bool(TensorShape shape, Device device = Device::CUDA);
        static Tensor ones_bool(TensorShape shape, Device device = Device::CUDA);

        static Tensor rand(TensorShape shape, Device device = Device::CUDA,
                           DataType dtype = DataType::Float32);
        static Tensor randn(TensorShape shape, Device device = Device::CUDA,
                            DataType dtype = DataType::Float32);
        static Tensor uniform(TensorShape shape, float low = 0.0f, float high = 1.0f,
                              Device device = Device::CUDA, DataType dtype = DataType::Float32);
        static Tensor normal(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                             Device device = Device::CUDA, DataType dtype = DataType::Float32);
        static Tensor randint(TensorShape shape, int low, int high,
                              Device device = Device::CUDA, DataType dtype = DataType::Int32);
        static Tensor bernoulli(TensorShape shape, float p = 0.5f,
                                Device device = Device::CUDA, DataType dtype = DataType::Float32);
        static Tensor multinomial(const Tensor& weights, int num_samples,
                         bool replacement = false);

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

        // ============= Data Access =============
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

        // ============= Properties =============
        const TensorShape& shape() const { return shape_; }
        Device device() const { return device_; }
        DataType dtype() const { return dtype_; }
        bool owns_memory() const { return owns_memory_; }

        bool is_empty() const {
            if (!initialized_) return true;
            return shape_.elements() == 0;
        }

        bool is_valid() const { return initialized_; }

        size_t numel() const { return shape_.elements(); }
        size_t bytes() const { return numel() * dtype_size(dtype_); }

        size_t ndim() const { return shape_.rank(); }
        size_t size(size_t dim) const { return shape_[dim]; }

        // ============= Memory Operations =============
        Tensor clone() const;
        Tensor contiguous() const;
        Tensor to(Device device) const;
        Tensor to(DataType dtype) const;
        bool is_contiguous() const { return true; }

        // ============= Shape Operations =============
        Tensor reshape(TensorShape new_shape) const;
        Tensor view(TensorShape new_shape) const;
        Tensor slice(size_t dim, size_t start, size_t end) const;
        Tensor squeeze(int dim = -1) const;
        Tensor unsqueeze(int dim) const;
        Tensor permute(std::vector<int> dims) const;
        Tensor transpose(int dim1 = -2, int dim2 = -1) const;
        Tensor t() const;
        Tensor flatten(int start_dim = 0, int end_dim = -1) const;

        // ============= Broadcasting Operations =============
        Tensor expand(const TensorShape& target_shape) const;
        Tensor broadcast_to(const TensorShape& target_shape) const;
        bool can_broadcast_to(const TensorShape& target) const;
        TensorShape broadcast_shape(const TensorShape& other) const;

        // ============= Unified Arithmetic Operations =============
        // Arithmetic operations
        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor add(const T& other) const { return binary_op_impl(other, BinaryOp::Add); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor sub(const T& other) const { return binary_op_impl(other, BinaryOp::Sub); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor mul(const T& other) const { return binary_op_impl(other, BinaryOp::Mul); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor div(const T& other) const { return binary_op_impl(other, BinaryOp::Div); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor pow(const T& other) const { return binary_op_impl(other, BinaryOp::Pow); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor mod(const T& other) const { return binary_op_impl(other, BinaryOp::Mod); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor maximum(const T& other) const { return binary_op_impl(other, BinaryOp::Maximum); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor minimum(const T& other) const { return binary_op_impl(other, BinaryOp::Minimum); }

        Tensor neg() const { return mul(-1.0f); }

        // Comparison operations - now use unified implementation
        Tensor eq(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Equal); }
        Tensor ne(const Tensor& other) const { return binary_op_impl(other, BinaryOp::NotEqual); }
        Tensor lt(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Less); }
        Tensor le(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LessEqual); }
        Tensor gt(const Tensor& other) const { return binary_op_impl(other, BinaryOp::Greater); }
        Tensor ge(const Tensor& other) const { return binary_op_impl(other, BinaryOp::GreaterEqual); }

        Tensor eq(float value) const { return binary_op_impl(value, BinaryOp::Equal); }
        Tensor ne(float value) const { return binary_op_impl(value, BinaryOp::NotEqual); }
        Tensor lt(float value) const { return binary_op_impl(value, BinaryOp::Less); }
        Tensor le(float value) const { return binary_op_impl(value, BinaryOp::LessEqual); }
        Tensor gt(float value) const { return binary_op_impl(value, BinaryOp::Greater); }
        Tensor ge(float value) const { return binary_op_impl(value, BinaryOp::GreaterEqual); }

        // Logical operations for bool tensors
        Tensor logical_and(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LogicalAnd); }
        Tensor logical_or(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LogicalOr); }
        Tensor logical_xor(const Tensor& other) const { return binary_op_impl(other, BinaryOp::LogicalXor); }
        Tensor logical_not() const;

        // In-place operations
        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor& add_(const T& other) { return binary_op_inplace_impl(other, BinaryOp::Add); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor& sub_(const T& other) { return binary_op_inplace_impl(other, BinaryOp::Sub); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor& mul_(const T& other) { return binary_op_inplace_impl(other, BinaryOp::Mul); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor& div_(const T& other) { return binary_op_inplace_impl(other, BinaryOp::Div); }

        // Matrix operations
        Tensor mm(const Tensor& other) const;
        Tensor bmm(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor dot(const Tensor& other) const;

        static Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
        Tensor cat(const Tensor& other, int dim = 0) const;

        // ============= Masking Operations =============
        Tensor masked_select(const Tensor& mask) const;
        Tensor& masked_fill_(const Tensor& mask, float value);
        Tensor masked_fill(const Tensor& mask, float value) const;
        Tensor where(const Tensor& condition, const Tensor& other) const;
        static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

        // ============= Indexing Operations =============
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

        // Operator overloads - use unified implementation
        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor operator+(const T& other) const { return add(other); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor operator-(const T& other) const { return sub(other); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor operator*(const T& other) const { return mul(other); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor operator/(const T& other) const { return div(other); }

        template<typename T> requires (is_scalar_v<T> || is_tensor_v<T>)
        Tensor operator%(const T& other) const { return mod(other); }

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

        // Other in-place operations
        Tensor& zero_();
        Tensor& fill_(float value);
        Tensor& copy_from(const Tensor& other);

        Tensor& uniform_(float low = 0.0f, float high = 1.0f);
        Tensor& normal_(float mean = 0.0f, float std = 1.0f);

        // Changed from std::expected to std::optional for CUDA compatibility
        std::optional<Tensor> try_reshape(TensorShape shape) const;

        static std::vector<Tensor> split_batch(const Tensor& tensor, size_t batch_size);
        static void enable_profiling(bool enable) { profiling_enabled_ = enable; }

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

        // ============= Reduction Operations =============
        float sum() const;
        float mean() const;
        float min() const;
        float max() const;
        float std(float eps = 1e-8f) const;
        float var(float eps = 1e-8f) const;
        float norm(float p = 2.0f) const;
        float item() const;
        std::pair<float, float> minmax() const;

        size_t count_nonzero() const;

        // ============= Common Deep Learning Operations =============
        Tensor normalize(int dim = -1, float eps = 1e-12f) const;
        Tensor abs() const;
        Tensor sqrt() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sigmoid() const;
        Tensor logit(float eps = 1e-7f) const;
        Tensor relu() const;
        Tensor clamp(float min, float max) const;
        Tensor clamp_min(float min) const;
        Tensor clamp_max(float max) const;

        // ============= Validation & Assertions =============
        Tensor& assert_shape(TensorShape expected, const std::string& msg = "");
        Tensor& assert_device(Device expected);
        Tensor& assert_dtype(DataType expected);
        Tensor& assert_finite();

        // ============= Comparison Operations =============
        bool has_nan() const;
        bool has_inf() const;
        bool all_close(const Tensor& other, float rtol = 1e-5f, float atol = 1e-8f) const;
        bool any() const;
        bool all() const;

        // ============= Utility Functions =============
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

    // Helper classes remain the same
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
        inline Tensor empty(TensorShape shape, Device device = Device::CUDA) {
            return Tensor::empty(shape, device);
        }

        inline Tensor zeros(TensorShape shape, Device device = Device::CUDA) {
            return Tensor::zeros(shape, device);
        }

        inline Tensor ones(TensorShape shape, Device device = Device::CUDA) {
            return Tensor::ones(shape, device);
        }

        inline Tensor full(TensorShape shape, float value, Device device = Device::CUDA) {
            return Tensor::full(shape, value, device);
        }

        inline Tensor rand(TensorShape shape, Device device = Device::CUDA) {
            return Tensor::rand(shape, device);
        }

        inline Tensor randn(TensorShape shape, Device device = Device::CUDA) {
            return Tensor::randn(shape, device);
        }

        inline Tensor uniform(TensorShape shape, float low = 0.0f, float high = 1.0f,
                              Device device = Device::CUDA) {
            return Tensor::uniform(shape, low, high, device);
        }

        inline Tensor normal(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                             Device device = Device::CUDA) {
            return Tensor::normal(shape, mean, std, device);
        }

        inline Tensor randint(TensorShape shape, int low, int high, Device device = Device::CUDA) {
            return Tensor::randint(shape, low, high, device);
        }

        inline Tensor bernoulli(TensorShape shape, float p = 0.5f, Device device = Device::CUDA) {
            return Tensor::bernoulli(shape, p, device);
        }

        inline void manual_seed(uint64_t seed) {
            RandomGenerator::instance().manual_seed(seed);
        }

        Tensor zeros_like(const Tensor& other);
        Tensor ones_like(const Tensor& other);
        Tensor rand_like(const Tensor& other);
        Tensor randn_like(const Tensor& other);

        Tensor eye(size_t n, Device device = Device::CUDA);
        Tensor eye(size_t m, size_t n, Device device = Device::CUDA);
        Tensor diag(const Tensor& diagonal);

        Tensor arange(float end);
        Tensor arange(float start, float end, float step = 1.0f);
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
                // Use a temporary to avoid copying - work with rvalue
                Tensor result = input.clone();  // Start with a copy
                ((result = funcs(std::move(result))), ...);  // Use move semantics
                return result;
            };
        }
    }

} // namespace gs