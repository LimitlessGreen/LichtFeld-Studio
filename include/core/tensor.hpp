/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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
        Bool = 5 // Added for masking
    };

    // Get size in bytes for each data type
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

    // Convert to string for logging
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

    // Boundary checking mode for indexing operations
    enum class BoundaryMode : uint8_t {
        Assert = 0, // Throw on out-of-bounds
        Clamp = 1,  // Clamp to valid range
        Wrap = 2    // Wrap around (modulo)
    };

    // Reduction mode for scatter operations
    enum class ScatterMode : uint8_t {
        None = 0,     // Direct assignment (last write wins)
        Add = 1,      // Add to existing value
        Multiply = 2, // Multiply with existing value
        Max = 3,      // Take maximum
        Min = 4       // Take minimum
    };

    class TensorShape {
    private:
        std::vector<size_t> dims_;
        size_t total_elements_ = 0;
        bool initialized_ = false; // Track if shape was explicitly set

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

        bool operator==(const TensorShape& other) const {
            return dims_ == other.dims_;
        }

        bool operator!=(const TensorShape& other) const {
            return !(*this == other);
        }

        // String representation for debugging
        std::string str() const;

    private:
        void compute_total() {
            total_elements_ = dims_.empty() ? 0 : 1;
            for (auto d : dims_) {
                total_elements_ *= d;
            }
        }
    };

    // Forward declarations
    class TensorError;
    class TensorIndexer;
    class MaskedTensorProxy;

    // Random number generator management
    class RandomGenerator {
    public:
        static RandomGenerator& instance();

        void manual_seed(uint64_t seed);
        uint64_t get_seed() const { return seed_; }

        // Get generator for a specific device
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

    // Lightweight tensor - a view over memory
    class Tensor {
    private:
        void* data_ = nullptr;
        TensorShape shape_;
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::Float32;
        bool owns_memory_ = false;
        bool initialized_ = false; // Track if tensor was properly initialized

        // For debugging
        mutable size_t id_ = 0;
        static std::atomic<size_t> next_id_;
        static inline bool profiling_enabled_ = false;

    public:
        // ============= Constructors & Destructor =============
        Tensor() = default;

        // Create from existing memory (non-owning view)
        Tensor(void* data, TensorShape shape, Device device, DataType dtype);

        // Move constructor
        Tensor(Tensor&& other) noexcept;

        // Move assignment
        Tensor& operator=(Tensor&& other) noexcept;

        // Delete copy (use clone() for explicit copies)
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        // Destructor
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

        // Boolean tensor creation
        static Tensor full_bool(TensorShape shape, bool value, Device device = Device::CUDA);
        static Tensor zeros_bool(TensorShape shape, Device device = Device::CUDA);
        static Tensor ones_bool(TensorShape shape, Device device = Device::CUDA);

        // Random tensors
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

        // Create view from raw memory
        static Tensor from_blob(void* data, TensorShape shape, Device device, DataType dtype) {
            return Tensor(data, shape, device, dtype);
        }

        // Boolean tensor element access
        void set_bool(std::initializer_list<size_t> indices, bool value);
        bool get_bool(std::initializer_list<size_t> indices) const;
        // Create from vector
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

        // Empty tensor: has 0 elements
        bool is_empty() const {
            // Default constructed tensors are considered empty
            if (!initialized_)
                return true;
            return shape_.elements() == 0;
        }

        // Valid tensor: is initialized (may be empty)
        bool is_valid() const { return initialized_; }

        size_t numel() const { return shape_.elements(); }
        size_t bytes() const { return numel() * dtype_size(dtype_); }

        // Shape queries
        size_t ndim() const { return shape_.rank(); }
        size_t size(size_t dim) const { return shape_[dim]; }

        // ============= Memory Operations =============
        Tensor clone() const;
        Tensor contiguous() const;
        Tensor to(Device device) const;
        Tensor to(DataType dtype) const;
        bool is_contiguous() const { return true; } // Always true for now

        // ============= Shape Operations =============
        Tensor reshape(TensorShape new_shape) const;
        Tensor view(TensorShape new_shape) const;
        Tensor slice(size_t dim, size_t start, size_t end) const;
        Tensor squeeze(int dim = -1) const;
        Tensor unsqueeze(int dim) const;
        Tensor permute(std::vector<int> dims) const;
        Tensor transpose(int dim1 = -2, int dim2 = -1) const;
        Tensor t() const; // Transpose last two dimensions
        Tensor flatten(int start_dim = 0, int end_dim = -1) const;

        // ============= Broadcasting Operations =============
        Tensor expand(const TensorShape& target_shape) const;
        Tensor broadcast_to(const TensorShape& target_shape) const;
        bool can_broadcast_to(const TensorShape& target) const;
        TensorShape broadcast_shape(const TensorShape& other) const;

        // ============= Arithmetic Operations =============
        // All element-wise operations support broadcasting
        Tensor add(float scalar) const;
        Tensor sub(float scalar) const;
        Tensor mul(float scalar) const;
        Tensor div(float scalar) const;
        Tensor neg() const;
        Tensor pow(float exponent) const;

        // Element-wise operations with broadcasting
        Tensor add(const Tensor& other) const;
        Tensor sub(const Tensor& other) const;
        Tensor mul(const Tensor& other) const;
        Tensor div(const Tensor& other) const;
        Tensor pow(const Tensor& other) const;

        // Matrix operations
        Tensor mm(const Tensor& other) const;     // Matrix multiply
        Tensor bmm(const Tensor& other) const;    // Batch matrix multiply
        Tensor matmul(const Tensor& other) const; // General matrix multiply
        Tensor dot(const Tensor& other) const;    // Dot product for 1D tensors

        // ============= Comparison Operations (NEW) =============
        Tensor eq(const Tensor& other) const; // Equal
        Tensor ne(const Tensor& other) const; // Not equal
        Tensor lt(const Tensor& other) const; // Less than
        Tensor le(const Tensor& other) const; // Less or equal
        Tensor gt(const Tensor& other) const; // Greater than
        Tensor ge(const Tensor& other) const; // Greater or equal

        Tensor eq(float value) const;
        Tensor ne(float value) const;
        Tensor lt(float value) const;
        Tensor le(float value) const;
        Tensor gt(float value) const;
        Tensor ge(float value) const;

        // Logical operations for boolean tensors
        Tensor logical_and(const Tensor& other) const;
        Tensor logical_or(const Tensor& other) const;
        Tensor logical_not() const;
        Tensor logical_xor(const Tensor& other) const;

        // ============= Masking Operations (NEW) =============
        Tensor masked_select(const Tensor& mask) const;
        Tensor& masked_fill_(const Tensor& mask, float value);
        Tensor masked_fill(const Tensor& mask, float value) const;
        Tensor where(const Tensor& condition, const Tensor& other) const;
        static Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

        // ============= Indexing Operations (NEW) =============
        Tensor index_select(int dim, const Tensor& indices) const;
        Tensor gather(int dim, const Tensor& indices) const;
        Tensor take(const Tensor& indices) const; // 1D indexing

        Tensor& scatter_(int dim, const Tensor& indices, const Tensor& src,
                         ScatterMode mode = ScatterMode::None);
        Tensor& scatter_(int dim, const Tensor& indices, float value,
                         ScatterMode mode = ScatterMode::None);
        Tensor& index_fill_(int dim, const Tensor& indices, float value);
        Tensor& index_copy_(int dim, const Tensor& indices, const Tensor& src);

        // Advanced indexing with boundary modes
        Tensor index_select(int dim, const Tensor& indices, BoundaryMode mode) const;
        Tensor gather(int dim, const Tensor& indices, BoundaryMode mode) const;

        // ============= Python-like Indexing (NEW) =============
        // Indexing proxy for Python-like syntax
        TensorIndexer operator[](const Tensor& indices);
        TensorIndexer operator[](const std::vector<Tensor>& indices);
        MaskedTensorProxy operator[](const Tensor& mask) const;

        // For scalar indexing
        float& at(std::initializer_list<size_t> indices);
        float at(std::initializer_list<size_t> indices) const;

        // Operator overloads
        Tensor operator+(const Tensor& other) const { return add(other); }
        Tensor operator+(float scalar) const { return add(scalar); }
        Tensor operator-(const Tensor& other) const { return sub(other); }
        Tensor operator-(float scalar) const { return sub(scalar); }
        Tensor operator*(const Tensor& other) const { return mul(other); }
        Tensor operator*(float scalar) const { return mul(scalar); }
        Tensor operator/(const Tensor& other) const { return div(other); }
        Tensor operator/(float scalar) const { return div(scalar); }
        Tensor operator-() const { return neg(); }

        // Comparison operator overloads
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

        // Logical operator overloads (for boolean tensors)
        Tensor operator&&(const Tensor& other) const { return logical_and(other); }
        Tensor operator||(const Tensor& other) const { return logical_or(other); }
        Tensor operator!() const { return logical_not(); }

        // In-place operations (no broadcasting for in-place)
        Tensor& add_(const Tensor& other);
        Tensor& add_(float scalar);
        Tensor& sub_(const Tensor& other);
        Tensor& sub_(float scalar);
        Tensor& mul_(const Tensor& other);
        Tensor& mul_(float scalar);
        Tensor& div_(const Tensor& other);
        Tensor& div_(float scalar);
        Tensor& zero_();                        // Fill with zeros in-place
        Tensor& fill_(float value);             // Fill tensor with a value in-place
        Tensor& copy_from(const Tensor& other); // Copy data from another tensor

        // In-place random operations
        Tensor& uniform_(float low = 0.0f, float high = 1.0f);
        Tensor& normal_(float mean = 0.0f, float std = 1.0f);

        // Additional utility methods that tests expect:
        std::expected<Tensor, std::string> try_reshape(TensorShape shape) const; // Safe reshape with error handling

        // Static utility methods:
        static std::vector<Tensor> split_batch(const Tensor& tensor, size_t batch_size);
        static void enable_profiling(bool enable) { profiling_enabled_ = enable; }

        // Functional-style methods:
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
        float item() const; // Get single value (for scalar tensors)
        std::pair<float, float> minmax() const;

        // Count non-zero elements (useful for masks)
        size_t count_nonzero() const;

        // ============= Common Deep Learning Operations =============
        Tensor normalize(int dim = -1, float eps = 1e-12f) const;
        Tensor abs() const;
        Tensor sqrt() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sigmoid() const;
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
        bool any() const; // For boolean tensors
        bool all() const; // For boolean tensors

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

    // ============= Indexing Helper Classes (NEW) =============

    // Proxy class for masked tensor assignment
    class MaskedTensorProxy {
    private:
        const Tensor* tensor_;
        Tensor mask_;

    public:
        MaskedTensorProxy(const Tensor* tensor, Tensor mask)
            : tensor_(tensor),
              mask_(std::move(mask)) {}

        // Assignment operators for Python-like syntax
        void operator=(float value);
        void operator=(const Tensor& other);

        // Conversion to tensor for selection
        operator Tensor() const;
    };

    // Indexer for advanced indexing
    class TensorIndexer {
    private:
        Tensor* tensor_;
        std::vector<Tensor> indices_;

    public:
        TensorIndexer(Tensor* tensor, std::vector<Tensor> indices)
            : tensor_(tensor),
              indices_(std::move(indices)) {}

        // Assignment for indexed positions
        void operator=(float value);
        void operator=(const Tensor& other);

        // Get indexed values
        operator Tensor() const;
    };

    // ============= TensorBuilder Class =============
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

    // ============= Tensor Utilities Namespace =============
    namespace tensor {
        // Creation functions
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

        // Random functions
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

        // Set random seed
        inline void manual_seed(uint64_t seed) {
            RandomGenerator::instance().manual_seed(seed);
        }

        // Like operations - declarations only, implemented in tensor_utils.cpp
        Tensor zeros_like(const Tensor& other);
        Tensor ones_like(const Tensor& other);
        Tensor rand_like(const Tensor& other);
        Tensor randn_like(const Tensor& other);

        // Matrix creation helpers
        Tensor eye(size_t n, Device device = Device::CUDA);
        Tensor eye(size_t m, size_t n, Device device = Device::CUDA);
        Tensor diag(const Tensor& diagonal);

        // Range operations
        Tensor arange(float end);
        Tensor arange(float start, float end, float step = 1.0f);
        Tensor linspace(float start, float end, size_t steps);

        // Stack/concatenate operations - using rvalue references for move semantics
        Tensor stack(std::vector<Tensor>&& tensors, int dim = 0);
        Tensor cat(std::vector<Tensor>&& tensors, int dim = 0);

        // Utility functions
        bool check_valid(const Tensor& t, const std::string& name);
        void assert_same_shape(const Tensor& a, const Tensor& b);
        void assert_same_device(const Tensor& a, const Tensor& b);
    } // namespace tensor

    // ============= Tensor Error Handling =============
    class TensorError : public std::runtime_error {
    public:
        TensorError(const std::string& msg, const Tensor* t = nullptr);
        const std::string& tensor_info() const { return tensor_info_; }

    private:
        std::string tensor_info_;
    };

    // ============= Safe Operations Namespace =============
    namespace SafeOps {
        using Tensor = gs::Tensor;

        // Safe division with epsilon to avoid division by zero
        Tensor divide(const Tensor& a, const Tensor& b, float epsilon = 1e-6f);

        // Safe log with clamping to avoid NaN
        Tensor log(const Tensor& input, float epsilon = 1e-6f);

        // Safe sqrt with clamping to avoid NaN
        Tensor sqrt(const Tensor& input, float epsilon = 0.0f);
    } // namespace SafeOps

    // ============= Memory Info Class =============
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

    // ============= Functional Operations Namespace =============
    namespace functional {
        // Map a function over tensor elements
        Tensor map(const Tensor& input, std::function<float(float)> func);

        // Reduce tensor with binary operation
        float reduce(const Tensor& input, float init, std::function<float(float, float)> func);

        // Filter tensor elements based on predicate (returns mask)
        Tensor filter(const Tensor& input, std::function<bool(float)> predicate);

        // Pipe multiple operations
        template <typename... Funcs>
        auto pipe(Funcs... funcs) {
            return [=](const Tensor& input) -> Tensor {
                Tensor result = input;
                ((result = funcs(result)), ...);
                return result;
            };
        }
    } // namespace functional

} // namespace gs
