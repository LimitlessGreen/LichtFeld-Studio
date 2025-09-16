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
        UInt8 = 4
    };

    // Get size in bytes for each data type
    constexpr size_t dtype_size(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
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
        default: return "unknown";
        }
    }

    inline const char* device_name(Device device) {
        return device == Device::CPU ? "cpu" : "cuda";
    }

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

    // Forward declaration
    class TensorError;

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

        // Valid tensor: properly initialized and either has data or is truly empty (0 elements)
        bool is_valid() const {
            // Default constructed tensors are invalid
            if (!initialized_)
                return false;

            // Empty tensors (0 elements) are valid even without data
            if (shape_.elements() == 0)
                return true;

            // Non-empty tensors need valid data pointer
            return data_ != nullptr;
        }

        bool is_contiguous() const { return true; } // For now, assume all tensors are contiguous
        size_t numel() const { return shape_.elements(); }
        size_t bytes() const { return shape_.elements() * dtype_size(dtype_); }
        size_t id() const { return id_; }

        // ============= View Operations (No Copy) =============
        Tensor view(TensorShape new_shape) const;
        Tensor reshape(TensorShape new_shape) const { return view(new_shape); }
        Tensor slice(size_t dim, size_t start, size_t end) const;
        Tensor squeeze(int dim = -1) const;
        Tensor unsqueeze(int dim) const;
        Tensor permute(std::vector<int> dims) const;
        Tensor transpose(int dim1 = 0, int dim2 = 1) const;
        Tensor flatten(int start_dim = 0, int end_dim = -1) const;

        // New transpose that returns a proper transposed tensor
        Tensor t() const; // Transpose last two dimensions

        // ============= Memory Operations =============
        Tensor to(Device device) const;      // Copy to device
        Tensor clone() const;                // Deep copy
        void copy_from(const Tensor& other); // Copy data from another tensor
        void fill_(float value);             // Fill with value (in-place)
        void zero_();                        // Fill with zeros (in-place)

        // ============= Basic Math Operations =============
        Tensor add(const Tensor& other) const;
        Tensor add(float scalar) const;
        Tensor sub(const Tensor& other) const;
        Tensor sub(float scalar) const;
        Tensor mul(const Tensor& other) const;
        Tensor mul(float scalar) const;
        Tensor div(const Tensor& other) const;
        Tensor div(float scalar) const;
        Tensor neg() const;

        // Matrix operations
        Tensor matmul(const Tensor& other) const;
        Tensor mm(const Tensor& other) const { return matmul(other); } // Alias
        Tensor bmm(const Tensor& other) const;                         // Batch matrix multiply
        Tensor dot(const Tensor& other) const;                         // Dot product for 1D tensors

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
        // Note: operator@ is not valid C++, use matmul() or mm() instead

        // In-place operations
        Tensor& add_(const Tensor& other);
        Tensor& add_(float scalar);
        Tensor& sub_(const Tensor& other);
        Tensor& sub_(float scalar);
        Tensor& mul_(const Tensor& other);
        Tensor& mul_(float scalar);
        Tensor& div_(const Tensor& other);
        Tensor& div_(float scalar);

        // In-place random operations
        Tensor& uniform_(float low = 0.0f, float high = 1.0f);
        Tensor& normal_(float mean = 0.0f, float std = 1.0f);

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

        // ============= Quick Checks =============
        bool has_nan() const;
        bool has_inf() const;
        bool all_close(const Tensor& other, float rtol = 1e-5f, float atol = 1e-8f) const;

        // ============= Debug Utilities =============
        std::string str() const;
        void print(const std::string& name = "") const;
        void print_formatted(const std::string& name = "", size_t max_per_dim = 6) const;
        std::vector<float> debug_values(size_t max_values = 10) const;
        std::vector<float> to_vector() const;
        void dump_diagnostic(const std::string& filename) const;

        // ============= Performance Monitoring =============
        static void enable_profiling(bool enable = true) { profiling_enabled_ = enable; }

        template <typename Func>
        auto timed(const std::string& name, Func func) -> decltype(func(*this));

        // ============= Batch Operations =============
        static std::vector<Tensor> split_batch(const Tensor& t, size_t batch_size);

        // ============= Error Handling =============
        std::expected<Tensor, std::string> try_reshape(TensorShape new_shape) const;

        // ============= Chainable Operations =============
        Tensor& inplace(std::function<void(Tensor&)> op) {
            op(*this);
            return *this;
        }

        template <typename Op>
        Tensor apply(Op op) const {
            return op(*this);
        }

    private:
        void print_1d(size_t max_elem) const;
        void print_2d(size_t max_per_dim) const;
    };

    // ============= Convenience Namespace =============
    namespace tensor {
        // Factory functions
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

        inline Tensor from_blob(void* data, TensorShape shape,
                                Device device = Device::CUDA,
                                DataType dtype = DataType::Float32) {
            return Tensor::from_blob(data, shape, device, dtype);
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

        // Like operations
        inline Tensor zeros_like(const Tensor& other) {
            return zeros(other.shape(), other.device());
        }

        inline Tensor ones_like(const Tensor& other) {
            return ones(other.shape(), other.device());
        }

        inline Tensor rand_like(const Tensor& other) {
            return rand(other.shape(), other.device());
        }

        inline Tensor randn_like(const Tensor& other) {
            return randn(other.shape(), other.device());
        }

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
        void assert_shape(const Tensor& t, TensorShape expected, const std::string& name);

        // Builder pattern
        struct TensorBuilder {
            TensorShape shape;
            Device device = Device::CUDA;
            DataType dtype = DataType::Float32;
            std::optional<float> fill_value;
            bool check_finite = false;

            TensorBuilder& with_shape(TensorShape s) {
                shape = s;
                return *this;
            }
            TensorBuilder& on_device(Device d) {
                device = d;
                return *this;
            }
            TensorBuilder& with_dtype(DataType d) {
                dtype = d;
                return *this;
            }
            TensorBuilder& filled_with(float v) {
                fill_value = v;
                return *this;
            }
            TensorBuilder& ensure_finite() {
                check_finite = true;
                return *this;
            }

            Tensor build();
        };

        // Safe operations
        struct SafeOps {
            static Tensor divide(const Tensor& a, const Tensor& b, float eps = 1e-8f);
            static Tensor log(const Tensor& t, float eps = 1e-8f);
            static Tensor sqrt(const Tensor& t, float eps = 1e-8f);
        };

        // Memory info
        struct MemoryInfo {
            size_t allocated_bytes;
            size_t reserved_bytes;
            size_t free_bytes;

            static MemoryInfo cuda();
            void log() const;
        };

        // Batch processing
        template <typename Func>
        Tensor apply_batched(const Tensor& input, size_t batch_size, Func func);
    } // namespace tensor

    // ============= Functional Programming Support =============
    namespace tensor::functional {
        template <typename Func>
        Tensor map(const Tensor& input, Func func);

        template <typename Func>
        float reduce(const Tensor& input, float init, Func func);

        template <typename Pred>
        Tensor filter(const Tensor& input, Pred predicate);

        template <typename... Ops>
        class Pipeline {
            std::tuple<Ops...> ops_;

        public:
            explicit Pipeline(Ops... ops) : ops_(ops...) {}

            Tensor operator()(const Tensor& input) const {
                return apply_impl(input, std::index_sequence_for<Ops...>{});
            }

        private:
            template <size_t... Is>
            Tensor apply_impl(const Tensor& input, std::index_sequence<Is...>) const {
                Tensor result = input.clone(); // Use clone instead of copy
                ((result = std::get<Is>(ops_)(result)), ...);
                return result;
            }
        };

        template <typename... Ops>
        Pipeline<Ops...> pipe(Ops... ops) {
            return Pipeline<Ops...>(ops...);
        }
    } // namespace tensor::functional

    // ============= Error Classes =============
    class TensorError : public std::runtime_error {
    public:
        TensorError(const std::string& msg, const Tensor* t = nullptr);
        const std::string& tensor_info() const { return tensor_info_; }

    private:
        std::string tensor_info_;
    };

    // ============= Performance Monitoring =============
    class TensorTimer {
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point start_;
        std::string name_;

    public:
        explicit TensorTimer(std::string name);
        ~TensorTimer();
    };

    // ============= Stream Output =============
    std::ostream& operator<<(std::ostream& os, const Tensor& t);
    std::ostream& operator<<(std::ostream& os, const TensorShape& shape);

// ============= Debug Macros =============
#ifdef DEBUG
#define TENSOR_DEBUG(tensor)                                          \
    do {                                                              \
        LOG_DEBUG("Tensor {} at {}:{}", #tensor, __FILE__, __LINE__); \
        (tensor).print(#tensor);                                      \
    } while (0)

#define TENSOR_ASSERT_SHAPE(tensor, ...)                        \
    do {                                                        \
        TensorShape expected{__VA_ARGS__};                      \
        if ((tensor).shape() != expected) {                     \
            LOG_ERROR("Shape assertion failed for {} at {}:{}", \
                      #tensor, __FILE__, __LINE__);             \
            LOG_ERROR("  Expected: {}, Got: {}",                \
                      expected.str(), (tensor).shape().str());  \
            assert(false);                                      \
        }                                                       \
    } while (0)

#define TENSOR_TIME(name) TensorTimer _timer##__LINE__(name)
#else
#define TENSOR_DEBUG(tensor)             ((void)0)
#define TENSOR_ASSERT_SHAPE(tensor, ...) ((void)0)
#define TENSOR_TIME(name)                ((void)0)
#endif

} // namespace gs
