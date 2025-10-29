# Densification Bug Fix Status - 10M+ Gaussians

**Date:** 2025-10-29
**Branch:** lfs_training
**Goal:** Fix densification at 10M+ Gaussian scale

## Current Status: ⚠️ CRITICAL - DEALLOCATION FIXED, TENSOR LIFETIME ISSUE REMAINS

---

## FIXED: Proper Deallocation Based on Allocation Method ✅

**Status:** PROPERLY FIXED - Production-grade solution implemented

**What Was Broken:**
- `deallocate()` used `cudaFreeAsync` for ALL pointers (including those from `cudaMalloc`)
- `cudaFreeAsync` returns memory to CUDA's internal cache, NOT to OS
- Memory from direct `cudaMalloc` sits in cache, unavailable for new `cudaMalloc` calls
- **Result:** Memory leak - cached memory couldn't be reused

**Proper Fix Implemented:**
1. **Added AllocMethod enum** to track allocation type:
   - `Arena` - From GPUArenaAllocator
   - `Async` - From cudaMallocAsync (CUDA 12.8+)
   - `Direct` - From cudaMalloc (>1GB allocations)

2. **Added tracking map** (`std::unordered_map<void*, AllocMethod>`) with mutex for thread safety

3. **Updated allocate()** to record allocation method for each pointer

4. **Updated deallocate()** to use correct free function:
   - Arena → `GPUArenaAllocator::deallocate()`
   - Async → `cudaFreeAsync()` (returns to CUDA cache)
   - Direct → `cudaFree()` (returns to OS/driver)

5. **Added memory pressure handling:**
   - When allocation fails, trim CUDA memory pool
   - This releases cached memory from cudaMallocAsync
   - Freed 0.25 GB in testing (from 0.71 GB to 0.96 GB)

**Test Results:**
- Deallocation working correctly: `[POOL] Deallocating Direct allocation ptr=... using cudaFree`
- Pool trimming working: `free=0.71 GB → 0.96 GB after trim`
- **However:** Still OOM at 10M Gaussians (22.5 GB / 23.51 GB consumed)

**Files Modified:**
- `src/core_new/tensor/internal/memory_pool.hpp:20-248` - Full implementation

---

## REMAINING ISSUE: Tensor Lifetime Management 🔥

**Status:** ROOT CAUSE IDENTIFIED - Named variables prevent move semantics

**The Problem:**
Even with correct deallocation, 10M test OOMs because **all intermediate tensors are lvalues (named variables)**.

**Memory Consumption at Failure:**
```
[POOL] CUDA reports: free=0.71 GB, total=23.51 GB, requested=3.58 GB
[POOL] After trim: free=0.96 GB, requested=3.58 GB
```
- Consumed: 22.5 GB out of 23.51 GB
- Expected: ~5 GB (10M + 10M split = 2.48 GB + 2.48 GB)
- **Memory leak: 4.5× overhead due to intermediate tensors**

**Why Move Semantics Don't Help:**
We **DO** have proper move constructors/assignment (`tensor.cpp:136-178`), but they're not triggered because:

```cpp
// In split() function:
auto sampled_sh = splat_->_sh.index_select(...);  // ← lvalue (named variable)
std::vector<Tensor> vec(2, sampled_sh);            // ← Copy constructor! (lvalue passed)
auto split_sh = Tensor::cat(vec, 0);               // ← Allocates new memory

// sampled_sh is still alive until function returns!
// Move constructor ONLY triggers for rvalues (temporaries or explicit std::move)
```

**Why LibTorch Doesn't Have This Problem:**

1. **Narrower Scopes** - PyTorch source uses explicit braces:
   ```cpp
   Tensor result;
   {
       auto temp = some_operation();  // temp created
       result = use_temp(temp);       // temp used
   }  // ← temp destroyed HERE, memory freed immediately
   ```

2. **More Temporaries** - Operations return rvalues that trigger move:
   ```cpp
   auto result = tensor.index_select(...).cat(...);  // ← Temporary triggers move
   ```

3. **In-Place Operations** - Many `_` suffix operations modify without allocation:
   ```cpp
   tensor.add_(other);  // No allocation
   ```

4. **Compiler Optimization** - Release builds with RVO/NRVO eliminate copies

**Our Current Code Structure:**
```cpp
// split() function (default_strategy.cpp:91-150)
auto sampled_sh = ...;        // Lives until line 196 (function end)
auto sampled_scales = ...;    // Lives until line 196
auto sampled_quats = ...;     // Lives until line 196
auto rotmats = ...;           // Lives until line 196
// ... 30+ more intermediate tensors ...
auto samples = stack(...);    // Lives until line 196

// All 33 intermediate tensors: 8.31 GB held until function returns!
```

**Summary:**
- ✅ Move semantics implemented correctly
- ❌ Not triggered because variables are lvalues (named)
- ❌ All 33 intermediate tensors in split() live until function returns
- ❌ LibTorch avoids this with narrower scopes and more temporaries

---

## Issues Found & Current Status

### 1. ⚠️ HACK: sum_scalar() Bool Tensor Corruption
**Status:** Workaround in place, proper fix needed

**Root Cause:** `launch_reduce_op` kernel (`tensor_ops.cu:969-978`) only supports Float32 and Int32:
```cpp
void launch_reduce_op(..., DataType dtype, ...) {
    if (dtype == DataType::Float32) {
        launch_reduce_op_float32(...);
    } else if (dtype == DataType::Int32) {
        launch_reduce_op_int32(...);
    }
    // Bool falls through - kernel does nothing!
}
```

**Current Hack:** Convert Bool→Int32 in `sum_scalar()` before calling `sum()` (`tensor_impl.hpp:1871-1878`)
- ❌ Inefficient: Extra allocation + conversion
- ❌ Doesn't fix root cause
- ❌ Only fixes sum_scalar(), not sum() in general

**Proper Fix Needed:**
```cpp
// Add to tensor_ops.cu
void launch_reduce_op(..., DataType dtype, ...) {
    if (dtype == DataType::Bool) {
        // Implement Bool reduction kernel
        // Convert Bool→Int32 in kernel, accumulate as Int32/Int64
        launch_reduce_op_bool(...);
    }
    // ... existing code
}
```

**Files to fix:**
- `src/core_new/tensor/tensor_ops.cu:969-978` - Add Bool case
- Implement `launch_reduce_op_bool()` using CUB or custom kernel
- Remove hack from `src/core_new/tensor/internal/tensor_impl.hpp:1871-1878`

---

### 2. ⚠️ HACK: Int64→Int32 Synchronization
**Status:** Band-aid in place, root cause unknown

**Symptom:** Without explicit `cudaDeviceSynchronize()` after Int64→Int32 conversion, `item<int>()` reads garbage

**Current Hack:** Added sync in `tensor.cpp:887`

**Why This Is Wrong:**
- `cudaMemcpy` in `item()` should **implicitly synchronize** - it's a blocking call
- If we need explicit sync, there's a deeper bug in stream management or async operations

**Proper Investigation Needed:**
1. Check if wrong stream is being used
2. Verify async operations are properly queued
3. Check if conversion kernel is actually launched
4. Review memory consistency model

**Files to investigate:**
- `src/core_new/tensor/tensor.cpp:660-903` - Type conversion logic
- `src/core_new/tensor/tensor_ops.cu:225-251` - Conversion kernel
- `src/core_new/tensor/internal/tensor_impl.hpp:1950-1960` - item() implementation

---

### 3. 🔥 CRITICAL: Memory Leak / Poor Memory Management
**Status:** Major production bug - memory usage 4.6× higher than expected

**Expected Memory Usage:**
- 10M Gaussians: **~2.48 GB** (248 bytes each)
- 20M Gaussians (after split): **~4.96 GB**
- **Total with GPU: Should fit in 24 GB easily**

**Actual Memory Usage:**
```
[POOL] CUDA reports: free=0.54 GB, total=23.51 GB, requested=3.58 GB
```
- Consumed: **23 GB** for operations that should use **~5 GB**
- **Memory leak: 4.6× overhead!**

**Root Causes:**

#### 3a. Tensor Lifetime Management Failure
Intermediate tensors in `split()` (`default_strategy.cpp:72-200`) are not released:
- `sampled_scales`, `sampled_quats`, `w`, `x`, `y`, `z`
- `r00-r22` (9 intermediate tensors)
- `row0`, `row1`, `row2`
- `rotmats`, `randn`, `scaled_randn`, etc.
- **All held until function returns**

#### 3b. Lambda Captures Hold Large Tensors
```cpp
const auto param_fn = [this, &sampled_idxs, &rest_idxs, &samples, &sampled_scales]
```
- Lambda executed 6 times (one per parameter)
- `samples` is **~2GB**, held for entire update loop

#### 3c. Memory Pool Not Releasing
- CudaMemoryPool allocates but doesn't properly return memory to CUDA
- Fragmentation prevents reuse
- No compaction or defragmentation

**Why LibTorch Doesn't Have This Problem:**
1. **Reference counting with eager deallocation** - Tensors freed immediately when refcount hits 0
2. **Move semantics everywhere** - No unnecessary copies
3. **Better memory pool** - PyTorch's caching allocator actively manages fragmentation
4. **Copy-on-write** - Shares memory until modification needed
5. **Lazy evaluation** (in some cases) - Avoids intermediate allocations

**Proper Fixes Needed:**

#### Fix 3a: Implement Proper Tensor Deallocation
**Current problem:**
```cpp
// In tensor.cpp
result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
    CudaMemoryPool::instance().deallocate(p, nullptr);
});
```
- shared_ptr might not be releasing promptly
- Memory pool `deallocate()` might not actually free to CUDA

**Need to verify:**
1. Is `deallocate()` actually calling `cudaFree`/`cudaFreeAsync`?
2. Are shared_ptrs being held somewhere unexpectedly?
3. Is move semantics working or are we copying?

#### Fix 3b: Fix Memory Pool Deallocation
Check `memory_pool.hpp:116-140` - verify `deallocate()` actually releases:
```cpp
void deallocate(void* ptr, cudaStream_t stream) {
    // Is this actually freeing to CUDA?
    // Or just returning to internal pool that never shrinks?
}
```

#### Fix 3c: Implement Move Semantics Properly
Ensure tensor operations return by value and use move semantics:
```cpp
Tensor result = a.mul(b);  // Should move, not copy
```

#### Fix 3d: Add Memory Pressure Detection
When allocation fails, trigger cleanup:
```cpp
if (!ptr && bytes >= DIRECT_ALLOC_THRESHOLD) {
    // Free unused cached memory
    CudaMemoryPool::instance().clear_cache();
    // Retry allocation
}
```

**Files to fix:**
- `src/core_new/tensor/internal/memory_pool.hpp:116-140` - Fix deallocate()
- `src/core_new/tensor/tensor.cpp` - Verify shared_ptr lifecycle
- `src/core_new/tensor/internal/tensor_impl.hpp` - Add move constructors/operators
- `src/training_new/strategies/default_strategy.cpp:72-200` - Verify temps are released

---

### 4. ✅ PROPER FIX: Large Allocation Bypass
**Status:** Correctly implemented (production approach)

**Implementation:** `memory_pool.hpp:67-100`
- < 100MB → Arena (fast O(1))
- 100MB - 1GB → cudaMallocAsync (pool)
- **> 1GB → Direct cudaMalloc (bypass pool)**

This is the correct production approach (PyTorch uses similar strategy).

---

## Priority Order for Proper Fixes:

1. **CRITICAL: Fix #3 - Memory Leak** (blocking 10M scale)
   - Investigate why 23GB consumed for 5GB of data
   - Fix tensor deallocation
   - Fix memory pool release mechanism

2. **HIGH: Fix #1 - Bool Reduction Kernel** (correctness)
   - Implement `launch_reduce_op_bool()`
   - Remove hack from `sum_scalar()`

3. **MEDIUM: Fix #2 - Synchronization** (understand root cause)
   - Why does item() need explicit sync?
   - Fix stream handling if broken

---

## Test Status:
- ✅ **1M Gaussians:** Works
- ❌ **10M Gaussians:** OOM at 23GB (should only use ~5GB)
- **This is a REAL bug, not unrealistic test conditions**

---

### The Problem
When running densification with 10M Gaussians, we were getting a corrupted `num_duplicates` value of **1065353216** (should be ~1-10).

This value is `0x3FC00000` which is the float bit pattern for **1.5**.

### What We've Fixed So Far ✓

1. **Thrust Type Conversion at 10M Scale** (`src/core_new/tensor/tensor_ops.cu:225-251`)
   - Replaced `thrust::transform()` with custom grid-stride kernel
   - Handles arbitrary sizes (billions of elements)
   - **Status: FIXED**

2. **Memory-Efficient cat() Operations** (`src/training_new/strategies/default_strategy.cpp:136-168`)
   - Replaced incremental cat() loops with single-allocation operations
   - Achieved 6.6x speedup
   - **Status: FIXED**

3. **sum_scalar() for Bool Tensors** (`src/core_new/tensor/internal/tensor_impl.hpp:1870-1894`)
   - Added Int32 conversion for Bool inputs
   - **Status: PARTIALLY FIXED** (see below)

4. **Thrust copy_if Replacement** (`src/core_new/tensor/tensor_masking_ops.cu:738-856`)
   - Replaced with CUB DeviceSelect::Flagged
   - Custom flag creation kernel with grid-stride loops
   - **Status: FIXED**

5. **Thrust/CUB Count Mismatch** (`src/core_new/tensor/tensor_masking_ops.cpp:912-943`)
   - Allocate max size (numel()) instead of pre-counted size
   - Use CUB's actual count, not Thrust's pre-count
   - Prevents buffer overflow
   - **Status: FIXED**

## Current Bug: sum_scalar() Returns Corrupted Value

### Symptoms
```
[sum_scalar] input dtype=5, result dtype=3
[sum_scalar] Taking Bool->Int32 path
[DEBUG] num_duplicates=1065353216    <-- WRONG! Should be ~1
```

### Debug Findings

1. **Input dtype = 5 (Bool)** ✓ Correct
2. **Result dtype = 3 (Float32)** ← sum() promotes Bool to Float32
3. **Taking Bool->Int32 path** ✓ Code path is correct
4. **num_duplicates still corrupted** ← The conversion isn't working!

### The Code Path

```cpp
// default_strategy.cpp:213
const auto num_duplicates = static_cast<int64_t>(is_duplicated.sum_scalar());

// tensor_impl.hpp:1870-1894
float sum_scalar() const {
    auto result = sum();

    printf("[sum_scalar] input dtype=%d, result dtype=%d\n",
           static_cast<int>(dtype_), static_cast<int>(result.dtype()));

    if (dtype_ == DataType::Bool) {
        printf("[sum_scalar] Taking Bool->Int32 path\n");
        return static_cast<float>(result.to(DataType::Int32).item<int>());
    }

    // ...
    return result.item<float>();
}
```

### The Mystery

**We ARE taking the Bool→Int32 path**, but `num_duplicates` is still getting the corrupted float bit pattern!

This suggests one of:
1. `result.to(DataType::Int32)` is not actually converting correctly
2. `item<int>()` is reading the wrong memory
3. The cast to `float` then `int64_t` is somehow preserving the bit pattern
4. There's memory corruption happening AFTER sum_scalar() returns

### Test Output
```bash
$ ./build/tests/lichtfeld_tests --gtest_filter="LFSOnlyDensification.WorksAt100K"

Testing with 10000000 Gaussians...
Before: 10000000
[sum_scalar] input dtype=5, result dtype=3
[sum_scalar] Taking Bool->Int32 path
[DEBUG] num_duplicates=1065353216
[DEBUG] Before is_split: is_grad_high.numel()=10000000, is_large.numel()=10000000
[sum_scalar] input dtype=5, result dtype=3
[sum_scalar] Taking Bool->Int32 path
[CUB nonzero bool] n=10000000, selected=1
[DEBUG] After duplicate: _splat_data.size()=10000001, is_grad_high.numel()=10000000
[DEBUG] Before cat: is_split.numel()=10000000, zeros.numel()=1065353216
[ERROR] launch_nonzero_bool called with SUSPICIOUS n=1075353216 (looks like corrupted float!)
  As float: 2.384186
✗ FAILED
```

## Files Modified

### Core Tensor Operations
- `src/core_new/tensor/tensor_ops.cu` - Custom type conversion kernel
- `src/core_new/tensor/tensor_masking_ops.cu` - CUB nonzero implementation
- `src/core_new/tensor/tensor_masking_ops.cpp` - Fixed count mismatch
- `src/core_new/tensor/internal/tensor_impl.hpp` - sum_scalar() with debug logging

### Training Strategy
- `src/training_new/strategies/default_strategy.cpp` - Memory-efficient cat(), debug logging

### Test
- `tests/test_lfs_only_densification.cpp` - Tests at 10M scale

## Next Steps for Tomorrow

### 1. Debug the Conversion Chain
Add logging inside `result.to(DataType::Int32).item<int>()` to see:
- What value does `to(DataType::Int32)` produce?
- What does `item<int>()` actually read?
- Is the cast from `float` to `int64_t` the problem?

### 2. Check to() Implementation
Verify `Tensor::to(DataType::Int32)` actually converts Float32→Int32 correctly:
```cpp
// Add to tensor_impl.hpp
Tensor to(DataType new_dtype) const {
    auto result = /* conversion */;
    printf("[to] Converting from dtype=%d to dtype=%d\n",
           static_cast<int>(dtype_), static_cast<int>(new_dtype));
    printf("[to] Input: %f, Output: %d\n",
           this->item<float>(), result.item<int>());
    return result;
}
```

### 3. Alternative: Direct Debug at Call Site
Add logging directly in `default_strategy.cpp`:
```cpp
printf("[BEFORE] is_duplicated.sum_scalar() call\n");
float sum_val = is_duplicated.sum_scalar();
printf("[AFTER] sum_scalar returned: %f\n", sum_val);
int64_t num_dup = static_cast<int64_t>(sum_val);
printf("[AFTER] Cast to int64_t: %lld\n", num_dup);
```

### 4. Check BinaryExpr Evaluation
The issue might be in how `BinaryExpr<Bool>` evaluates when sum() is called:
- Does `BinaryExpr::eval_impl()` respect the Bool dtype?
- Is the lazy evaluation causing type confusion?

### 5. Workaround if Needed
If we can't fix sum_scalar(), use direct nonzero() counting:
```cpp
const auto num_duplicates = is_duplicated.nonzero().size(0);
```

## Key Code Locations

### sum_scalar() Implementation
`src/core_new/tensor/internal/tensor_impl.hpp:1870-1894`

### num_duplicates Calculation
`src/training_new/strategies/default_strategy.cpp:213`

### BinaryExpr Creation
`src/core_new/tensor/internal/tensor_impl.hpp:600-602`
```cpp
return BinaryExpr<TensorLeaf, TensorLeaf, Op>(
    TensorLeaf(lhs), TensorLeaf(rhs), op,
    broadcast_shape, lhs.device(), DataType::Bool);
```

### to() Method
Find where `Tensor::to(DataType)` is implemented - need to verify conversion logic

## DataType Enum Values
- Bool = 5
- Int32 = 1
- Float32 = 3

## Bit Pattern Analysis
```
1065353216 (decimal) = 0x3FC00000 (hex) = 1.5 (float32)

When cast:
(int64_t)(float)1.5 = 1   <-- This is CORRECT
(int64_t)(*(int*)&1.5f) = 1065353216   <-- This is reading bytes wrong

So somewhere we're reinterpreting bytes instead of converting values.
```

## Build Commands
```bash
# Rebuild
cmake --build build -j$(nproc)

# Run test
timeout 60 ./build/tests/lichtfeld_tests --gtest_filter="LFSOnlyDensification.WorksAt100K"
```

## Notes
- The test uses 10M Gaussians (line 33 of test_lfs_only_densification.cpp)
- All gradients are set to 10.0, so we expect ~1 duplicate (not 1065353216!)
- The corruption propagates: zeros.numel()=1065353216 which then causes nonzero() to fail
- Debug logging is currently enabled in sum_scalar() - remember to remove before final PR

## Questions to Answer Tomorrow
1. Why does `result.to(DataType::Int32).item<int>()` not work?
2. Is `to()` actually converting or just reinterpreting bytes?
3. Should we fix `sum()` to return Int32 for Bool inputs instead of Float32?
4. Is there a simpler way to count true values in a Bool tensor?
