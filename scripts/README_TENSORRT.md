# TensorRT LPIPS Setup

## Overview

This directory contains a script to convert the LPIPS PyTorch model to TensorRT for faster inference.

**TensorRT LPIPS is the only supported method** - LibTorch LPIPS has been completely removed from the codebase for better performance and reduced dependencies.

## Version Compatibility

**CRITICAL**: The TensorRT engine must be built with the **same TensorRT version** that the C++ code links against.

### Current Setup
- **C++ Build**: Links `/usr/lib/x86_64-linux-gnu/libnvinfer.so` (system TensorRT)
- **Python Script**: Must use system Python 3 with system TensorRT

### Check Versions

**C++ TensorRT version** (used by CMake):
```bash
dpkg -l | grep tensorrt
# Should show: tensorrt 10.12.0.36-1+cuda12.9 (or similar)
```

**Python TensorRT version**:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
# Should match C++ version: 10.12.0.36
```

## Building the Engine

### Single Command: PyTorch → TensorRT
```bash
# Default: supports up to 1024x1024 images
python3 scripts/convert_lpips_to_tensorrt.py

# For larger images (e.g., 1536x1536):
python3 scripts/convert_lpips_to_tensorrt.py --max-image-size 1536

# For different batch size:
python3 scripts/convert_lpips_to_tensorrt.py --max-image-size 1536 --max-batch-size 4
```

This script:
1. Loads `weights/lpips_vgg.pt` (PyTorch TorchScript model)
2. Exports to ONNX (temporary file in `/tmp`)
3. Converts ONNX to TensorRT engine
4. Saves `weights/lpips_vgg.trt` (~30 MB)
5. Cleans up temporary ONNX file

**IMPORTANT**: The script will print the TensorRT version it's using. Verify it matches your system TensorRT!

### Step 3: Verify
When you run LichtFeld-Studio, it will log the TensorRT version:
```
[info] TensorRT library version: 10.12.0.36
```

This is the compile-time version from the headers that CMake links against.

## Troubleshooting

### Error: "engine plan file is not compatible with this version of TensorRT"

**Cause**: Engine was built with different TensorRT version than C++ runtime.

**What happens**: Evaluation will fail - TensorRT is required (no fallback).

**Solution**:
1. Check versions (see above)
2. Rebuild engine with correct Python:
   ```bash
   # Use SYSTEM python3, not conda/micromamba
   python3 scripts/convert_lpips_to_tensorrt.py --max-image-size 1536
   ```

### TensorRT LPIPS fails to load

**Check the logs** during initialization. You'll see:
- `✓ TensorRT LPIPS loaded successfully` - Working correctly
- `✗ Error: Failed to load TensorRT LPIPS: ...` - Failed with error details
- `✗ Error: TensorRT LPIPS engine not found at: ...` - Engine file missing

**If missing**: Run `python3 scripts/convert_lpips_to_tensorrt.py --max-image-size 1536`

### Error: "Parameter check failed ... does not satisfy any optimization profiles"

**Cause**: Input image size exceeds maximum specified during engine build.

**Solution**: Rebuild with larger `--max-image-size`:
```bash
python3 scripts/convert_lpips_to_tensorrt.py --max-image-size 2048
```

**Note**: Larger max sizes require more GPU memory during build:
- 1024x1024: ~4 GB
- 1536x1536: ~9 GB
- 2048x2048: ~16 GB
- 8192x8192: ~243 GB (will fail on most GPUs!)

### Using Conda/Micromamba TensorRT

If you must use conda/micromamba Python, ensure it has the **exact same** TensorRT version:
```bash
# Check system version
dpkg -l | grep tensorrt-dev

# Install matching version in conda
conda install tensorrt==10.12.0 -c nvidia
```

## Engine Specifications

The default engine (`--max-image-size 1024`):
- **Precision**: FP16 (faster, slightly less accurate than FP32)
- **Input range**: [-1, 1] (automatically normalized from [0, 1])
- **Batch size**: 1-4 images
- **Image size**: 64x64 to 1024x1024
- **Optimal size**: 256x256 (used for optimization)
- **File size**: ~30 MB
- **Memory**: ~3.5 GB GPU during inference (max size)

## Performance

Typical speedup vs LibTorch LPIPS:
- **Small images** (256x256): 2-3x faster
- **Large images** (1024x1024): 3-5x faster
- **Accuracy**: <0.05% relative error

## Files

**Script:**
- `convert_lpips_to_tensorrt.py` - Convert PyTorch → ONNX → TensorRT (single command)

**Model Files (in `weights/`):**
- `lpips_vgg.pt` (57 MB) - PyTorch TorchScript model (**only needed for conversion**, can be deleted after generating .trt file)
- `lpips_vgg.trt` (31 MB) - TensorRT engine (**required for runtime**)

**Note**: ONNX is used as a temporary intermediate format during conversion but is not saved.

**After conversion**: You can delete `lpips_vgg.pt` to save 57 MB if you don't plan to rebuild the TensorRT engine.
