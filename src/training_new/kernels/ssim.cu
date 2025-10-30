/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/ssim.cuh"
#include "lfs/kernels/ssim_reduction.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace {
    // ------------------------------------------
    // Constant Memory for Gaussian Coefficients
    // ------------------------------------------
    __constant__ float cGauss[11] = {
        0.001028380123898387f,
        0.0075987582094967365f,
        0.036000773310661316f,
        0.10936068743467331f,
        0.21300552785396576f,
        0.26601171493530273f,
        0.21300552785396576f,
        0.10936068743467331f,
        0.036000773310661316f,
        0.0075987582094967365f,
        0.001028380123898387f};

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

    // ------------------------------------------
    // Utility: Safe pixel fetch w/ zero padding
    // ------------------------------------------
    __device__ __forceinline__ float get_pix_value(
        const float* img,
        int b, int c, int y, int x,
        int CH, int H, int W) {
        if (x < 0 || x >= W || y < 0 || y >= H) {
            return 0.0f;
        }
        return img[b * CH * H * W + c * H * W + y * W + x];
    }

    // ------------------------------------------
    // Forward Kernel: Fused SSIM
    //  - Two-pass convolution to get mu1, mu2,
    //    sigma1_sq, sigma2_sq, sigma12, etc.
    //  - Writes final SSIM map to ssim_map
    //  - Optionally writes partial derivatives
    //    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    // ------------------------------------------
    __global__ void fusedssimCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        float* __restrict__ ssim_map,
        float* __restrict__ dm_dmu1,
        float* __restrict__ dm_dsigma1_sq,
        float* __restrict__ dm_dsigma12) {
        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z; // batch index
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        // Shared memory for the tile (img1, img2)
        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        // After horizontal pass, store partial sums here
        // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
        __shared__ float xconv[CONV_Y][CONV_X][5];

        // Each block processes B x C sub-batches. We loop over channels:
        for (int c = 0; c < CH; ++c) {
            // ------------------------------------------------------------
            // 1) Load (img1, img2) tile + halo into shared memory
            // ------------------------------------------------------------
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;

                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                        float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                        sTile[local_y][local_x][0] = X;
                        sTile[local_y][local_x][1] = Y;
                    }
                }
            }
            block.sync();

            // ------------------------------------------------------------
            // 2) Horizontal convolution (11x1) in shared memory
            //    We'll accumulate symmetrical pairs around center.
            // ------------------------------------------------------------
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO; // skip left halo

                float sumX = 0.f;
                float sumX2 = 0.f;
                float sumY = 0.f;
                float sumY2 = 0.f;
                float sumXY = 0.f;

                // #pragma unroll for those 5 pairs
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft = sTile[ly][lx - d][0];
                    float Yleft = sTile[ly][lx - d][1];
                    float Xright = sTile[ly][lx + d][0];
                    float Yright = sTile[ly][lx + d][1];

                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float centerX = sTile[ly][lx][0];
                    float centerY = sTile[ly][lx][1];
                    float wc = cGauss[HALO];
                    sumX += centerX * wc;
                    sumX2 += (centerX * centerX) * wc;
                    sumY += centerY * wc;
                    sumY2 += (centerY * centerY) * wc;
                    sumXY += (centerX * centerY) * wc;
                }

                // Write out partial sums
                xconv[ly][threadIdx.x][0] = sumX;
                xconv[ly][threadIdx.x][1] = sumX2;
                xconv[ly][threadIdx.x][2] = sumY;
                xconv[ly][threadIdx.x][3] = sumY2;
                xconv[ly][threadIdx.x][4] = sumXY;

                // Possibly handle second row in same warp
                int ly2 = ly + BLOCK_Y;
                if (ly2 < CONV_Y) {
                    sumX = 0.f;
                    sumX2 = 0.f;
                    sumY = 0.f;
                    sumY2 = 0.f;
                    sumXY = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float Xleft = sTile[ly2][lx - d][0];
                        float Yleft = sTile[ly2][lx - d][1];
                        float Xright = sTile[ly2][lx + d][0];
                        float Yright = sTile[ly2][lx + d][1];

                        sumX += (Xleft + Xright) * w;
                        sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                        sumY += (Yleft + Yright) * w;
                        sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                        sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                    }
                    // center
                    {
                        float cx = sTile[ly2][lx][0];
                        float cy = sTile[ly2][lx][1];
                        float wc = cGauss[HALO];
                        sumX += cx * wc;
                        sumX2 += (cx * cx) * wc;
                        sumY += cy * wc;
                        sumY2 += (cy * cy) * wc;
                        sumXY += (cx * cy) * wc;
                    }
                    xconv[ly2][threadIdx.x][0] = sumX;
                    xconv[ly2][threadIdx.x][1] = sumX2;
                    xconv[ly2][threadIdx.x][2] = sumY;
                    xconv[ly2][threadIdx.x][3] = sumY2;
                    xconv[ly2][threadIdx.x][4] = sumXY;
                }
            }
            block.sync();

            // ------------------------------------------------------------
            // 3) Vertical convolution (1x11) + final SSIM
            // ------------------------------------------------------------
            {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = xconv[ly - d][lx];
                    float* bot = xconv[ly + d][lx];

                    out0 += (top[0] + bot[0]) * w;
                    out1 += (top[1] + bot[1]) * w;
                    out2 += (top[2] + bot[2]) * w;
                    out3 += (top[3] + bot[3]) * w;
                    out4 += (top[4] + bot[4]) * w;
                }
                // center
                {
                    float wC = cGauss[HALO];
                    float* ctr = xconv[ly][lx];
                    out0 += ctr[0] * wC;
                    out1 += ctr[1] * wC;
                    out2 += ctr[2] * wC;
                    out3 += ctr[3] * wC;
                    out4 += ctr[4] * wC;
                }

                if (pix_x < W && pix_y < H) {
                    float mu1 = out0;
                    float mu2 = out2;
                    float mu1_sq = mu1 * mu1;
                    float mu2_sq = mu2 * mu2;

                    float sigma1_sq = out1 - mu1_sq;
                    float sigma2_sq = out3 - mu2_sq;
                    float sigma12 = out4 - mu1 * mu2;

                    float A = mu1_sq + mu2_sq + C1;
                    float B = sigma1_sq + sigma2_sq + C2;
                    float C_ = 2.f * mu1 * mu2 + C1;
                    float D_ = 2.f * sigma12 + C2;

                    float val = (C_ * D_) / (A * B);

                    int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                    ssim_map[global_idx] = val;

                    if (dm_dmu1) {
                        // partial derivatives
                        float d_m_dmu1 = ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) - (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
                        float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                        float d_m_dsigma12 = (2.f * C_) / (A * B);

                        dm_dmu1[global_idx] = d_m_dmu1;
                        dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                        dm_dsigma12[global_idx] = d_m_dsigma12;
                    }
                }
            }
        }
    }

    // ------------------------------------------
    // Backward Kernel: Apply chain rule to get
    //    dL/d(img1) from partial derivatives
    //    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
    //    and dL/dmap (the gradient from above).
    // ------------------------------------------
    __global__ void fusedssim_backwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        const float* __restrict__ dL_dmap,
        float* __restrict__ dL_dimg1,
        const float* __restrict__ dm_dmu1,
        const float* __restrict__ dm_dsigma1_sq,
        const float* __restrict__ dm_dsigma12) {
        auto block = cg::this_thread_block();

        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;
        const int bIdx = block.group_index().z;

        // Shared memory for the fused data:
        // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
        __shared__ float sData[3][SHARED_Y][SHARED_X];
        __shared__ float sScratch[CONV_Y][CONV_X][3];

        for (int c = 0; c < CH; ++c) {
            float p1 = 0.f, p2 = 0.f;
            if (pix_x < W && pix_y < H) {
                p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
                p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
            }

            // (1) Load + fuse multiplication
            {
                const int start_y = block.group_index().y * BLOCK_Y;
                const int start_x = block.group_index().x * BLOCK_X;

                int tid = threadIdx.y * blockDim.x + threadIdx.x;
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int totalThreads = BLOCK_X * BLOCK_Y;
                int num_warps = (totalThreads + 31) / 32;

                for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    for (int col = lane_id; col < SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;

                        float chain = get_pix_value(dL_dmap, bIdx, c, gy, gx, CH, H, W);
                        float vmu = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        float vs1 = get_pix_value(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                        float vs12 = get_pix_value(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);

                        sData[0][row][col] = vmu * chain;
                        sData[1][row][col] = vs1 * chain;
                        sData[2][row][col] = vs12 * chain;
                    }
                }
            }
            block.sync();

            // (2) Horizontal pass
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                for (int pass = 0; pass < 2; ++pass) {
                    int yy = ly + pass * BLOCK_Y;
                    if (yy < CONV_Y) {
                        float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                        for (int d = 1; d <= HALO; ++d) {
                            float w = cGauss[HALO - d];
                            float left0 = sData[0][yy][lx - d];
                            float left1 = sData[1][yy][lx - d];
                            float left2 = sData[2][yy][lx - d];

                            float right0 = sData[0][yy][lx + d];
                            float right1 = sData[1][yy][lx + d];
                            float right2 = sData[2][yy][lx + d];

                            accum0 += (left0 + right0) * w;
                            accum1 += (left1 + right1) * w;
                            accum2 += (left2 + right2) * w;
                        }
                        // center
                        {
                            float wc = cGauss[HALO];
                            float c0 = sData[0][yy][lx];
                            float c1 = sData[1][yy][lx];
                            float c2 = sData[2][yy][lx];
                            accum0 += c0 * wc;
                            accum1 += c1 * wc;
                            accum2 += c2 * wc;
                        }

                        sScratch[yy][threadIdx.x][0] = accum0;
                        sScratch[yy][threadIdx.x][1] = accum1;
                        sScratch[yy][threadIdx.x][2] = accum2;
                    }
                }
            }
            block.sync();

            // (3) Vertical pass -> finalize dL/d(img1)
            if (pix_x < W && pix_y < H) {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = sScratch[ly - d][lx];
                    float* bot = sScratch[ly + d][lx];

                    sum0 += (top[0] + bot[0]) * w;
                    sum1 += (top[1] + bot[1]) * w;
                    sum2 += (top[2] + bot[2]) * w;
                }
                // center
                {
                    float wc = cGauss[HALO];
                    float* ctr = sScratch[ly][lx];
                    sum0 += ctr[0] * wc;
                    sum1 += ctr[1] * wc;
                    sum2 += ctr[2] * wc;
                }

                // final accumulation
                float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2)*sum2;

                int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                dL_dimg1[out_idx] = dL_dpix;
            }
            block.sync();
        }
    }

} // anonymous namespace

// LibTorch-Free API
namespace lfs::training::kernels {

    std::pair<float, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        bool apply_valid_padding) {

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Make tensors contiguous and ensure 4D [N, C, H, W]
        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();

        if (img1.ndim() == 3) {
            img1 = img1.unsqueeze(0);
        }
        if (img2.ndim() == 3) {
            img2 = img2.unsqueeze(0);
        }

        int N = static_cast<int>(img1.shape()[0]);
        int C = static_cast<int>(img1.shape()[1]);
        int H = static_cast<int>(img1.shape()[2]);
        int W = static_cast<int>(img1.shape()[3]);

        // Launch config
        dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
                  (H + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        // Output SSIM map
        auto ssim_map = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);

        // Allocate derivative Tensors
        auto dm_dmu1 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dsigma1_sq = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dsigma12 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);

        fusedssimCUDA<<<grid, block>>>(
            H, W, C, C1, C2,
            img1.ptr<float>(),
            img2.ptr<float>(),
            ssim_map.ptr<float>(),
            dm_dmu1.ptr<float>(),
            dm_dsigma1_sq.ptr<float>(),
            dm_dsigma12.ptr<float>());

        // Store original dimensions
        int h = H;
        int w = W;

        // Apply valid padding (crop 5 pixels from each side) using efficient view slicing
        // Then compute mean using optimized tensor reduction (matches PyTorch speed!)
        lfs::core::Tensor ssim_map_cropped = ssim_map;
        if (apply_valid_padding && H > 10 && W > 10) {
            ssim_map_cropped = ssim_map.slice(2, 5, H - 5).slice(3, 5, W - 5);
        }

        // Use tensor library's optimized mean (warp reductions + vectorized loads)
        float ssim_value = ssim_map_cropped.mean().item<float>();

        // Save context for backward
        SSIMContext ctx;
        ctx.img1 = img1;
        ctx.img2 = img2;
        ctx.dm_dmu1 = dm_dmu1;
        ctx.dm_dsigma1_sq = dm_dsigma1_sq;
        ctx.dm_dsigma12 = dm_dsigma12;
        ctx.original_h = h;
        ctx.original_w = w;
        ctx.apply_valid_padding = apply_valid_padding;

        return std::make_pair(ssim_value, std::move(ctx));
    }

    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        float grad_loss) {

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Compute gradient map size (after cropping if applicable)
        int grad_h = ctx.original_h;
        int grad_w = ctx.original_w;
        size_t N = ctx.img1.shape()[0];
        size_t C = ctx.img1.shape()[1];
        size_t numel = N * C * grad_h * grad_w;

        if (ctx.apply_valid_padding && grad_h > 10 && grad_w > 10) {
            grad_h -= 10; // Remove 5 pixels from each side
            grad_w -= 10;
            numel = N * C * grad_h * grad_w;
        }

        // Create gradient map: d(loss)/d(ssim_scalar) = grad_loss
        // d(ssim_scalar)/d(ssim_map[i]) = 1/numel
        // So: d(loss)/d(ssim_map[i]) = grad_loss / numel
        float grad_per_pixel = grad_loss / static_cast<float>(numel);

        // Create gradient tensor for cropped region
        auto dL_dmap = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);

        if (ctx.apply_valid_padding && ctx.original_h > 10 && ctx.original_w > 10) {
            // Fill cropped region with gradient
            auto cropped_view = dL_dmap.slice(2, 5, ctx.original_h - 5).slice(3, 5, ctx.original_w - 5);
            cropped_view.fill_(grad_per_pixel);
        } else {
            // No cropping - fill entire map
            dL_dmap.fill_(grad_per_pixel);
        }

        // Allocate output gradient
        auto dL_dimg1 = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);

        // Launch backward kernel
        dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                  (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        fusedssim_backwardCUDA<<<grid, block>>>(
            ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
            ctx.img1.ptr<float>(),
            ctx.img2.ptr<float>(),
            dL_dmap.ptr<float>(),
            dL_dimg1.ptr<float>(),
            ctx.dm_dmu1.ptr<float>(),
            ctx.dm_dsigma1_sq.ptr<float>(),
            ctx.dm_dsigma12.ptr<float>());

        return dL_dimg1;
    }

} // namespace lfs::training::kernels
