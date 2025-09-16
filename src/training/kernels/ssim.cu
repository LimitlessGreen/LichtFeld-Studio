/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "kernels/ssim.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace gs::training {

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
            // ------------------------------------------------------------
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO; // skip left halo

                float sumX = 0.f;
                float sumX2 = 0.f;
                float sumY = 0.f;
                float sumY2 = 0.f;
                float sumXY = 0.f;

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
    // Backward Kernel
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

    // ------------------------------------------
    // Mean reduction kernel with proper parallelization
    // ------------------------------------------
    __global__ void ssim_reduce_mean_kernel(
        const float* ssim_map,
        float* partial_sums,
        int total_elements,
        int start_h,
        int end_h,
        int start_w,
        int end_w,
        int height,
        int width,
        int channels) {

        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        float sum = 0.0f;

        // Grid-stride loop
        for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
            // Decode position for cropping check
            int c = (i / (height * width)) % channels;
            int h = (i / width) % height;
            int w = i % width;

            if (h >= start_h && h < end_h && w >= start_w && w < end_w) {
                sum += ssim_map[i];
            }
        }

        sdata[tid] = sum;
        __syncthreads();

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial_sums[blockIdx.x] = sdata[0];
        }
    }

    // Second stage reduction
    __global__ void reduce_partial_sums(float* partial_sums, float* result, int n_blocks, int valid_elements) {
        float sum = 0.0f;
        for (int i = 0; i < n_blocks; i++) {
            sum += partial_sums[i];
        }
        *result = sum / float(valid_elements);
    }

    // Fill gradient kernel
    __global__ void ssim_fill_gradient_kernel(
        float* dL_dmap,
        float gradient_value,
        int total_elements,
        int start_h,
        int end_h,
        int start_w,
        int end_w,
        int height,
        int width,
        int channels) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            int c = (idx / (height * width)) % channels;
            int h = (idx / width) % height;
            int w = idx % width;

            if (h >= start_h && h < end_h && w >= start_w && w < end_w) {
                dL_dmap[idx] = gradient_value;
            } else {
                dL_dmap[idx] = 0.0f;
            }
        }
    }

    // ------------------------------------------
    // Launch functions
    // ------------------------------------------
    void launch_ssim_forward(
        const float* img1,
        const float* img2,
        float* ssim_map,
        float* dm_dmu1,
        float* dm_dsigma1_sq,
        float* dm_dsigma12,
        int batch,
        int channels,
        int height,
        int width,
        float C1,
        float C2,
        cudaStream_t stream) {

        dim3 grid((width + BLOCK_X - 1) / BLOCK_X,
                  (height + BLOCK_Y - 1) / BLOCK_Y,
                  batch);
        dim3 block(BLOCK_X, BLOCK_Y);

        fusedssimCUDA<<<grid, block, 0, stream>>>(
            height, width, channels, C1, C2,
            img1, img2, ssim_map,
            dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
    }

    void launch_ssim_backward(
        const float* img1,
        const float* img2,
        const float* dL_dmap,
        const float* dm_dmu1,
        const float* dm_dsigma1_sq,
        const float* dm_dsigma12,
        float* dL_dimg1,
        int batch,
        int channels,
        int height,
        int width,
        float C1,
        float C2,
        cudaStream_t stream) {

        dim3 grid((width + BLOCK_X - 1) / BLOCK_X,
                  (height + BLOCK_Y - 1) / BLOCK_Y,
                  batch);
        dim3 block(BLOCK_X, BLOCK_Y);

        fusedssim_backwardCUDA<<<grid, block, 0, stream>>>(
            height, width, channels, C1, C2,
            img1, img2, dL_dmap, dL_dimg1,
            dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
    }

    void launch_ssim_reduce_mean(
        const float* ssim_map,
        float* mean_value,
        int batch,
        int channels,
        int height,
        int width,
        int crop_border,
        cudaStream_t stream) {

        int start_h = crop_border;
        int end_h = height - crop_border;
        int start_w = crop_border;
        int end_w = width - crop_border;

        if (end_h <= start_h || end_w <= start_w) {
            start_h = 0;
            end_h = height;
            start_w = 0;
            end_w = width;
        }

        int valid_elements = batch * channels * (end_h - start_h) * (end_w - start_w);
        int total_elements = batch * channels * height * width;

        // Two-stage reduction
        int block_size = 256;
        int grid_size = std::min(1024, (total_elements + block_size - 1) / block_size);

        // Allocate temp buffer for partial sums
        float* partial_sums;
        cudaMalloc(&partial_sums, grid_size * sizeof(float));

        ssim_reduce_mean_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
            ssim_map, partial_sums, total_elements,
            start_h, end_h, start_w, end_w, height, width, channels);

        // Final reduction
        reduce_partial_sums<<<1, 1, 0, stream>>>(partial_sums, mean_value, grid_size, valid_elements);

        cudaFree(partial_sums);
    }

    void launch_ssim_fill_gradient(
        float* dL_dmap,
        float gradient_value,
        int batch,
        int channels,
        int height,
        int width,
        int crop_border,
        cudaStream_t stream) {

        int start_h = crop_border;
        int end_h = height - crop_border;
        int start_w = crop_border;
        int end_w = width - crop_border;

        if (end_h <= start_h || end_w <= start_w) {
            start_h = 0;
            end_h = height;
            start_w = 0;
            end_w = width;
        }

        int total_elements = batch * channels * height * width;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        ssim_fill_gradient_kernel<<<grid_size, block_size, 0, stream>>>(
            dL_dmap, gradient_value, total_elements,
            start_h, end_h, start_w, end_w, height, width, channels);
    }

} // namespace gs::training