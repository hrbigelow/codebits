#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include <chrono>
#include <climits>
#include "blur.h"
#include "dfuncs.cuh"
#include "tools.h"

__constant__ Homography trajectory_buf[MAX_HOMOGRAPHY_MATS];

__global__ void motionBlur_kernel(
        uint num_mats,
        const uchar3 *image, 
        uint input_width, 
        uint input_height, 
        int pixel_buf_bytes,
        float steps_per_occu_block, 
        uchar3 *blurred,
        uint viewportWidth,
        uint viewportHeight,
        float exposure_mul) {
    /* 
     * `steps_per_occu_block` specifies number of samples per 32 x 32 are of
     * receptive field
     */
    uint thread_id = threadIdx.y * blockDim.x + threadIdx.x;  // 0:1024

    extern __shared__ uchar4 pixel_buf[]; // box of pixels in row-major order
    __shared__ int2 box_start, box_end; // viewport coordinate of top left pixel or
                                        // (exclusive) bottom right pixel in
                                        // `pixel_buf`

    // each block of threads computes a OUT_DIM_X x OUT_DIM_Y pixel patch, starting at (gx, gy)
    float gx = (float)(blockIdx.x * blockDim.x);
    float gy = (float)(blockIdx.y * blockDim.y * 4);

    // 1. Determine grid bounding box (in global viewport grid indices)
    float2 targetCorners[4] = {
        {gx, gy}, 
        {gx, gy + OUT_DIM_Y}, 
        {gx + OUT_DIM_X, gy}, 
        {gx + OUT_DIM_X, gy + OUT_DIM_Y}};
    float min_x, min_y, max_x, max_y;
    float increment = (float)num_mats / (warpSize - 1);

    if (thread_id < 128) {
        // use one warp for each target square corner
        // use warpSize timesteps to estimate full extent of receptive field
        int cornerIdx = thread_id % 4;
        int step = thread_id / 4;
        float t = step * increment;
        float2 source = lin_transform(trajectory_buf, num_mats, t, targetCorners[cornerIdx]);
        min_x = max_x = source.x;
        min_y = max_y = source.y;
    } else {
        min_x = min_y = FLT_MAX;
        max_x = max_y = -FLT_MAX;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
          BLOCK_DIM_Y>;
    __shared__ typename BlockReduce::TempStorage br_storage;
    min_y = BlockReduce(br_storage).Reduce(min_y, cub::Min());
    __syncthreads();
    min_x = BlockReduce(br_storage).Reduce(min_x, cub::Min());
    __syncthreads();
    max_y = BlockReduce(br_storage).Reduce(max_y, cub::Max());
    __syncthreads();
    max_x = BlockReduce(br_storage).Reduce(max_x, cub::Max());
    __syncthreads();

    if (thread_id == 0) {
        box_start.x = (int)floorf(min_x);
        box_start.y = (int)floorf(min_y);
        box_end.x = (int)ceilf(max_x);
        box_end.y = (int)ceilf(max_y);
    }
    __syncthreads();

    // curandState randState;
    // curand_init(0, thread_id, 0, &randState);

    // load the data
    const int box_width = box_end.x - box_start.x; 
    const int box_size = box_width * (box_end.y - box_start.y);
    const int buf_size = pixel_buf_bytes / sizeof(uchar4);
    const int num_timesteps = (int)((unsigned int)ceilf(box_size * steps_per_occu_block))>>10;

    int voff = 0; // offset into virtual buffer [0, box_size)
    int loff; // offset into pixel_buf [0, buf_size)
    int x, y;
    float t;
    unsigned int idx;
    const uchar4 zero = make_uchar4(0, 0, 0, 0);
    int num_accum[4] = {0, 0, 0, 0};
    int4 out_pixel[4] = { 
        make_int4(0, 0, 0, 0), 
        make_int4(0, 0, 0, 0), 
        make_int4(0, 0, 0, 0), 
        make_int4(0, 0, 0, 0), 
    };
    uchar4 pixel4; 

    increment = (float)num_mats / (float)(num_timesteps - 1);
    float2 source, target;

    while (voff < box_size) {
        loff = thread_id;
        while (loff < buf_size) {
            x = box_start.x + (voff + loff) % box_width;
            y = box_start.y + (voff + loff) / box_width; 
            idx = wrap_mod(y, input_height) * input_width + wrap_mod(x, input_width);
            uchar3 pixel3 = image[idx];
            pixel_buf[loff] = make_uchar4(pixel3.x, pixel3.y, pixel3.z, 0);
            loff += BLOCK_SIZE;
        }
        __syncthreads();

        target = make_float2(gx + (float)threadIdx.x + 0.5, gy + (float)threadIdx.y + 0.5);
        for (int ty = 0; ty != 4; ty++) {
            t = 0.0;
            for (int i = 0; i != num_timesteps; ++i) {
                // t = increment * (i + curand_normal(&randState) * 0.05);
                source = lin_transform(trajectory_buf, num_mats, t, target);
                t += increment;
                int voff_src = (
                        ((int)floorf(source.y) - box_start.y) * box_width +
                        ((int)floorf(source.x) - box_start.x));

                loff = voff_src - voff;
                pixel4 = (loff >= 0 && loff < buf_size) ? pixel_buf[loff] : zero;
                out_pixel[ty].x += pixel4.x;
                out_pixel[ty].y += pixel4.y;
                out_pixel[ty].z += pixel4.z;
                num_accum[ty] += (loff >= 0 && loff < buf_size) ? 1 : 0;
            }
            target.y += (float)BLOCK_DIM_Y;
        }
        __syncthreads();
        voff += buf_size;
    }

    uint2 target_pixel = make_uint2((int)gx + threadIdx.x, (int)gy + threadIdx.y);
    for (int ty = 0; ty != 4; ty++) {
        if (target_pixel.x < viewportWidth && target_pixel.y < viewportHeight) {
            idx = target_pixel.y * viewportWidth + target_pixel.x;
            float mul = exposure_mul / (float)num_accum[ty];
            blurred[idx].x = min(roundf((float)out_pixel[ty].x * mul), 255.0);
            blurred[idx].y = min(roundf((float)out_pixel[ty].y * mul), 255.0);
            blurred[idx].z = min(roundf((float)out_pixel[ty].z * mul), 255.0);
        }
        target_pixel.y += BLOCK_DIM_Y;
    }
}

uint ceil_ratio(uint a, uint b) {
    return (a + b - 1) / b;
}

double motionBlur(
        Homography *trajectory,
        uint num_mats,
        const uchar3 *h_image,
        uint input_width,
        uint input_height,
        int pixel_buf_bytes,
        float steps_per_occu_block,
        uchar3 *h_blurred,
        uint viewportWidth,
        uint viewportHeight,
        float exposure_mul) {

    size_t inputSize = input_width * input_height * 3;

    uchar3 *d_image = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_image, inputSize));
    CUDA_CHECK(cudaMemcpy(d_image, h_image, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t outputSize = viewportWidth * viewportHeight * 3; 
    uchar3 *d_blurred = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_blurred, outputSize));

    assert(num_mats <= MAX_HOMOGRAPHY_MATS);
    size_t trajectorySize = sizeof(Homography) * num_mats;
    CUDA_CHECK(cudaMemcpyToSymbol(trajectory_buf, trajectory, trajectorySize)); 

    dim3 dimGrid = dim3(
            ceil_ratio(viewportWidth, BLOCK_DIM_X), 
            ceil_ratio(viewportHeight, BLOCK_DIM_Y * 4), 1);

    dim3 dimBlock = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    auto start_time = std::chrono::high_resolution_clock::now();
    motionBlur_kernel<<<dimGrid, dimBlock, pixel_buf_bytes>>>(
            num_mats, d_image, input_width, input_height, 
            pixel_buf_bytes, steps_per_occu_block, d_blurred,
            viewportWidth, viewportHeight, exposure_mul);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = end_time - start_time;
    std::cerr << "Rendering time (seconds): " << elapsed.count() << std::endl;

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_blurred, d_blurred, outputSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_blurred));
    return elapsed.count();
}

