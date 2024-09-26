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
        uint viewportHeight) {
    /* `numPixelLayers` is number of blocks worth of image data to accumulate at a
     * given time.
     * `steps_per_occu_block` is 
     */
    uint thread_id = threadIdx.y * blockDim.x + threadIdx.x;  // 0:1024

    extern __shared__ uchar4 pixel_buf[]; // box of pixels in row-major order
    __shared__ int2 box_start, box_end; // viewport coordinate of top left pixel or
                                        // (exclusive) bottom right pixel in
                                        // `pixel_buf`

    // each block of threads computes a 64 x 64 pixel patch, starting at (gx, gy)
    float gx = (float)(blockIdx.x * blockDim.x);
    float gy = (float)(blockIdx.y * blockDim.y * 4);

    // 1. Determine grid bounding box (in global viewport grid indices)
    float2 targetCorners[4] = {{gx, gy}, {gx, gy + 64}, {gx + 64, gy}, {gx + 64, gy + 64}};
    int min_x, min_y, max_x, max_y;

    if (thread_id < 128) {
        // use one warp for each target square corner
        // use warpSize timesteps to estimate full extent of receptive field
        const float inc = (float)num_mats / (warpSize - 1);
        int cornerIdx = thread_id % 4;
        int step = thread_id / 4;
        float t = step * inc;
        float2 source = lin_transform(trajectory_buf, num_mats, t, targetCorners[cornerIdx]);
        min_x = max_x = (int)source.x;
        min_y = max_y = (int)source.y;
    } else {
        min_x = min_y = INT_MAX;
        max_x = max_y = INT_MIN;
    }

    using BlockReduce = cub::BlockReduce<int, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
          BLOCK_DIM_Y>;
    __shared__ typename BlockReduce::TempStorage br_storage;
    min_y = BlockReduce(br_storage).Reduce(min_y, cub::Min());
    __syncthreads();
    min_x = BlockReduce(br_storage).Reduce(min_x, cub::Min());
    __syncthreads();
    max_y = BlockReduce(br_storage).Reduce(max_y, cub::Max());
    __syncthreads();
    max_x = BlockReduce(br_storage).Reduce(max_x, cub::Max());

    if (thread_id == 0) {
        box_start.x = min_x;
        box_start.y = min_y;
        box_end.x = max_x;
        box_end.y = max_y;
    }
    __syncthreads();

    /*
    if (thread_id == 0) {
        printf("box_start: %d, %d, box_end: %d, %d\n", box_start.x, box_start.y,
                box_end.x, box_end.y);
    }
    */

    // curandState randState;
    // curand_init(0, thread_id, 0, &randState);

    // load the data
    const int box_width = box_end.x - box_start.x; 
    const int box_size = box_width * (box_end.y - box_start.y);
    const int buf_size = pixel_buf_bytes / sizeof(uchar4);
    const int num_timesteps = (int)((unsigned int)(box_size * steps_per_occu_block))>>10;
    float inc = (float)num_mats / (float)(num_timesteps - 1);

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
    /*
    if (thread_id == 0) {
        printf("buf_size: %d, pixel_buf_bytes: %d, num_timesteps: %d, inc: %f\n", 
                buf_size, pixel_buf_bytes, num_timesteps, inc);
    }
    */

    float2 source;
    float2 target = make_float2(gx + threadIdx.x + 0.5, gy + threadIdx.y + 0.5);

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

        for (int ty = 0; ty != 4; ty++) {
            for (int i = 0; i != num_timesteps; ++i) {
                // t = inc * (i + curand_normal(&randState) * 0.05);
                t = inc * i;
                source = lin_transform(trajectory_buf, num_mats, t, target);
                // assert(source.x >= box_start.x);
                // assert(source.y >= box_start.y);
                // assert(source.x < box_end.x);
                // assert(source.y < box_end.y);
                // NOTE: This original formula caused a scrambling of 32 byte blocks
                // int voff_src = (source.y - box_start.y) * box_width + (source.x - box_start.x);
                int voff_src = ((int)source.y - box_start.y) * box_width + ((int)source.x - box_start.x);
                loff = voff_src - voff;
                /*
                if (thread_id == 0) {
                    printf("voff_src: %d\n", voff_src);
                }
                */
                pixel4 = (loff >= 0 && loff < buf_size) ? pixel_buf[loff] : zero;
                out_pixel[ty].x += pixel4.x;
                out_pixel[ty].y += pixel4.y;
                out_pixel[ty].z += pixel4.z;
                num_accum[ty] += (loff >= 0 && loff < buf_size) ? 1 : 0;
            }
            target.y += BLOCK_DIM_Y;
        }
        voff += buf_size;
    }

    uint2 target_pixel = make_uint2(gx + threadIdx.x, gy + threadIdx.y);
    for (int ty = 0; ty != 4; ty++) {
        if (target_pixel.x < viewportWidth && target_pixel.y < viewportHeight) {
            idx = target_pixel.y * viewportWidth + target_pixel.x;
            blurred[idx].x = roundf((float)out_pixel[ty].x / (float)num_accum[ty]);
            blurred[idx].y = roundf((float)out_pixel[ty].y / (float)num_accum[ty]);
            blurred[idx].z = roundf((float)out_pixel[ty].z / (float)num_accum[ty]);
        }
        target_pixel.y += BLOCK_DIM_Y;
    }
}

uint ceil_ratio(uint a, uint b) {
    return (a + b - 1) / b;
}

void motionBlur(
        Homography *trajectory,
        uint num_mats,
        const uchar3 *h_image,
        uint input_width,
        uint input_height,
        int pixel_buf_bytes,
        float steps_per_occu_block,
        uchar3 *h_blurred,
        uint viewportWidth,
        uint viewportHeight) {

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
            viewportWidth, viewportHeight);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    printf("Rendering time (seconds): %f\n", elapsed.count());

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_blurred, d_blurred, outputSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_blurred));
}

