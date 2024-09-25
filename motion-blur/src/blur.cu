#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include <chrono>
#include "blur.h"
#include "dfuncs.cuh"
#include "tools.h"

__constant__ Homography trajectoryBuffer[MAX_HOMOGRAPHY_MATS];

__global__ void motionBlur_kernel(
        uint numMats,
        const uchar3 *image, 
        uint inputWidth, 
        uint inputHeight, 
        uint numPixelLayers,
        float stepsPerOccuBlock, 
        uchar3 *blurred,
        uint viewportWidth,
        uint viewportHeight) {
    /* `numPixelLayers` is number of blocks worth of image data to accumulate at a
     * given time.
     * `stepsPerOccuBlock` is 
     */
    uint threadId = threadIdx.y * blockDim.x + threadIdx.x;  // 0:1024

    extern __shared__ char shbuf[];
    PixelBlock *pixelBuf = reinterpret_cast<PixelBlock *>(shbuf);

    // defines (bx, by) minimum block found to be occupied in the receptive field
    __shared__ int2 gridOffset;
    // gridIndex[gy][gx] with (gy, gx) block index relative to gridOffset
    // In the first phase, this is a 0-1 mask to show which blocks overlap the receptive field.
    // In second phase, empty blocks are assigned -1, and other blocks are
    // consecutively numbered in row-major order from 0 onwards.
    __shared__ GridIndex gridIndex;

    // occupiedBlocks[i] = (gx, gy), index into gridIndex
    __shared__ ushort2 occupiedBlocks[MAX_OCCU_BLOCKS];
    __shared__ int numOccupiedBlocks;

    for (int y=threadIdx.y; y<FIELD_BLOCKS; y+=blockDim.y) {
        for (int x=threadIdx.x; x<FIELD_BLOCKS; x+=blockDim.x) {
            gridIndex[y][x] = 0;
        }
    }
    if (threadId == 0) {
        gridOffset.x = 10000;
        gridOffset.y = 10000;
    }

    __syncthreads();

    /*
       estimate receptive field gridRect in viewport grid using 32 timesteps.
       Also, populate the gridIndex based on the grid extent of each timestep.

       Work is divided into four warps with each warp containing 8 timesteps.  groups
       of four lanes each represent the four corners of the target squre for that
       timestep.

       update shared variables:
       gridRect
       gridIndex
     */

    // 1. Determine grid bounding box (in global viewport grid indices)
    float xbeg = blockIdx.x * blockDim.x;
    float ybeg = blockIdx.y * blockDim.y;
    float xend = xbeg + blockDim.x;
    float yend = ybeg + blockDim.y;
    float2 targetCorners[4] = { { xbeg, ybeg }, { xbeg, yend }, { xend, ybeg }, { xend, yend } };
    float2 lmin, lmax, gmin;

    if (threadId < 128) {
        // use one warp for each target square corner
        // use warpSize timesteps to estimate full extent of receptive field
        const float inc = (float)numMats / (warpSize - 1);
        int cornerIdx = threadId % 4;
        int step = threadId / 4;
        float t = step * inc;
        // assert(t < 1.0);
        float2 source = linTransform(trajectoryBuffer, numMats, t, targetCorners[cornerIdx]);
        // printf("source: %f, %f\n", source.x, source.y);
        warp_min(source, 1, 4, lmin);
        warp_max(source, 1, 4, lmax);
        warp_min(lmin, 4, 16, gmin);
        if (threadId % 32 == 0) {
            atomicMin(&gridOffset.x, to_block(gmin.x));
            atomicMin(&gridOffset.y, to_block(gmin.y));
        }
    }

    __syncthreads();

    if (threadId < 128 && threadId % 4 == 0) {
        // populate for first phase of gridIndex (0-1 occupancy mask)
        // find the bounding grid for a timestep's source quadrilateral.
        // use each of the 32 timesteps in the trajectory.
        int grid_lmin_x = to_block(lmin.x) - gridOffset.x;
        int grid_lmax_x = to_block(lmax.x) - gridOffset.x + 1;
        int grid_lmin_y = to_block(lmin.y) - gridOffset.y;
        int grid_lmax_y = to_block(lmax.y) - gridOffset.y + 1;
        // assert(grid_lmin_x < grid_lmax_x);
        // assert(grid_lmin_y < grid_lmax_y);
        for (int y=grid_lmin_y; y != grid_lmax_y; ++y) {
            for (int x=grid_lmin_x; x != grid_lmax_x; ++x) {
                gridIndex[y][x] = 1;
                // atomicExch(&gridIndex[y][x], 1);
            }
        }
    }
    __syncthreads();

    int thread_data[NUM_CELLS_PER_THREAD];
    uint elemId = threadId * NUM_CELLS_PER_THREAD;
    uint sy = elemId / FIELD_BLOCKS;                    // 0:FIELD_BLOCKS
    uint sx = elemId % FIELD_BLOCKS;                    // 0:FIELD_BLOCKS:4

    for (uint c=0; c!=NUM_CELLS_PER_THREAD; c++) {
        thread_data[c] = gridIndex[sy][sx+c];
    }

    using BlockReduce = cub::BlockReduce<int, BLOCK_DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
          BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage br_storage;
    int maskSum = BlockReduce(br_storage).Sum(thread_data);

    if (threadId == 0) {
        numOccupiedBlocks = maskSum;
        // assert(numOccupiedBlocks > 0);
        // assert(numOccupiedBlocks < MAX_OCCU_BLOCKS); 
    }


    __syncthreads();

    for (int c=0; c!=NUM_CELLS_PER_THREAD; c++) {
        thread_data[c] = gridIndex[sy][sx+c];
    }

    using BlockScan = cub::BlockScan<int, BLOCK_DIM, cub::BLOCK_SCAN_RAKING, BLOCK_DIM>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

    for (int c=0; c!=NUM_CELLS_PER_THREAD; c++) {
        int mask = gridIndex[sy][sx+c];
        gridIndex[sy][sx+c] = mask ? thread_data[c] : -1;
    }

    __syncthreads();
    // only need to call __syncthreads() if temp_storage is reused

    // occupiedBlocks[i] holds index into gridIndex
    for (int c=0; c!=NUM_CELLS_PER_THREAD; c++) {
        int idx = gridIndex[sy][sx+c];
        if (idx >= 0 && idx < numOccupiedBlocks) {
            occupiedBlocks[idx] = make_ushort2(sx+c, sy);
        }
    }

    // finished writing occupiedBlocks
    __syncthreads();

    // occupiedBlocks, gridOffset are now initialized.
    // iterate in phases of loading pixel data, then accumulating it
    // how many timesteps should we use per occupiedBlock?
    int numTimesteps = (int)(numOccupiedBlocks * stepsPerOccuBlock); 

    //if (threadId == 0) {
        //printf("numTimesteps: %d\n", numTimesteps);
    //}

    float inc = (float)numMats / (float)(numTimesteps - 1);
    uint3 outColor = make_uint3(0, 0, 0);
    uint numAccum = 0;
    int begBlock = 0; // [begBlock, endBlock) defines current range
    int endBlock = 0; // 
    float2 sourceLoc; // location of the source in the viewport
                      // Each iteration loads one layer of pixelBuf
    curandState randState;
    curand_init(0, threadId, 0, &randState);

    float2 targetLoc = make_float2(
            blockIdx.x * blockDim.x + threadIdx.x + 0.5,
            blockIdx.y * blockDim.y + threadIdx.y + 0.5);

    while (endBlock < numOccupiedBlocks) {
        int layer = endBlock - begBlock;
        // assert(layer < numPixelLayers);

        int2 source = get_source_coords(gridOffset, occupiedBlocks, endBlock,
            threadIdx.x, threadIdx.y, inputWidth, inputHeight);
        pixelBuf[layer][threadIdx.y][threadIdx.x] = get_source_pixel(source, image, inputWidth);

        endBlock++;
        if ((endBlock - begBlock) < numPixelLayers && endBlock < numOccupiedBlocks) {
            continue;
        }

        __syncthreads();

        // pixelBuf holds blocks [begBlock, endBlock), up to numPixelLayers layers 
        for (int i=0; i != numTimesteps; i++) {
            // add a little jitter to get irregular points along the path
            float t = inc * (i + curand_normal(&randState) * 0.05);
            sourceLoc = linTransform(trajectoryBuffer, numMats, t, targetLoc);

            int block, px, py;
            get_block_coords(gridOffset, gridIndex, sourceLoc, &block, &px, &py);

            if (block < begBlock || block >= endBlock) continue;

            int layer = block - begBlock;

            // assert(layer < numPixelLayers); 

            Pixel thisColor = pixelBuf[layer][py][px];

            outColor.x += thisColor.x;
            outColor.y += thisColor.y;
            outColor.z += thisColor.z;
            numAccum += 1;
        } // finished processing blocks [begBlock, endBlock) 
        begBlock = endBlock;
    } // while (endBlock < numOccupiedBlocks

    uint2 targetPixel = make_uint2(targetLoc.x, targetLoc.y);
    // assert(numAccum > 0);
    if (targetPixel.x < viewportWidth && targetPixel.y < viewportHeight) {
        uint outIdx = targetPixel.y * viewportWidth + targetPixel.x;
        // assert(numAccum > 0);
        blurred[outIdx].x = roundf((float)outColor.x / (float)numAccum);
        blurred[outIdx].y = roundf((float)outColor.y / (float)numAccum);
        blurred[outIdx].z = roundf((float)outColor.z / (float)numAccum);
    }
}

uint ceil_ratio(uint a, uint b) {
    return (a + b - 1) / b;
}

void motionBlur(
        Homography *trajectory,
        uint numMats,
        const uchar3 *h_image,
        uint inputWidth,
        uint inputHeight,
        uint numPixelLayers,
        float stepsPerOccuBlock,
        uchar3 *h_blurred,
        uint viewportWidth,
        uint viewportHeight) {

    size_t inputSize = inputWidth * inputHeight * 3;

    uchar3 *d_image = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_image, inputSize));
    CUDA_CHECK(cudaMemcpy(d_image, h_image, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t outputSize = viewportWidth * viewportHeight * 3; 
    uchar3 *d_blurred = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_blurred, outputSize));

    assert(numMats <= MAX_HOMOGRAPHY_MATS);
    size_t trajectorySize = sizeof(Homography) * numMats;
    CUDA_CHECK(cudaMemcpyToSymbol(trajectoryBuffer, trajectory, trajectorySize)); 

    dim3 dimGrid = dim3(
            ceil_ratio(viewportWidth, BLOCK_DIM), 
            ceil_ratio(viewportHeight, BLOCK_DIM), 1);

    dim3 dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    size_t sharedBytes = numPixelLayers * sizeof(PixelBlock); 

    auto start_time = std::chrono::high_resolution_clock::now();
    motionBlur_kernel<<<dimGrid, dimBlock, sharedBytes>>>(
            numMats, d_image, inputWidth, inputHeight, 
            numPixelLayers, stepsPerOccuBlock, d_blurred,
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

