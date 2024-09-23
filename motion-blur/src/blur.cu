#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include "blur.h"
#include "dfuncs.cuh"
#include "tools.h"

#define MAX_TRAJECTORIES 64 
__constant__ Homography trajectoryBuffer[MAX_TRAJECTORIES];

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

    // 1. Determine grid bounding box (in global viewport grid indices)
    float xbeg = blockIdx.x * blockDim.x + 0.5;
    float ybeg = blockIdx.y * blockDim.y + 0.5;
    float xend = xbeg + blockDim.x - 1.0;
    float yend = ybeg + blockDim.y - 1.0;
    float2 targetCorners[4] = { { xbeg, ybeg }, { xbeg, yend }, { xend, ybeg }, { xend, yend } };

    if (threadIdx.y < FIELD_BLOCKS && threadIdx.x < FIELD_BLOCKS){
        for (int y=threadIdx.y; y<FIELD_BLOCKS; y+=blockDim.y) {
            for (int x=threadIdx.x; x<FIELD_BLOCKS; x+=blockDim.x) {
                gridIndex[y][x] = 0;
            }
        }
    }

    /*
    if (threadId == 0){
        gridOffset = make_int2(10000, 10000); // large initial seed values for minimization
        for (int y=0; y != FIELD_BLOCKS; y++) {
            for (int x=0; x != FIELD_BLOCKS; x++) {
                gridIndex[y][x] = 0;
            }
        }
    }
    */
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
    float2 lmin, lmax, gmin;

    if (threadIdx.y < 4) {
        // use one warp for each target square corner
        // use warpSize timesteps to estimate full extent of receptive field
        const float inc = 0.999 / warpSize;
        int cornerIdx = threadIdx.x % 4;
        int step = (threadIdx.y * blockDim.y + threadIdx.x) / 4;
        float t = step * inc;
        assert(t < 1.0);
        float2 corner = targetCorners[cornerIdx];
        float2 source = linTransform(trajectoryBuffer, numMats, t, corner);
        warp_min(source, 1, 4, lmin);
        warp_max(source, 1, 4, lmax);
        warp_min(lmin, 4, 16, gmin);

        if (threadIdx.x == 0) {
            atomicMin(&gridOffset.x, to_block(gmin.x));
            atomicMin(&gridOffset.y, to_block(gmin.y));
        }
    }
    __syncthreads();

    // TODO: delete
    // if (threadId == 0) {
      //   gridOffset = make_int2(0, 0);
    // }
    __syncthreads();

    if (threadIdx.y < 4 && threadIdx.x % 4 == 0) {
        // populate for first phase of gridIndex (0-1 occupancy mask)
        // find the bounding grid for a timestep's source quadrilateral.
        // use each of the 32 timesteps in the trajectory.
        int grid_lmin_x = to_block(lmin.x) - gridOffset.x;
        int grid_lmax_x = to_block(lmax.x) - gridOffset.x + 1;
        int grid_lmin_y = to_block(lmin.y) - gridOffset.y;
        int grid_lmax_y = to_block(lmax.y) - gridOffset.y + 1;
        assert(grid_lmin_x < grid_lmax_x);
        assert(grid_lmin_y < grid_lmax_y);
        for (int y=grid_lmin_y; y != grid_lmax_y; ++y) {
            for (int x=grid_lmin_x; x != grid_lmax_x; ++x) {
                atomicExch(&gridIndex[y][x], 1);
            }
        }
    }
    
    __syncthreads();

    int thread_data[4];
    uint elemId = threadId * 4;                         // 0:4096:4
    uint sy = elemId / 64;                              // 0:64
    uint sx = elemId % 64;                              // 0:64:4

    for (uint c=0; c!=4; c++) {
        thread_data[c] = gridIndex[sy][sx+c];
    }
    // __syncthreads();

    using BlockReduce = cub::BlockReduce<int, BLOCK_DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
          BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage br_storage;
    int maskSum = BlockReduce(br_storage).Sum(thread_data);

    if (threadId == 0) {
        numOccupiedBlocks = maskSum;
        assert(numOccupiedBlocks > 0);
    }

    // __syncthreads();

    for (int c=0; c!=4; c++) {
        thread_data[c] = gridIndex[sy][sx+c];
    }
    // __syncthreads();

    using BlockScan = cub::BlockScan<int, BLOCK_DIM, cub::BLOCK_SCAN_RAKING, BLOCK_DIM>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

    // __syncthreads();

    for (int c=0; c!=4; c++) {
        int mask = gridIndex[sy][sx+c];
        gridIndex[sy][sx+c] = mask ? thread_data[c] : -1;
    }

    __syncthreads();
    // only need to call __syncthreads() if temp_storage is reused

    // occupiedBlocks[i] holds index into gridIndex
    uint dataIdx = 4 * threadId;
    uint y = dataIdx / 64;
    uint x = dataIdx % 64;
    for (uint xi = 0; xi != 4; xi++) {
        int idx = gridIndex[y][x+xi];
        if (idx >= 0 && idx < numOccupiedBlocks) {
            occupiedBlocks[idx] = make_ushort2(x+xi, y);
        }
    }

    // finished writing occupiedBlocks
    __syncthreads();
    /*
       if (blockIdx.x == 0 && blockIdx.y == 0 && threadNum < numOccupiedBlocks) {
       ushort2 blk = occupiedBlocks[threadNum];
       printf("i: %d, occupiedBlock(adjusted): (%d, %d)\n", 
       threadNum, blk.y + gridOffset.y, blk.x + gridOffset.x);
       }
     */

    // occupiedBlocks, gridOffset are now initialized.
    // iterate in phases of loading pixel data, then accumulating it
    // how many timesteps should we use per occupiedBlock?
    int numTimesteps = (int)(numOccupiedBlocks * stepsPerOccuBlock); 
    float inc = 1.0 / (float)numTimesteps;
    uint3 outColor = make_uint3(0, 0, 0);
    uint numAccum = 0;
    int begBlock = 0; // [begBlock, endBlock) defines current range
    int endBlock = 0; // 
    float2 sourceLoc; // location of the source in the viewport
                      // Each iteration loads one layer of pixelBuf
    curandState randState;
    float2 targetLoc = make_float2(
            blockIdx.x * blockDim.x + threadIdx.x + 0.5,
            blockIdx.y * blockDim.y + threadIdx.y + 0.5);

    /*
       if (blockIdx.x == QUERY_X && blockIdx.y == QUERY_Y && threadIdx.x == 16 && threadIdx.y == 16) {
       for (int b=0; b != numOccupiedBlocks; b++) {
       printf("b: %d, bc: (%d, %d)\n", b, occupiedBlocks[b].x, occupiedBlocks[b].y);
       }
       }
     */
    // bitmask of layers initialized
    // uint layerInitialized = 0;
    // uint blockInitialized = 0;

    /*
    for (int l=0; l != numPixelLayers; l++) {
        pixelBuf[l][threadIdx.y][threadIdx.x] = make_uchar3(255,255,255);
    }
    __syncthreads();
    */

    while (endBlock < numOccupiedBlocks) {
        // __syncthreads();
        int layer = endBlock - begBlock;
        assert(layer < numPixelLayers);

        pixelBuf[layer][threadIdx.y][threadIdx.x] = get_source_pixel(
                gridOffset, occupiedBlocks, 
                endBlock, threadIdx.x, threadIdx.y, image, inputWidth, inputHeight);

        // layerInitialized |= 1u<<layer;
        // blockInitialized |= 1u<<endBlock;
        // __syncthreads();

        endBlock++;
        if ((endBlock - begBlock) < numPixelLayers && endBlock < numOccupiedBlocks) {
            continue;
        }
        /*
           if (blockIdx.x == QUERY_X && blockIdx.y == QUERY_Y && threadIdx.x == 16 && threadIdx.y == 16) {
           printf("bl: [%d, %d), bc: (%d, %d) n: %d\n", begBlock, endBlock,
           blockCoord.x, blockCoord.y, numOccupiedBlocks);
           }
         */
        __syncthreads();

        // pixelBuf holds blocks [begBlock, endBlock), up to numPixelLayers layers 
        for (int i=0; i != numTimesteps; i++) {
            // add a little jitter to get irregular points along the path
            float t = inc * (i + curand_normal(&randState) * 0.05);
            t = max(0.0, min(t, 0.9999));
            assert(t < 1.0);
            sourceLoc = linTransform(trajectoryBuffer, numMats, t, targetLoc);

            int block, px, py;
            get_block_coords(gridOffset, gridIndex, sourceLoc, &block, &px, &py);

            if (block < begBlock || block >= endBlock) continue;

            int layer = block - begBlock;

            assert(layer < numPixelLayers); 

            /*
            if (blockIdx.x == QUERY_X && blockIdx.y == QUERY_Y && threadIdx.x == 16 && 
                    ! (layerInitialized & 1<<layer)) {
                printf("layer %d uninitialized\n", layer);
            }
            */
            // assert(layerInitialized & 1<<layer);
            // assert(blockInitialized & 1<<block);
            // assert(gridOffset.x == 0 && gridOffset.y == 0);
            // if (! layerInitialized & 1<<layer) continue;

            uchar3 thisColor = pixelBuf[layer][py][px];
            // uchar3 thisColor = get_source_pixel(gridOffset, occupiedBlocks, block, px, py, 
              //       image, inputWidth, inputHeight); 
            // uchar3 thisColor = get_source_pixel_dumb(gridOffset, gridIndex,
              //       occupiedBlocks, block, px, py, image, inputWidth, inputHeight); 

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
        assert(numAccum > 0);
        blurred[outIdx].x = roundf((float)outColor.x / (float)numAccum);
        blurred[outIdx].y = roundf((float)outColor.y / (float)numAccum);
        blurred[outIdx].z = roundf((float)outColor.z / (float)numAccum);
    }
    /*
    // TODO: remove
    pixelBuf[0][threadIdx.y][threadIdx.x] = make_uchar3(255,0,0);
    pixelBuf[1][threadIdx.y][threadIdx.x] = make_uchar3(0,255,0);
    pixelBuf[2][threadIdx.y][threadIdx.x] = make_uchar3(0,0,255);

    if (threadId < MAX_OCCU_BLOCKS) {
        occupiedBlocks[threadId] = make_ushort2(0, 0);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        numOccupiedBlocks = 0; 
    }
    __syncthreads();
    */
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

    assert(numMats <= MAX_TRAJECTORIES);
    size_t trajectorySize = sizeof(Homography) * numMats;
    CUDA_CHECK(cudaMemcpyToSymbol(trajectoryBuffer, trajectory, trajectorySize)); 

    dim3 dimGrid = dim3(
            ceil_ratio(viewportWidth, BLOCK_DIM), 
            ceil_ratio(viewportHeight, BLOCK_DIM), 1);

    dim3 dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    // size_t sharedBytes = numPixelLayers * BLOCK_DIM * BLOCK_DIM * 3; 
    size_t sharedBytes = numPixelLayers * sizeof(PixelBlock); 
    motionBlur_kernel<<<dimGrid, dimBlock, sharedBytes>>>(
            numMats, d_image, inputWidth, inputHeight, 
            numPixelLayers, stepsPerOccuBlock, d_blurred,
            viewportWidth, viewportHeight);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_blurred, d_blurred, outputSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_blurred));
}



