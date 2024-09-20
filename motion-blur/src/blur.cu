#include <stdio.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include "blur.h"
#include "tools.h"

#define BLOCK_DIM 32
// maximum square side in pixels
#define SOURCE_FIELD_SIZE 2048 
// #define SOURCE_FIELD_SIZE 1024
#define FIELD_BLOCKS (unsigned int)(SOURCE_FIELD_SIZE / BLOCK_DIM)

// maximum number of occupied blocks
#define MAX_OCCU_BLOCKS 512

#define MAX_TRAJECTORIES 64 
__constant__ Homography trajectoryBuffer[MAX_TRAJECTORIES];

__device__ float2 transform(const Homography *mats, unsigned int t, float2 coord)
{
  float x = coord.x;
  float y = coord.y;
  const Homography &h = mats[t];
  // printf("t: %d, x: %f, y: %f\n", t, x, y);
  /*
  printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", 
      h[0][0], h[0][1], h[0][2],
      h[1][0], h[1][1], h[1][2],
      h[2][0], h[2][1], h[2][2]);
  */

  float xs = h[0][0] * x + h[0][1] * y + h[0][2];
  float ys = h[1][0] * x + h[1][1] * y + h[1][2];
  float ws = h[2][0] * x + h[2][1] * y + h[2][2];
  // printf("xs: %f, ys: %f, ws: %f\n", xs, ys, ws);
  return make_float2(xs / ws, ys / ws);
}

__device__ float2 linTransform(
    const Homography *mats, unsigned int num_steps, float t, float2 coord)
{
  // compute a linearly interpolated transformed coordinate, assuming t in [0, 1)
  assert(t >= 0.0);
  assert(t < 1.0);
  float i;
  float rem = modf(t * (num_steps - 1), &i);
  assert(i < num_steps - 1);
  // printf("rem: %f, i: %d\n", rem, (unsigned)i);

  float2 beg = transform(mats, unsigned(i), coord);
  float2 end = transform(mats, unsigned(i)+1, coord);
  // printf("rem: %f, i: %d, beg: %f, %f, end: %f, %f\n", rem, (unsigned)i, beg.x, beg.y, end.x, end.y);
  return make_float2(
      beg.x * (1.0 - rem) + end.x * rem,
      beg.y * (1.0 - rem) + end.y * rem);
}

__inline__ __device__ void warp_min(float2 val, int beg_offset, int end_offset, float2 &min_val) {
  // set min_val to the minimum of  values for lanes 
  min_val = val;
  float2 other;
  for (int offset = beg_offset; offset < end_offset; offset<<=1) {
    other.x = __shfl_down_sync(0xFFFFFFFF, min_val.x, offset);
    other.y = __shfl_down_sync(0xFFFFFFFF, min_val.y, offset);
    min_val.x = min(min_val.x, other.x);
    min_val.y = min(min_val.y, other.y);
  }
}

__inline__ __device__ void warp_max(float2 val, int beg_offset, int end_offset, float2 &max_val) {
  // set max_val to the maximum of  values for lanes 
  max_val = val;
  float2 other;
  for (int offset = beg_offset; offset < end_offset; offset<<=1) {
    other.x = __shfl_down_sync(0xFFFFFFFF, max_val.x, offset);
    other.y = __shfl_down_sync(0xFFFFFFFF, max_val.y, offset);
    max_val.x = max(max_val.x, other.x);
    max_val.y = max(max_val.y, other.y);
  }
}

__inline__ __device__ int wrap_mod(int val, uint mod) {
  // safe, wrapping modulo for signed integers
  return ((val % mod) + mod) % mod;
}

__inline__ __device__ float wrap_fmod(float val, float mod) {
  return fmodf(fmodf(val, mod) + mod, mod);
}

__inline__ __device__ int to_block(float val) {
  return (int)std::floor(val / BLOCK_DIM);
}


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
  extern __shared__ int shbuf[];
  typedef uchar3 PixelBlock[BLOCK_DIM][BLOCK_DIM];
  PixelBlock *pixelBuf = reinterpret_cast<PixelBlock *>(shbuf);

  // defines (bx, by) minimum block found to be occupied in the receptive field
  __shared__ int2 gridOffset;
  // gridIndex[gy][gx] with (gy, gx) block index relative to gridOffset
  // In the first phase, this is a 0-1 mask to show which blocks overlap the receptive field.
  // In second phase, empty blocks are assigned -1, and other blocks are
  // consecutively numbered in row-major order from 0 onwards.
  __shared__ int gridIndex[FIELD_BLOCKS][FIELD_BLOCKS];

  // occupiedBlocks[i] = (gy, gx), index into gridIndex
  __shared__ ushort2 occupiedBlocks[MAX_OCCU_BLOCKS];
  __shared__ uint numOccupiedBlocks;

  // 1. Determine grid bounding box (in global viewport grid indices)
  float xbeg = blockIdx.x * blockDim.x + 0.5;
  float ybeg = blockIdx.y * blockDim.y + 0.5;
  float xend = xbeg + blockDim.x - 1.0;
  float yend = ybeg + blockDim.y - 1.0;
  float2 targetCorners[4] = { { xbeg, ybeg }, { xbeg, yend }, { xend, ybeg }, { xend, yend } };

  // TODO - further optimize?
  if (threadIdx.y < FIELD_BLOCKS && threadIdx.x < FIELD_BLOCKS){
    for (uint y=threadIdx.y; y<FIELD_BLOCKS; y+=blockDim.y) {
      for (uint x=threadIdx.x; x<FIELD_BLOCKS; x += blockDim.x) {
        gridIndex[y][x] = 0;
      }
    }
  }
  if (threadIdx.y == 0 && threadIdx.x == 0){
    gridOffset = make_int2(10000, 10000); // large initial seed values for minimization
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

  if (threadIdx.y < 4) {
    // use one warp for each target square corner
    // use warpSize timesteps to estimate full extent of receptive field
    const float inc = 0.999 / warpSize;
    int cornerIdx = threadIdx.x % 4;
    uint step = (threadIdx.y * blockDim.y + threadIdx.x) / 4;
    float t = step * inc;
    assert(t < 1.0);
    float2 corner = targetCorners[cornerIdx];
    float2 source;
    source = linTransform(trajectoryBuffer, numMats, t, corner);
    /*
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      printf("source: %f, %f\n", source.x, source.y);
    }
    */
    // source = make_float2(10000.0, -10000.0);
    float2 lmin, lmax, gmin;
    warp_min(source, 1, 4, lmin);
    warp_max(source, 1, 4, lmax);
    warp_min(lmin, 4, 16, gmin);
    // warp_max(lmax, 4, 16, gmax);
    __syncthreads();
    atomicMin(&gridOffset.x, to_block(gmin.x));
    atomicMin(&gridOffset.y, to_block(gmin.y));
    __syncthreads();

    // populate for first phase of gridIndex (0-1 occupancy mask)
    // find the bounding grid for a timestep's source quadrilateral.
    // use each of the 32 timesteps in the trajectory.
    if (cornerIdx == 0) {
      uint grid_lmin_x = to_block(lmin.x) - gridOffset.x;
      uint grid_lmax_x = to_block(lmax.x) - gridOffset.x + 1;
      uint grid_lmin_y = to_block(lmin.y) - gridOffset.y;
      uint grid_lmax_y = to_block(lmax.y) - gridOffset.y + 1;
      assert(grid_lmin_x < grid_lmax_x);
      assert(grid_lmin_y < grid_lmax_y);
      for (uint y=grid_lmin_y; y != grid_lmax_y; ++y) {
        for (uint x=grid_lmin_x; x != grid_lmax_x; ++x) {
          atomicExch(&gridIndex[y][x], 1);
        }
      }
    }
  }
  // TODO: is this needed?
  __syncthreads();

  using BlockScan = cub::BlockScan<int, BLOCK_DIM, cub::BLOCK_SCAN_RAKING, BLOCK_DIM>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_data[4];
  uint threadId = threadIdx.y * blockDim.x + threadIdx.x;  // 0:1024
  uint elemId = threadId * 4;                              // 0:4096:4
  uint2 sourceGrid;
  sourceGrid.y = elemId / 64;                              // 0:64
  sourceGrid.x = elemId % 64;                              // 0:64:4

  for (uint c=0; c!=4; c++) {
    thread_data[c] = gridIndex[sourceGrid.y][sourceGrid.x+c];
  }
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

  using BlockReduce = cub::BlockReduce<int, BLOCK_DIM, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 
        BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage br_storage;
  numOccupiedBlocks = BlockReduce(br_storage).Reduce(thread_data, cub::Max()) + 1;

  for (uint c=0; c!=4; c++) {
    if (gridIndex[sourceGrid.y][sourceGrid.x+c] == 1) {
      gridIndex[sourceGrid.y][sourceGrid.x+c] = thread_data[c];
    } else {
      gridIndex[sourceGrid.y][sourceGrid.x+c] = -1;
    }
  }

  __syncthreads();
  // only need to call __syncthreads() if temp_storage is reused
  if (blockIdx.x == 0 && blockIdx.y == 0 && elemId == 0) {
    printf("numOccupiedBlocks: %d\n", numOccupiedBlocks);
    for (uint y=0; y != 64; y++) {
      for (uint x=0; x != 64; x++) {
        printf("%3d", gridIndex[y][x]);
      }
      printf("\n");
    }
  }

  assert(numOccupiedBlocks > 0);

  // if (threadIdx.x == 0 && threadIdx.y == 0) {
    // printf("numOccupiedBlocks: %d\n", numOccupiedBlocks);
  // }

  // occupiedBlocks[i] holds index into gridIndex
  uint threadNum = threadIdx.y * blockDim.x + threadIdx.x;
  uint dataIdx = 4 * threadNum;
  ushort row = (ushort)(dataIdx / 64);
  ushort col = dataIdx % 64;
  for (ushort c=0; c!=4; c++) {
    uint idx = thread_data[c];
    if (gridIndex[row][col+c] >= 0 && idx < MAX_OCCU_BLOCKS) {
      occupiedBlocks[idx] = make_ushort2(row, col+c);
    }
  }
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadNum < numOccupiedBlocks) {
    ushort2 blk = occupiedBlocks[threadNum];
    printf("i: %d, occupiedBlock(adjusted): (%d, %d)\n", 
        threadNum, blk.y + gridOffset.y, blk.x + gridOffset.x);
  }

  // occupiedBlocks, gridOffset are now initialized.
  // iterate in phases of loading pixel data, then accumulating it
  // how many timesteps should we use per occupiedBlock?
  uint numTimesteps = (uint)(numOccupiedBlocks * stepsPerOccuBlock); 
  uint3 outColor = make_uint3(0, 0, 0);
  uint numAccum = 0;
  uint nextBlock = 0; // the next block to load
  uint blockBegin = 0;
  uint nextLayer = 0; // index into pixelBuf first dimension
  float2 sourceLoc; // location of the source in the viewport
  uint2 sourceGridLoc; // location of the grid block
  // Each iteration loads one layer of pixelBuf
  curandState randState;
  float2 targetLoc = make_float2(
      blockIdx.x * blockDim.x + threadIdx.x + 0.5,
      blockIdx.y * blockDim.y + threadIdx.y + 0.5);


  while (nextBlock < numOccupiedBlocks) {
    ushort2 blockCoord = occupiedBlocks[nextBlock++];
    int x = (blockCoord.x + gridOffset.x) * BLOCK_DIM + threadIdx.x;
    int y = (blockCoord.y + gridOffset.y) * BLOCK_DIM + threadIdx.y;
    // wrap-around, safe with negative numbers
    x = wrap_mod(x, inputWidth);
    y = wrap_mod(y, inputHeight);
    uint inIdx = y * inputWidth + x; 
    pixelBuf[nextLayer][threadIdx.y][threadIdx.x] = image[inIdx];
    nextLayer++;

    if (nextLayer == numPixelLayers || nextBlock == numOccupiedBlocks) {
      // process all blocks in [blockBegin, nextBlock)
      // buffer is full.  start to unload
      // is it safe to sync here?
      __syncthreads();
      float inc = 1.0 / (float)numTimesteps;
      for (uint i=0; i != numTimesteps; i++) {
        // add a little jitter to get irregular points along the path
        float t = inc * (i + curand_normal(&randState) * 0.05);
        t = max(0.0, min(t, 0.9999));
        assert(t < 1.0);
        sourceLoc = linTransform(trajectoryBuffer, numMats, t, targetLoc);
        // TODO: wrapping logic
        sourceGridLoc = make_uint2(
            to_block(sourceLoc.x) - gridOffset.x, 
            to_block(sourceLoc.y) - gridOffset.y);
        uint block = gridIndex[sourceGridLoc.y][sourceGridLoc.x];
        if (block >= blockBegin && block < nextBlock) {
          uint2 pixelOffset = make_uint2(
              wrap_fmod(sourceLoc.x, BLOCK_DIM),
              wrap_fmod(sourceLoc.y, BLOCK_DIM));
          uint layer = block - blockBegin;
          assert(layer < numPixelLayers); 
          assert(pixelOffset.x < BLOCK_DIM);
          assert(pixelOffset.y < BLOCK_DIM);
          uchar3 thisColor = pixelBuf[layer][pixelOffset.y][pixelOffset.x];
          /*
          if (blockIdx.x == 3 && blockIdx.y == 3 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("layer: %d, sourceLoc: %f, %f, pixelOffset: %d, %d\n", layer, 
                sourceLoc.y, sourceLoc.x, pixelOffset.y, pixelOffset.x);
          }
          */
          
          outColor.x += thisColor.x;
          outColor.y += thisColor.y;
          outColor.z += thisColor.z;
          numAccum += 1;
        }
      } // finished processing blocks [blockBegin, nextBlock) 
      blockBegin = nextBlock;
      nextLayer = 0;
    }
  }
  uint2 targetPixel = make_uint2(targetLoc.x, targetLoc.y);
  assert(numOccupiedBlocks > 0);
  assert(numAccum > 0);
  // if (true) return;
  if (targetPixel.x < viewportWidth && targetPixel.y < viewportHeight) {
    uint outIdx = targetPixel.y * viewportWidth + targetPixel.x;
    int cx = roundf((float)outColor.x / (float)numAccum);
    int cy = roundf((float)outColor.y / (float)numAccum);
    int cz = roundf((float)outColor.z / (float)numAccum);
    if (blockIdx.x == 3 && blockIdx.y == 3) {
      printf("color: %d %d %d\n", cx, cy, cz);
    }
    blurred[outIdx].x = cx;
    blurred[outIdx].y = cy;
    blurred[outIdx].z = cz;
    /*
    blurred[outIdx].x = 255;
    blurred[outIdx].y = 128;
    blurred[outIdx].z = 39;
    cx = max(0, min(cx, 255));
    */
    // blurred[outIdx].x = static_cast<unsigned char>(cx);
    // blurred[outIdx].y = outColor.y / numOccupiedBlocks;
    // blurred[outIdx].z = outColor.z / numOccupiedBlocks;
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

  assert(numMats <= MAX_TRAJECTORIES);
  size_t trajectorySize = sizeof(Homography) * numMats;
  CUDA_CHECK(cudaMemcpyToSymbol(trajectoryBuffer, trajectory, trajectorySize)); 
  // CUDA_CHECK(cudaMalloc((void **)&d_trajectory, trajectorySize));
  // CUDA_CHECK(cudaMemcpy(d_trajectory, trajectory, trajectorySize, cudaMemcpyHostToDevice));

  dim3 dimGrid = dim3(
      ceil_ratio(viewportWidth, BLOCK_DIM), 
      ceil_ratio(viewportHeight, BLOCK_DIM), 1);

  dim3 dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
  size_t sharedBytes = numPixelLayers * BLOCK_DIM * BLOCK_DIM * 3; 
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



