#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define BLOCK_DIM 32; 
// maximum square side in pixels
#define SOURCE_FIELD_SIDE 2048 
#define FIELD_BLOCKS (unsigned int)(SOURCE_FIELD_SIZE / BLOCK_DIM)

// maximum number of occupied blocks
#define MAX_OCCU_BLOCKS 512

struct GridRect {
  unsigned xmin, xmax;
  unsigned ymin, ymax;
};

__device__ transform(float *ptr, unsigned int t, float2 coord)
{
  float x = coord.x;
  float y = coord.y;
  float (*mats)[3][3] = (float (*)[3][3]) ptr;
  float xs = mats[t][0][0] * x + mats[t][0][1] * y + mats[t][0][2];
  float ys = mats[t][1][0] * x + mats[t][1][1] * y + mats[t][1][2];
  float ws = mats[t][2][0] * x + mats[t][2][1] * y + mats[t][2][2];
  return make_float2(xs / ws, ys / ws);
}

__device__ float2 linTransform(float *ptr, float t, unsigned num_steps, float2 coord)
{
  // compute a linearly interpolated transformed coordinate, assuming t in [0, 1)
  assert(t >= 0.0 && t < 1.0);
  float i;
  float rem = modf(t * num_steps, &i);
  assert(i < num_steps);

  unsigned v = unsigned(t * num_steps);
  float2 beg = transform(ptr, unsigned(i), coord);
  float2 end = transform(ptr, unsigned(i)+1, coord);
  return make_float2(
      beg.x * (1.0 - rem) + end.x * rem,
      beg.y * (1.0 - rem) + end.y * rem);
}

__inline__ __device__ void warp_min(float2 val, int beg_offset, int end_offset, float2 &min_val) {
  // set min_val to the minimum of  values for lanes 
  min_val = val;
  float2 other;
  for (int offset = beg; offset < end; offset<<=1) {
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
  for (int offset = beg; offset < end; offset<<=1) {
    other.x = __shfl_down_sync(0xFFFFFFFF, max_val.x, offset);
    other.y = __shfl_down_sync(0xFFFFFFFF, max_val.y, offset);
    max_val.x = max(max_val.x, other.x);
    max_val.y = max(max_val.y, other.y);
  }
}


__global__ motion_blur(const uchar3 *image, uint width, uint height, uint numPixelBlocks,
    float stepsPerOccuBlock, uchar3 *blurred) {
  /* `numPixelBlocks` is number of blocks worth of image data to accumulate at a
   * given time
   */
  extern __shared__ void shbuf[];
  typedef uchar3 PixelBlock[BLOCK_DIM][BLOCK_DIM];
  __shared__ PixelBlock *pixelBuf = reinterpret_cast<PixelBlock *>(shbuf);

  // defines extent and location of block-aligned receptive field
  __shared__ GridRect gridRect = { 10_000, -10_000, 10_000, -10_000 };
  // In the first phase, this is a 0-1 mask to show which blocks overlap the receptive field.
  // In second phase, empty blocks are assigned -1, and other blocks are
  // consecutively numbered in row-major order from 0 onwards.
  __shared__ int gridIndex[FIELD_BLOCKS][FIELD_BLOCKS];
  // __shared__ ushort2 occupiedBlocks[MAX_OCCU_BLOCKS];
  __shared__ uint numOccupiedBlocks;
  __shared__ uchar3 output[BLOCK_DIM][BLOCK_DIM];

  // 1. Determine grid bounding box (in global viewport grid indices)
  float xbeg = blockIdx.x * blockDim.x + 0.5;
  float ybeg = blockIdx.y * blockDim.y + 0.5;
  float xend = xbeg + blockDim.x - 1.0;
  float yend = ybeg + blockDim.y - 1.0;
  float2 targetCorners[4] = { { xbeg, ybeg }, { xbeg, yend }, { xend, ybeg }, { xend, yend } };

  // TODO - further optimize?
  if (threadIdx.y < FIELD_BLOCKS && threadIdx.x < FIELD_BLOCKS){
    for (uint y=0; y<FIELD_BLOCKS; y += blockDim.y) {
      for (uint x=0; x<FIELD_BLOCKS; x += blockDim.x) {
        gridIndex[y][x] = 0;
      }
    }
  }
  __syncthreads();
  
  /*
     estimate receptive field gridRect in viewport grid using 32 timesteps.
     Also, populate the gridIndex based on the grid extent of each timestep.

     Work is divided into four warps with each warp containing 8 timesteps.  groups
     of four lanes each represent the four corners of the target squre for that
     timestep.
   */
  if (threadIdx.y < 4) {
    // use one warp for each target square corner
    // use warpSize timesteps to estimate full extent of receptive field
    const float inc = 0.999 / (warpSize - 1);
    int corner = threadIdx.x % 4;
    uint step = threadIdx.y * blockDim.y + threadIdx.x;
    float t = step * inc;
    float2 corner = targetCorners[corner];
    float2 source = linTransform(trajectory, t, warpSize, corner);
    float2 lmin, lmax, gmin, gmax;
    warp_min(source, 1, 4, lmin);
    warp_max(source, 1, 4, lmax);
    warp_min(lmin, 4, 16, gmin);
    warp_max(lmax, 4, 16, gmax);
    __syncthreads();
    atomicMin(&gridRect.xmin, uint(gmin.x / blockDim.x));
    atomicMax(&gridRect.xmax, uint(gmax.x / blockDim.x) + 1);
    atomicMin(&gridRect.ymin, uint(gmin.y / blockDim.y));
    atomicMax(&gridRect.ymax, uint(gmax.y / blockDim.y) + 1);
    __syncthreads();

    // populate for first phase of gridIndex (0-1 occupancy mask)
    // find the bounding grid for a timestep's source quadrilateral.
    // use each of the 32 timesteps in the trajectory.
    if (corner == 0) {
      uint grid_lmin_x = uint(lmin.x / blockDim.x);
      uint grid_lmax_x = uint(lmax.x / blockDim.x) + 1;
      uint grid_lmin_y = uint(lmin.y / blockDim.y);
      uint grid_lmax_y = uint(lmax.y / blockDim.y) + 1;
      for (int y=grid_lmin_y; y != grid_lmax_y; ++y) {
        for (int x=grid_lmin_x; x != grid_lmax_x; ++x) {
          atomicExch(&gridIndex[y][x], 1);
        }
      }
    }
  }
  // TODO: is this needed?
  __syncthreads();

  using BlockScan = cub::BlockScan<int, BLOCK_DIM, BLOCK_SCAN_RANKING, BLOCK_DIM>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_data[4];
  uint threadId = threadIdx.y * blockDim.y + threadIdx.x;
  uint elemId = threadId * 4;
  uint2 sourceGrid = make_uint2(elemId / 64, elemId % 64);

  for (uint c=0; c!=4; c++) {
    thread_data[c] = gridIndex[sourceGrid.y][sourceGrid.x+c];
  }
  BlockScan::InclusiveScan(thread_data, thread_data);

  for (uint c=0; c!=4; c++) {
    if (gridIndex[sourceGrid.y][sourceGrid.x+c] == 1) {
      gridIndex[sourceGrid.y][sourceGrid.x+c] = thread_data[c];
    }
  }
  // only need to call __syncthreads() if temp_storage is reused

  // find maximum value in thread_data
  int maxVal = max(thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
  for (uint off=1; off < warpSize; off<<=1) {
    int other = __shfl_down_sync(0xFFFFFFFF, maxVal, off);
    maxVal = max(maxVal, other);
  }
  __shared__ maxVals[BLOCK_DIM];
  if (threadIdx.x == 0) {
    maxVals[threadIdx.y] = maxVal;
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    maxVal = maxVals[threadIdx.x];
    for (uint off=1; off < warpSize; off<<=1) {
      int other = __shfl_down_sync(0xFFFFFFFF, maxVal, off);
      maxVal = max(maxVal, other);
    }
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    numOccupiedBlocks = min(maxVal + 1, MAX_OCCU_BLOCKS);
  }
  __syncthreads();

  /*
  uint dataIdx = 4 * threadNum;
  ushort row = (ushort)(dataIdx / 64);
  ushort col = dataIdx % 64;
  for (ushort c=0; c!=4; c++) {
    uint idx = thread_data[c];
    if (gridIndex[row][col+c] && idx < MAX_OCCU_BLOCKS) {
      occupiedBlocks[idx] = make_ushort2(row, col+c);
    }
  }
  */

  // occupiedBlocks, gridRect are now initialized.
  // iterate in phases of loading pixel data, then accumulating it
  // how many timesteps should we use per occupiedBlock?
  uint numTimesteps = (uint)(numOccupiedBlocks * stepsPerOccuBlock); 
  uint3 outColor = make_uint3(0, 0, 0);
  uint blockEnd = 0; // the next block to load
  uint blockBegin = 0;
  uint layerIndex = 0; // index into pixelBuf first dimension
  float2 sourceLoc; // location of the source in the viewport
  uint2 sourceGridLoc; // location of the grid block
  /*
   * Each iteration loads one layer of pixelBuf
   */
  while (blockEnd < numOccupiedBlocks) {
    ushort2 blockCoord = occupiedBlocks[blockEnd];
    blockEnd++;
    uint x = blockCoord.x * BLOCK_DIM + threadIdx.x;
    uint y = blockCoord.y * BLOCK_DIM + threadIdx.y;
    uint inIdx = y * width + x; 
    pixelBuf[layerIndex][threadIdx.y][threadIdx.x] = image[inIdx];
    layerIndex++;

    if (layerIndex == numPixelBlocks || blockEnd == numOccupiedBlocks) {
      // buffer is full.  start to unload
      float inc = 1.0 / (float)numTimesteps;
      for (uint i=0; i != numTimesteps; i++) {
        // add a little jitter to get irregular points along the path
        float t = inc * i + curand_normal(&randState) * inc * 0.05;
        sourceLoc = linTransform(trajectory, t, numTimesteps, targetLoc);
        sourceGridLoc = make_uint2(
            sourceLoc.x / BLOCK_DIM - gridRect.xmin, 
            sourceLoc.y / BLOCK_DIM - gridRect.ymin);
        uint blockIndex = gridIndex[sourceGridLoc.y][sourceGridLoc.x];
        if (blockIndex >= blockBegin && blockIndex < blockEnd) {
          uint2 pixelOffset = make_uint2(
              fmod(sourceLoc.x, BLOCK_DIM),
              fmod(sourceLoc.y, BLOCK_DIM));
          uint layer = blockIndex - blockBegin;
          uchar3 thisColor =  pixelBuf[layer][pixelOffset.x][pixelOffset.y];
          outColor.x += thisColor.x;
          outColor.y += thisColor.y;
          outColor.z += thisColor.z;
        }
      }
      layerIndex = 0;
    }
  }
  uint2 targetLoc = make_uint2(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y);
  uint outIdx = targetLoc.y * width + targetLoc.x;
  blurred[outIdx].x = outColor.x / numOccupiedBlocks;
  blurred[outIdx].y = outColor.y / numOccupiedBlocks;
  blurred[outIdx].z = outColor.z / numOccupiedBlocks;
}





