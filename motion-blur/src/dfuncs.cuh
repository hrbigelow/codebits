#ifndef _DFUNCS_H
#define _DFUNCS_H

#include "dims.h"


#if USE_COMP4 == 1
    typedef uchar4 Pixel;
#else
    typedef uchar3 Pixel;
#endif

typedef Pixel PixelBlock[BLOCK_DIM][BLOCK_DIM];

__device__ float2 transform(const Homography *mats, unsigned int t, float2 coord)
{
    float x = coord.x;
    float y = coord.y;
    const Homography &h = mats[t];

    float xs = h[0][0] * x + h[0][1] * y + h[0][2];
    float ys = h[1][0] * x + h[1][1] * y + h[1][2];
    float ws = h[2][0] * x + h[2][1] * y + h[2][2];
    assert(ws != 0.0);
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

    float2 beg = transform(mats, unsigned(i), coord);
    float2 end = transform(mats, unsigned(i)+1, coord);
    return make_float2(
            beg.x * (1.0 - rem) + end.x * rem,
            beg.y * (1.0 - rem) + end.y * rem);
}

__inline__ __device__ void warp_min(float2 val, int beg_offset, int end_offset, float2 &min_val) {
    // set min_val to the minimum of values for lanes 
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
    return (int)fmodf(fmodf(val, mod) + mod, mod);
}

__inline__ __device__ int to_block(float val) {
    return (int)floorf(val / BLOCK_DIM);
}

__inline__ __device__ int2 get_source_coords(
        const int2 grid_offset, const ushort2 *occu_blocks, 
        int block, int px, int py, int width, int height) {
    // get source world coordinates of upper-left corner of given receptive field block 
    ushort2 coord = occu_blocks[block];
    int sx = (coord.x + grid_offset.x) * BLOCK_DIM + px;
    int sy = (coord.y + grid_offset.y) * BLOCK_DIM + py;
    // wrap-around, safe with negative numbers
    sx = wrap_mod(sx, width);
    sy = wrap_mod(sy, height);
    return make_int2(sx, sy);
}

__inline__ __device__ Pixel get_source_pixel(int2 src, const uchar3 *image, int width) {
    uint idx = src.y * width + src.x; 
#if USE_COMP4 == 1 
    return make_uchar4(image[idx].x, image[idx].y, image[idx].z, 0);
#else
    return make_uchar3(image[idx].x, image[idx].y, image[idx].z);
#endif
}

__inline__ __device__ void get_block_coords(
        const int2 &grid_offset,
        const GridIndex &grid_index,
        const float2 &source_loc,
        int *block, int *px, int *py) {
    // find the block, px and py corresponding to source_loc.
    // source_loc is in world coordinates (which can be negative)
    // if source_loc points to an unoccupied block, block will be set to -1

    int gx = to_block(source_loc.x) - grid_offset.x;
    int gy = to_block(source_loc.y) - grid_offset.y;
    *block = grid_index[gy][gx];
    *px = wrap_fmod(source_loc.x, BLOCK_DIM);
    *py = wrap_fmod(source_loc.y, BLOCK_DIM);
}

#endif // _DFUNCS_H
