#ifndef _DIMS_H
#define _DIMS_H
#include <cuda_runtime.h>

#define MAX_HOMOGRAPHY_MATS 64 

#define BLOCK_DIM_X 64
// #define BLOCK_DIM_X 32 
#define BLOCK_DIM_Y 16
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)
#define OUT_DIM_X BLOCK_DIM_X
#define OUT_DIM_Y (BLOCK_DIM_Y * 4)

typedef float Homography[3][3];

#endif // _DIMS_H
