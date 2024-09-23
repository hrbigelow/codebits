#ifndef _DIMS_H
#define _DIMS_H
#include <cuda_runtime.h>

#define QUERY_X 27
#define QUERY_Y 5

#define BLOCK_DIM 32
// maximum square side in pixels
// #define SOURCE_FIELD_SIZE 2048 
#define SOURCE_FIELD_SIZE 1024
#define FIELD_BLOCKS (unsigned int)(SOURCE_FIELD_SIZE / BLOCK_DIM)

// maximum number of occupied blocks
#define MAX_OCCU_BLOCKS 512

typedef float Homography[3][3];
typedef uchar3 PixelBlock[BLOCK_DIM][BLOCK_DIM];
typedef int GridIndex[FIELD_BLOCKS][FIELD_BLOCKS];

#endif // _DIMS_H
