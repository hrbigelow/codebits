#ifndef _BLUR_H
#define _BLUR_H

#include <cuda_runtime.h>

typedef float Homography[3][3];

void motionBlur(
    Homography *trajectory,
    unsigned int numMats,
    const uchar3 *image,
    unsigned int inputWidth,
    unsigned int inputHeight,
    unsigned int numPixelLayers,
    float stepsPerOccuBlock,
    uchar3 *blurred,
    unsigned int viewportWidth,
    unsigned int viewportHeight);

#endif // _BLUR_H
