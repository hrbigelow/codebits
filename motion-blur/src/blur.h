#ifndef _BLUR_H
#define _BLUR_H

#include "dims.h"

void motionBlur(
    Homography *trajectory,
    unsigned int numMats,
    const uchar3 *image,
    unsigned int inputWidth,
    unsigned int inputHeight,
    int pixel_buf_bytes,
    float stepsPerOccuBlock,
    uchar3 *blurred,
    unsigned int viewportWidth,
    unsigned int viewportHeight,
    float exposure_mul);

#endif // _BLUR_H
