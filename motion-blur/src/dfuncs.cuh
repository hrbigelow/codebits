#ifndef _DFUNCS_H
#define _DFUNCS_H

#include "dims.h"


#if USE_COMP4 == 1
    typedef uchar4 Pixel;
#else
    typedef uchar3 Pixel;
#endif

__device__ __forceinline__ float2 transform(const Homography *mats, int t, const float2 &coord)
{
    const Homography &h = mats[t];

    float xs = h[0][0] * coord.x + h[0][1] * coord.y + h[0][2];
    float ys = h[1][0] * coord.x + h[1][1] * coord.y + h[1][2];
    float ws = h[2][0] * coord.x + h[2][1] * coord.y + h[2][2];
    float wsinv = 1.0 / ws;
    return make_float2(xs * wsinv, ys * wsinv);
}

__device__ float2 lin_transform(
        const Homography *mats, unsigned int num_mats, float t, const float2 &coord)
{
    // compute a linearly interpolated transformed coordinate, associating each mat
    // with values [0, num_mats) and linearly interpolating t
    // choose i such that 
    int i = min(num_mats-2, max(0, (int)floorf(t)));
    float alpha = t - i;

    float2 beg = transform(mats, i, coord);
    float2 end = transform(mats, i+1, coord);
    return make_float2(
            beg.x * (1.0 - alpha) + end.x * alpha,
            beg.y * (1.0 - alpha) + end.y * alpha);
}

__inline__ __device__ int wrap_mod(int val, uint mod) {
    // safe, wrapping modulo for signed integers
    return ((val % mod) + mod) % mod;
}

__inline__ __device__ float wrap_fmod(float val, float mod) {
    return (int)fmodf(fmodf(val, mod) + mod, mod);
}

#endif // _DFUNCS_H
