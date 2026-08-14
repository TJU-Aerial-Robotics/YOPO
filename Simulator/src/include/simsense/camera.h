#ifndef SIMSENSE_CAMERA_H
#define SIMSENSE_CAMERA_H

#include <stdint.h>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <simsense/config.h>

#define ERROR_MARGIN 0.01
#define BORDER_VALUE 0

namespace simsense {

__global__
void initInfraredNoise(curandState_t *states, int seed, const int size);

__global__
void simInfraredNoise(uint8_t *src, uint8_t *dst, curandState_t *states, const int rows, const int cols,
                        float speckleShape, float speckleScale, float gaussianMu, float gaussianSigma);

__global__
void remap(const float *mapx, const float *mapy, const uint8_t *src, uint8_t *dst, const int rows, const int cols);

__global__
void disp2Depth(const float *disp, float *depth, const int size, const float focalLen, const float baselineLen);

__global__
void initRgbDepth(float *rgbDepth, const int size, const float maxDepth);

__global__
void depthRegistration(float *rgbDepth, const float *depth, const float *a1, const float *a2, const float *a3,
                        float b1, float b2, float b3, const int size, const int rgbRows, const int rgbCols);

__global__
void depthDilation(float *dilatedDepth, float *depth, const int rgbRows, const int rgbCols, const float maxDepth);

__global__
void correctDepthRange(float *depth, const int size, const float minDepth, const float maxDepth);

}

#endif
