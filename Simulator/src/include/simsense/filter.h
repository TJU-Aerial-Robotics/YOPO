#ifndef SIMSENSE_FILTER_H
#define SIMSENSE_FILTER_H

#include <stdint.h>
#include <simsense/config.h>

namespace simsense {

__global__
void boxFilterHorizontal(const cost_t *in, cost_t *out, const int rows, const int cols, const int maxDisp, const int size);

__global__
void boxFilterVertical(const cost_t *in, cost_t *out, const int rows, const int cols, const int maxDisp, const int size);

__global__
void medianFilter(const float *in, float *out, const int rows, const int cols, const int size);

}

#endif
