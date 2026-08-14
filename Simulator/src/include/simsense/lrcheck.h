#ifndef SIMSENSE_LRCHECK_H
#define SIMSENSE_LRCHECK_H

#include <cmath>
#include <simsense/config.h>

namespace simsense {

__global__
void lrConsistencyCheck(float *d_leftDisp, const uint16_t *d_rightDisp, const int rows, const int cols, const int lrMaxDiff);

}

#endif
