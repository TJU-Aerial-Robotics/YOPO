#ifndef SIMSENSE_AGGR_H
#define SIMSENSE_AGGR_H

#include <stdint.h>
#include <simsense/config.h>

namespace simsense {

__global__
void aggrLeft2Right(cost_t *d_cost, cost_t *d_L, const int p1, const int p2, const int rows, const int cols, const int maxDisp);

__global__
void aggrRight2Left(cost_t *d_cost, cost_t *d_L, const int p1, const int p2, const int rows, const int cols, const int maxDisp);

__global__
void aggrTop2Bottom(cost_t *d_cost, cost_t *d_L, const int p1, const int p2, const int rows, const int cols, const int maxDisp);

// Compute total aggregation cost of four paths
__global__
void aggrBottom2Top(cost_t *d_cost, cost_t *d_LAll, cost_t *d_L0, cost_t *d_L1, cost_t *d_L2,
                    const int p1, const int p2, const int rows, const int cols, const int maxDisp);
}

#endif
