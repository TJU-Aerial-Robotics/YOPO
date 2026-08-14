#ifndef SIMSENSE_COST_H
#define SIMSENSE_COST_H

#include <stdint.h>
#include <simsense/config.h>

namespace simsense {

__global__
void hammingCost(const uint32_t *censusL, const uint32_t *censusR, cost_t *d_cost, const int rows, const int cols, const int len);

}

#endif
