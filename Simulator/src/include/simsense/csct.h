#ifndef SIMSENSE_CSCT_H
#define SIMSENSE_CSCT_H

#include <stdint.h>
#include <simsense/config.h>

namespace simsense {

__global__
void CSCT(const uint8_t *im0, const uint8_t *im1, uint32_t *census0, uint32_t *census1, 
            const int rows, const int cols, const int censusWidth, const int censusHeight);

}

#endif
