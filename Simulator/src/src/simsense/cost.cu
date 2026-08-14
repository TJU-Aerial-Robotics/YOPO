#include <simsense/cost.h>

namespace simsense {

__global__
void hammingCost(const uint32_t *censusL, const uint32_t *censusR, cost_t *d_cost, const int rows, const int cols, const int maxDisp) {
    const int x = blockIdx.x * maxDisp;
    const int y = blockIdx.y;
    const int thrId = threadIdx.x;

    extern __shared__ uint32_t mem[];
    uint32_t *left = mem; // Capacity is maxDisp
    uint32_t *right = mem + maxDisp; // Capacity is 2*maxDisp

    if (x+thrId < cols) {
        left[thrId] = censusL[y*cols + x+thrId];
        right[thrId+maxDisp] = censusR[y*cols + x+thrId];
    }
    right[thrId] = (x == 0) ? censusR[y*cols] : censusR[y*cols + x+thrId-maxDisp];
    __syncthreads();

    int imax = maxDisp;
    if (cols % maxDisp != 0 && blockIdx.x == gridDim.x-1) {
        imax = cols % maxDisp;
    }
    for (int i = 0; i < imax; i++) {
        const int base = left[i];
        const int match = right[i+maxDisp-thrId];
        d_cost[(y*cols + x+i)*maxDisp + thrId] = __popc(base^match);
    }
}

}
