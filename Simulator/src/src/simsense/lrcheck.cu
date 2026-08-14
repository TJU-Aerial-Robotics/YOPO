#include <simsense/lrcheck.h>

namespace simsense {

__global__
void lrConsistencyCheck(float *d_leftDisp, const uint16_t *d_rightDisp, const int rows, const int cols, const int lrMaxDiff) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = pos % cols;
    const int y = pos / cols;
    if (y >= rows) { return; }

    int ld = (int)round(d_leftDisp[pos]);
    if (ld < 0 || x-ld < 0 || abs(ld - d_rightDisp[y*cols + x-ld]) > lrMaxDiff) {
        d_leftDisp[pos] = -1;
    }
}

}
