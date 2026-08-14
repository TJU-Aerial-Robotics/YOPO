#include <simsense/aggr.h>

namespace simsense {

__device__ __forceinline__
int findMin(int a1, int a2, int a3, int a4) {
    int min1 = a1 < a2 ? a1 : a2;
    int min2 = a3 < a4 ? a3 : a4;
    int min = min1 < min2 ? min1 : min2;
    return min;
}

__global__
void aggrLeft2Right(cost_t *d_cost, cost_t *d_L, const int p1, const int p2, const int rows, const int cols, const int maxDisp) {
    const int y = blockIdx.x; // HxWxD, H position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int min;
    extern __shared__ cost_t sharedAggr[]; // length of maxDisp

    // First iteration
    int x = 0; // HxWxD, W position
    const int cost = d_cost[(y*cols+x)*maxDisp+d];
    d_L[(y*cols+x)*maxDisp+d] = cost;

    if (d == 0) { min = cost; }
    __syncthreads();
    atomicMin(&min, cost);
    sharedAggr[d] = cost;

    // Remaining iterations
    for (x = 1; x < cols; x++) {
        const int cost = d_cost[(y*cols+x)*maxDisp+d];
        __syncthreads();

        int left, right;
        if (d != 0) {
            left = sharedAggr[d-1];
        }
        if (d != maxDisp-1) {
            right = sharedAggr[d+1];
        }
        
        int localMin;
        if (d != 0 && d != maxDisp-1) {
            localMin = findMin(sharedAggr[d], left+p1, right+p1, min+p2);
        } else if (d == 0) {
            localMin = findMin(sharedAggr[d], right+p1, right+p1, min+p2);
        } else {
            localMin = findMin(sharedAggr[d], left+p1, left+p1, min+p2);
        }
        int aggr = cost + localMin - min;
        d_L[(y*cols+x)*maxDisp+d] = aggr;
        __syncthreads();

        if (d == 0) { min = aggr; }
        __syncthreads();
        atomicMin(&min, aggr);
        sharedAggr[d] = aggr;
    }
}

__global__
void aggrRight2Left(cost_t *d_cost, cost_t *d_L, const int p1, const int p2, const int rows, const int cols, const int maxDisp) {
    const int y = blockIdx.x; // HxWxD, H position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int min;
    extern __shared__ cost_t sharedAggr[]; // length of maxDisp

    // First iteration
    int x = cols-1; // HxWxD, W position
    const int cost = d_cost[(y*cols+x)*maxDisp+d];
    d_L[(y*cols+x)*maxDisp+d] = cost;

    if (d == 0) { min = cost; }
    __syncthreads();
    atomicMin(&min, cost);
    sharedAggr[d] = cost;

    // Remaining iterations
    for (x = cols-2; x >= 0; x--) {
        const int cost = d_cost[(y*cols+x)*maxDisp+d];
        __syncthreads();

        int left, right;
        if (d != 0) {
            left = sharedAggr[d-1];
        }
        if (d != maxDisp-1) {
            right = sharedAggr[d+1];
        }
        
        int localMin;
        if (d != 0 && d != maxDisp-1) {
            localMin = findMin(sharedAggr[d], left+p1, right+p1, min+p2);
        } else if (d == 0) {
            localMin = findMin(sharedAggr[d], right+p1, right+p1, min+p2);
        } else {
            localMin = findMin(sharedAggr[d], left+p1, left+p1, min+p2);
        }
        int aggr = cost + localMin - min;
        d_L[(y*cols+x)*maxDisp+d] = aggr;
        __syncthreads();

        if (d == 0) { min = aggr; }
        __syncthreads();
        atomicMin(&min, aggr);
        sharedAggr[d] = aggr;
    }
}

__global__
void aggrTop2Bottom(cost_t *d_cost, cost_t *d_L, const int p1, const int p2, const int rows, const int cols, const int maxDisp) {
    const int x = blockIdx.x; // HxWxD, W position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int min;
    extern __shared__ cost_t sharedAggr[]; // length of maxDisp

    // First iteration
    int y = 0; // HxWxD, H position
    const int cost = d_cost[(y*cols+x)*maxDisp+d];
    d_L[(y*cols+x)*maxDisp+d] = cost;

    if (d == 0) { min = cost; }
    __syncthreads();
    atomicMin(&min, cost);
    sharedAggr[d] = cost;

    // Remaining iterations
    for (y = 1; y < rows; y++) {
        const int cost = d_cost[(y*cols+x)*maxDisp+d];
        __syncthreads();

        int left, right;
        if (d != 0) {
            left = sharedAggr[d-1];
        }
        if (d != maxDisp-1) {
            right = sharedAggr[d+1];
        }
        
        int localMin;
        if (d != 0 && d != maxDisp-1) {
            localMin = findMin(sharedAggr[d], left+p1, right+p1, min+p2);
        } else if (d == 0) {
            localMin = findMin(sharedAggr[d], right+p1, right+p1, min+p2);
        } else {
            localMin = findMin(sharedAggr[d], left+p1, left+p1, min+p2);
        }
        int aggr = cost + localMin - min;
        d_L[(y*cols+x)*maxDisp+d] = aggr;
        __syncthreads();

        if (d == 0) { min = aggr; }
        __syncthreads();
        atomicMin(&min, aggr);
        sharedAggr[d] = aggr;
    }
}

__global__
void aggrBottom2Top(cost_t *d_cost, cost_t *d_LAll, cost_t *d_L0, cost_t *d_L1, cost_t *d_L2,
                    const int p1, const int p2, const int rows, const int cols, const int maxDisp) {
    const int x = blockIdx.x; // HxWxD, W position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int min;
    extern __shared__ cost_t sharedAggr[]; // length of maxDisp

    // First iteration
    int y = rows-1; // HxWxD, H position
    int pos = (y*cols+x)*maxDisp+d;
    const int cost = d_cost[pos];
    d_LAll[pos] = (cost + d_L0[pos] + d_L1[pos] + d_L2[pos]) / 4;

    if (d == 0) { min = cost; }
    __syncthreads();
    atomicMin(&min, cost);
    sharedAggr[d] = cost;

    // Remaining iterations
    for (y = rows-2; y >= 0; y--) {
        pos = (y*cols+x)*maxDisp+d;
        const int cost = d_cost[pos];
        __syncthreads();

        int left, right;
        if (d != 0) {
            left = sharedAggr[d-1];
        }
        if (d != maxDisp-1) {
            right = sharedAggr[d+1];
        }
        
        int localMin;
        if (d != 0 && d != maxDisp-1) {
            localMin = findMin(sharedAggr[d], left+p1, right+p1, min+p2);
        } else if (d == 0) {
            localMin = findMin(sharedAggr[d], right+p1, right+p1, min+p2);
        } else {
            localMin = findMin(sharedAggr[d], left+p1, left+p1, min+p2);
        }
        int aggr = cost + localMin - min;
        d_LAll[pos] = (aggr + d_L0[pos] + d_L1[pos] + d_L2[pos]) / 4;
        __syncthreads();

        if (d == 0) { min = aggr; }
        __syncthreads();
        atomicMin(&min, aggr);
        sharedAggr[d] = aggr;
    }
}

}
