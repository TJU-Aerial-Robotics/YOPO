#include <simsense/wta.h>

namespace simsense {

/**
 * Minimum reduce of a warp to find minimum value index.
 *
 * @param val  values to find minimum and second minimum. Final minimum will be
 *             stored in lane 0.
 * @param val2 values to store second minimum. Should be initialized to INT_MAX
 *             during first call. If called recursively, intialize to previous
 *             val2. Final second minimum will be stored in lane 0.
 */
__device__ __forceinline__
void warpMinReduceIdx(int &val, int &idx) {
    for(int d = 1; d < WARP_SIZE; d *= 2) {
        int tmpVal = __shfl_xor_sync(0xffffffff, val, d);
        int tmpIdx = __shfl_xor_sync(0xffffffff, idx, d);
        if(tmpVal < val) {
            val = tmpVal;
            idx = tmpIdx;
        }
    }
}

/**
 * Minimum reduce of a block to find minimum value and index.
 *
 * @param val  values to find minimum and second minimum. Final minimum will be
 *             stored in thread 0.
 * @param val2 values to store second minimum. Should be intialized to INT_MAX.
 */
__device__ __forceinline__
void blockMinReduceIdx(int &val, int &idx) {
    static __shared__ int values[WARP_SIZE], indices[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    warpMinReduceIdx(val, idx);
    if (lane == 0) {
        values[wid] = val;
        indices[wid] = idx;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? values[lane] : INT_MAX;
    idx = (threadIdx.x < blockDim.x / WARP_SIZE) ? indices[lane] : INT_MAX;
    if (wid == 0) {
        warpMinReduceIdx(val, idx);
    }
}

/**
 * Minimum reduce of a warp to find minimum and second minimum value and index.
 *
 * @param val  values to find minimum and second minimum. Final minimum will be
 *             stored in lane 0.
 * @param val2 values to store second minimum. Should be initialized to INT_MAX
 *             during first call. If called recursively, intialize to previous
 *             val2. Final second minimum will be stored in lane 0.
 * @param idx  index of the current lane. Final minimum's index will be stored
 *             in lane 0.
 * @param idx2 Second minimum's index will be stored here in lane 0. This can
 *             be initialized to anything.
 */
__device__ __forceinline__
void warpMinReduceIdx2(int &val, int &val2, int &idx, int &idx2) {
    for(int d = 1; d < WARP_SIZE; d *= 2) {
        int tmpVal = __shfl_xor_sync(0xffffffff, val, d);
        int tmpVal2 = __shfl_xor_sync(0xffffffff, val2, d);
        int tmpIdx = __shfl_xor_sync(0xffffffff, idx, d);
        int tmpIdx2 = __shfl_xor_sync(0xffffffff, idx2, d);
        if(tmpVal < val) {
            if (tmpVal2 < val) {
                val2 = tmpVal2;
                idx2 = tmpIdx2;
            } else {
                val2 = val;
                idx2 = idx;
            }
            val = tmpVal;
            idx = tmpIdx;
        } else {
            if (tmpVal < val2) {
                val2 = tmpVal;
                idx2 = tmpIdx;
            }
        }
    }
}

/**
 * Minimum reduce of a block to find minimum and second minimum value and index.
 *
 * @param val  values to find minimum and second minimum. Final minimum will be
 *             stored in thread 0.
 * @param val2 values to store second minimum. Should be intialized to INT_MAX.
 * @param idx  index of the current thread. Final minimum's index will be stored
 *             in thread 0.
 * @param idx2 Second minimum's index will be stored here in thread 0. This can
 *             be initialized to anything.
 */
__device__ __forceinline__
void blockMinReduceIdx2(int &val, int &val2, int &idx, int &idx2) {
    static __shared__ int values[WARP_SIZE], values2[WARP_SIZE], indices[WARP_SIZE], indices2[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    warpMinReduceIdx2(val, val2, idx, idx2);
    if (lane == 0) {
        values[wid] = val;
        values2[wid] = val2;
        indices[wid] = idx;
        indices2[wid] = idx2;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? values[lane] : INT_MAX;
    val2 = (threadIdx.x < blockDim.x / WARP_SIZE) ? values2[lane] : INT_MAX;
    idx = (threadIdx.x < blockDim.x / WARP_SIZE) ? indices[lane] : INT_MAX;
    idx2 = (threadIdx.x < blockDim.x / WARP_SIZE) ? indices2[lane] : INT_MAX;
    if (wid == 0) {
        warpMinReduceIdx2(val, val2, idx, idx2);
    }
}

__device__ __forceinline__
void warpAndReduce(bool &val) {
    for(int d = 1; d < WARP_SIZE; d *= 2) {
        bool tempVal = __shfl_xor_sync(0xffffffff, val, d);
        bool temp = (val && tempVal);
        val = temp;
    }
}

__device__ __forceinline__
void blockAndReduce(bool &val) {
    static __shared__ bool vals[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    warpAndReduce(val);
    if (lane == 0) {
        vals[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? vals[lane] : true;
    if (wid == 0) {
        warpAndReduce(val);
    }
}

__device__ __forceinline__
float subpixelInterpolation(int d, int y0, int y1, int y2) {
    float a = (1.0*(y2-y0)) / (2.0*(y0-2*y1+y2));
    return d-a;
}

__global__
void winnerTakesAll(cost_t *d_LAll, float *d_leftDisp, uint16_t *d_rightDisp, const int cols, const int maxDisp, const int uniqRatio) {
    const int y = blockIdx.y; // HxWxD, H position
    const int x = blockIdx.x; // HxWxD, W position
    const int d = threadIdx.x; // HxWxD, D position

    __shared__ int leftGlobalMinAggr, leftGlobalMinIdx;
    extern __shared__ cost_t leftTotalAggr[];

    // Compute minimum index
    int leftLocalAggr = (d < maxDisp) ? d_LAll[(y*cols+x)*maxDisp+d] : INT_MAX; // Left
    if (d < maxDisp) { leftTotalAggr[d] = leftLocalAggr; }

    int rightLocalAggr = (d < maxDisp && x+d < cols) ? d_LAll[(y*cols+x+d)*maxDisp+d] : INT_MAX; // Right
    __syncthreads();

    int leftMinAggr = leftLocalAggr; // Left
    int leftMinIdx = d;
    (maxDisp == 32) ? warpMinReduceIdx(leftMinAggr, leftMinIdx) : blockMinReduceIdx(leftMinAggr, leftMinIdx);

    int rightMinAggr = rightLocalAggr; // Right
    int rightMinIdx = d;
    (maxDisp == 32) ? warpMinReduceIdx(rightMinAggr, rightMinIdx) : blockMinReduceIdx(rightMinAggr, rightMinIdx);

    if (d == 0) {
        leftGlobalMinAggr = leftMinAggr; // Left
        leftGlobalMinIdx = leftMinIdx;

        d_rightDisp[y*cols+x] = rightMinIdx; // Right
    }
    __syncthreads();

    // Uniqueness test and sub-pixel interpolation for left disparity
    bool leftUniq = (d < maxDisp) ? (leftLocalAggr*(100-uniqRatio) >= leftGlobalMinAggr*100 || abs(leftGlobalMinIdx-d) <= 1) : true;
    (maxDisp == 32) ? warpAndReduce(leftUniq) : blockAndReduce(leftUniq);
    if (d == 0) {
        float leftDisp = leftMinIdx;
        if (!leftUniq) { // Uniqueness test
            leftDisp = -1;
        } else if (leftMinIdx != 0 && leftMinIdx != maxDisp-1) { // Sub-pixel interpolation
            leftDisp = subpixelInterpolation(leftMinIdx, leftTotalAggr[leftMinIdx-1], leftMinAggr, leftTotalAggr[leftMinIdx+1]);
        }
        d_leftDisp[y*cols+x] = leftDisp;
    }
}

}
