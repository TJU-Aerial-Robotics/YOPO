#include <simsense/filter.h>

namespace simsense {

template<typename T> __device__ __forceinline__
T getMedian(T *arr, int n) { // Selection sort
    T result;
    for (int i = 0; i < n/2+1; i++) {
        int minIdx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }

        if (i == n/2) {
            result = arr[minIdx];
            break;
        }

        // Swap
        T tmp = arr[i];
        arr[i] = arr[minIdx];
        arr[minIdx] = tmp;
    }

    return result;
}

/**
 * This function should be called before boxFilterVertical.
 */
__global__
void boxFilterHorizontal(const cost_t *in, cost_t *out, const int rows, const int cols, const int maxDisp, const int size) {
    const int y = blockIdx.x; // HxWxD, H position
    const int d = threadIdx.x; // HxWxD, D position
    int half = size/2;
    int step = maxDisp;

    const cost_t *inPtr = in + (y*cols)*maxDisp;
    cost_t *outPtr = out + (y*cols)*maxDisp;

    outPtr[d] = 0;
    for(int i = 0; i <= half*step; i += step) {
        int scale = (i == 0) ? half+1 : 1;
        outPtr[d] += inPtr[i+d] * scale;
    }

    for(int i = 1*step; i < cols*step; i += step) {
        const cost_t *toAdd = inPtr + min(i+half*step, (cols-1)*step);
        const cost_t *toSub = inPtr + max(i-(half+1)*step, 0);
        outPtr[i+d] = outPtr[i-1*step+d] + toAdd[d] - toSub[d];
    }
}

/**
 * This function should be called after boxFilterHorizontal.
 */
__global__
void boxFilterVertical(const cost_t *in, cost_t *out, const int rows, const int cols, const int maxDisp, const int size) {
    const int x = blockIdx.x; // HxWxD, W position
    const int d = threadIdx.x; // HxWxD, D position
    int half = size/2;
    int step = cols*maxDisp;

    const cost_t *inPtr = in + x*maxDisp;
    cost_t *outPtr = out + x*maxDisp;

    outPtr[d] = 0;
    for(int i = 0; i <= half*step; i += step) {
        int scale = (i == 0) ? half+1 : 1;
        outPtr[d] += inPtr[i+d] * scale;
    }

    for(int i = 1*step; i < rows*step; i += step) {
        const cost_t *toAdd = inPtr + min(i+half*step, (rows-1)*step);
        const cost_t *toSub = inPtr + max(i-(half+1)*step, 0);
        outPtr[i+d] = outPtr[i-1*step+d] + toAdd[d] - toSub[d];
    }
}

__global__
void medianFilter(const float *in, float *out, const int rows, const int cols, const int size) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = pos % cols;
    const int y = pos / cols;
    if (y >= rows) { return; }

    extern __shared__ float mem[];
    float *win = mem + threadIdx.x*size*size;
    if (x >= size/2 && y >= size/2 && x < cols-(size/2) && y < rows-(size/2)) {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                win[j*size + i] = in[(y-size/2+j)*cols + x-size/2+i];
            }
        }
        out[pos] = getMedian<float>(win, size*size);
    } else {
        out[pos] = in[pos];
    }
}

}
