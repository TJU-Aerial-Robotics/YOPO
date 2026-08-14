#include <simsense/csct.h>

namespace simsense {

__global__
void CSCT(const uint8_t *im0, const uint8_t *im1, uint32_t *census0, uint32_t *census1, 
            const int rows, const int cols, const int censusWidth, const int censusHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // x position in image
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // y position in image
    
    const int winCols = (blockDim.x + censusWidth - 1); // # of cols of shared memory
    const int winRows = (blockDim.y + censusHeight - 1); // # of rows of shared memory
    extern __shared__ uint8_t mem[];
    uint8_t *win0 = mem;
    uint8_t *win1 = mem + winCols*winRows;

    // Copy data from memory into shared memory
    const int left = (censusWidth - 1) / 2; // BlockDim.x + 2*left = winCols
    const int top = (censusHeight - 1) / 2; // BlockDim.y + 2*top = winRows
    int pos = threadIdx.y * blockDim.x + threadIdx.x; // Position in block, use this to fill window
    int imx = blockIdx.x * blockDim.x + (pos%winCols) - left;
    int imy = blockIdx.y * blockDim.y + (pos/winCols) - top;
    if (imx < 0 || imx >= cols || imy < 0 || imy >= rows) {
        win0[pos] = 0;
        win1[pos] = 0;
    } else {
        win0[pos] = im0[imy*cols+imx];
        win1[pos] = im1[imy*cols+imx];
    }
    if (pos < winCols*winRows - blockDim.x*blockDim.y) {
        pos = blockDim.x*blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        imx = blockIdx.x * blockDim.x + (pos%winCols) - left;
        imy = blockIdx.y * blockDim.y + (pos/winCols) - top;
        if (imx < 0 || imx >= cols || imy < 0 || imy >= rows) {
        win0[pos] = 0;
        win1[pos] = 0;
        } else {
            win0[pos] = im0[imy*cols+imx];
            win1[pos] = im1[imy*cols+imx];
        }
    }
    __syncthreads();

    // Center-symmetric census transform
    uint32_t result0 = 0;
    uint32_t result1 = 0;
    uint32_t compare;
    int i = 0;

    if (x < cols && y < rows) {
        for (; i < top+1; i++) { // Row increment
            int jmax = (i == top ? censusWidth/2 : censusWidth);
            for (int j = 0; j < jmax; j++) { // Column increment
                const uint8_t im0I0 = win0[(threadIdx.y+i)*winCols + threadIdx.x+j];
                const uint8_t im0I1 = win0[(threadIdx.y+2*top-i)*winCols + threadIdx.x+2*left-j];
                compare = (im0I0 >= im0I1);
                compare <<= (i*censusWidth+j);
                result0 |= compare;
                
                const uint8_t im1I0 = win1[(threadIdx.y+i)*winCols + threadIdx.x+j];
                const uint8_t im1I1 = win1[(threadIdx.y+2*top-i)*winCols + threadIdx.x+2*left-j];
                compare = (im1I0 >= im1I1);
                compare <<= (i*censusWidth+j);
                result1 |= compare;
            }
        }

        census0[y*cols+x] = result0;
        census1[y*cols+x] = result1;
        }
}

}
