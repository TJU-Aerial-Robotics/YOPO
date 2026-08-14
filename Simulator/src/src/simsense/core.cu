#include <simsense/core.h>
#include <simsense/camera.h>
#include <simsense/csct.h>
#include <simsense/cost.h>
#include <simsense/aggr.h>
#include <simsense/wta.h>
#include <simsense/lrcheck.h>
#include <simsense/filter.h>

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
   }
}

namespace simsense {

// Constructor without registration
DepthSensorEngine::DepthSensorEngine(
        uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen, float _minDepth, float _maxDepth, uint64_t infraredNoiseSeed,
        float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma, bool _rectified, uint8_t _censusWidth, uint8_t _censusHeight,
        uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1, uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff,
        uint8_t _mfSize, const float *map_lx, const float *map_ly, const float *map_rx, const float *map_ry
) {
    // Create streams and free memory if necessary
    gpuErrCheck(cudaStreamCreate(&stream1));
    gpuErrCheck(cudaStreamCreate(&stream2));
    gpuErrCheck(cudaStreamCreate(&stream3));

    // Intialize class variables
    speckleShape = _speckleShape;
    speckleScale = _speckleScale;
    gaussianMu = _gaussianMu;
    gaussianSigma = _gaussianSigma;
    censusWidth = _censusWidth;
    censusHeight = _censusHeight;
    bfWidth = _bfWidth;
    bfHeight = _bfHeight;
    p1 = _p1;
    p2 = _p2;
    uniqRatio = _uniqRatio;
    lrMaxDiff = _lrMaxDiff;
    mfSize = _mfSize;
    rows = _rows;
    cols = _cols;
    size = rows*cols;
    maxDisp = _maxDisp;
    int size3d = size*maxDisp;
    focalLen = _focalLen;
    baselineLen = _baselineLen;
    minDepth = _minDepth;
    maxDepth = _maxDepth;
    rectified = _rectified;
    registration = false;

    // Allocate GPU memory
    gpuErrCheck(cudaMalloc((void **)&d_irNoiseStates0, sizeof(curandState_t)*size));
    gpuErrCheck(cudaMalloc((void **)&d_irNoiseStates1, sizeof(curandState_t)*size));
    initInfraredNoise<<<(size + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE, 0, stream1>>>(
      static_cast<curandState_t *>(d_irNoiseStates0), infraredNoiseSeed, size);
    initInfraredNoise<<<(size + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE, 0, stream2>>>(
      static_cast<curandState_t *>(d_irNoiseStates1), infraredNoiseSeed + 1, size);

    gpuErrCheck(cudaMalloc((void **)&d_mapLx, sizeof(float)*size));
    gpuErrCheck(cudaMalloc((void **)&d_mapLy, sizeof(float)*size));
    gpuErrCheck(cudaMalloc((void **)&d_mapRx, sizeof(float)*size));
    gpuErrCheck(cudaMalloc((void **)&d_mapRy, sizeof(float)*size));
    if (map_lx && map_ly && map_rx && map_ry) {
        gpuErrCheck(cudaMemcpyAsync(d_mapLx, map_lx, sizeof(float)*size, cudaMemcpyHostToDevice));
        gpuErrCheck(cudaMemcpyAsync(d_mapLy, map_ly, sizeof(float)*size, cudaMemcpyHostToDevice));
        gpuErrCheck(cudaMemcpyAsync(d_mapRx, map_rx, sizeof(float)*size, cudaMemcpyHostToDevice));
        gpuErrCheck(cudaMemcpyAsync(d_mapRy, map_ry, sizeof(float)*size, cudaMemcpyHostToDevice));
    }

    gpuErrCheck(cudaMalloc((void **)&d_rawim0, sizeof(uint8_t)*size));
    gpuErrCheck(cudaMalloc((void **)&d_rawim1, sizeof(uint8_t)*size));
    gpuErrCheck(cudaMalloc((void **)&d_noisyim0, sizeof(uint8_t)*size));
    gpuErrCheck(cudaMalloc((void **)&d_noisyim1, sizeof(uint8_t)*size));
    if (!rectified) {
        gpuErrCheck(cudaMalloc((void **)&d_recim0, sizeof(uint8_t)*size));
        gpuErrCheck(cudaMalloc((void **)&d_recim1, sizeof(uint8_t)*size));
    }

    gpuErrCheck(cudaMalloc((void **)&d_census0, sizeof(uint32_t)*size));
    gpuErrCheck(cudaMalloc((void **)&d_census1, sizeof(uint32_t)*size));

    if (bfWidth * bfHeight != 1) {
        gpuErrCheck(cudaMalloc((void **)&d_rawcost, sizeof(cost_t)*size3d));
        gpuErrCheck(cudaMalloc((void **)&d_hsum, sizeof(cost_t)*size3d));
    }
    gpuErrCheck(cudaMalloc((void **)&d_cost, sizeof(cost_t)*size3d));

    gpuErrCheck(cudaMalloc((void **)&d_L0, sizeof(cost_t)*size3d));
    gpuErrCheck(cudaMalloc((void **)&d_L1, sizeof(cost_t)*size3d));
    gpuErrCheck(cudaMalloc((void **)&d_L2, sizeof(cost_t)*size3d));
    gpuErrCheck(cudaMalloc((void **)&d_LAll, sizeof(cost_t)*size3d));

    gpuErrCheck(cudaMalloc((void **)&d_leftDisp, sizeof(float)*size));
    gpuErrCheck(cudaMalloc((void **)&d_rightDisp, sizeof(uint16_t)*size));

    if (mfSize != 1) {
        gpuErrCheck(cudaMalloc((void **)&d_filteredDisp, sizeof(float)*size));
    }

#ifdef DISP_ONLY
    h_disp = new float[size];
#else
    gpuErrCheck(cudaMalloc((void **)&d_depth, sizeof(float)*size));
    h_depth = new float[size];
#endif

    gpuErrCheck(cudaDeviceSynchronize());
}

void DepthSensorEngine::compute(const uint8_t *left_host, const uint8_t *right_host, float *out_host) {
    // Upload to GPU（调用方保证 left/right 为 rows*cols 的 uint8 主机缓冲）
    gpuErrCheck(cudaMemcpyAsync(d_rawim0, left_host, sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpyAsync(d_rawim1, right_host, sizeof(uint8_t)*size, cudaMemcpyHostToDevice));

#ifdef PRINT_RUNTIME
    float runtime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    uint8_t *d_srcim0 = d_rawim0;
    uint8_t *d_srcim1 = d_rawim1;
    // Infrared Noise Simulation
    if (speckleShape > 0) {
#ifdef PRINT_RUNTIME
        cudaEventRecord(start);
#endif
        simInfraredNoise<<<(size+WARP_SIZE-1)/(WARP_SIZE), WARP_SIZE, 0, stream1>>>(d_srcim0, d_noisyim0, d_irNoiseStates0, rows, cols, speckleShape, speckleScale, gaussianMu, gaussianSigma);
        simInfraredNoise<<<(size+WARP_SIZE-1)/(WARP_SIZE), WARP_SIZE, 0, stream2>>>(d_srcim1, d_noisyim1, d_irNoiseStates1, rows, cols, speckleShape, speckleScale, gaussianMu, gaussianSigma);
#ifdef PRINT_RUNTIME
        cudaEventRecord(stop);
        gpuErrCheck(cudaDeviceSynchronize());
        cudaEventElapsedTime(&runtime, start, stop);
        printf("Runtime of IR noise simulation: %f ms\n", runtime);
#endif
        d_srcim0 = d_noisyim0;
        d_srcim1 = d_noisyim1;
    }

    // Rectification
    if (!rectified) {
        gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
        cudaEventRecord(start);
#endif
        remap<<<(size+8*WARP_SIZE-1)/(8*WARP_SIZE), 8*WARP_SIZE, 0, stream1>>>(d_mapLx, d_mapLy, d_srcim0, d_recim0, rows, cols);
        remap<<<(size+8*WARP_SIZE-1)/(8*WARP_SIZE), 8*WARP_SIZE, 0, stream2>>>(d_mapRx, d_mapRy, d_srcim1, d_recim1, rows, cols);
#ifdef PRINT_RUNTIME
        cudaEventRecord(stop);
        gpuErrCheck(cudaDeviceSynchronize());
        cudaEventElapsedTime(&runtime, start, stop);
        printf("Runtime of rectification: %f ms\n", runtime);
#endif
        d_srcim0 = d_recim0;
        d_srcim1 = d_recim1;
    }

    // Center-symmetric census transform
    dim3 CSCTBlockSize;
    CSCTBlockSize.x = WARP_SIZE;
    CSCTBlockSize.y = WARP_SIZE;
    dim3 CSCTGridSize;
    CSCTGridSize.x = (cols+CSCTBlockSize.x-1) / CSCTBlockSize.x;
    CSCTGridSize.y = (rows+CSCTBlockSize.y-1) / CSCTBlockSize.y;
    int CSCTWinCols = (WARP_SIZE+censusWidth-1);
    int CSCTWinRows = (WARP_SIZE+censusHeight-1);
    int CSCTSharedMemSize = 2*CSCTWinCols*CSCTWinRows*sizeof(uint8_t);
    gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
    cudaEventRecord(start);
#endif
    CSCT<<<CSCTGridSize, CSCTBlockSize, CSCTSharedMemSize, stream1>>>(d_srcim0, d_srcim1, d_census0, d_census1, rows, cols, censusWidth, censusHeight);
#ifdef PRINT_RUNTIME
    cudaEventRecord(stop);
    gpuErrCheck(cudaDeviceSynchronize());
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Runtime of CSCT: %f ms\n", runtime);
#endif

    // Hamming distance
    dim3 costGridSize;
    costGridSize.x = (cols+maxDisp-1) / maxDisp;
    costGridSize.y = rows;
    gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
    cudaEventRecord(start);
#endif
    if (bfWidth * bfHeight == 1) {
        hammingCost<<<costGridSize, maxDisp, 3*maxDisp*sizeof(uint32_t), stream1>>>(d_census0, d_census1, d_cost, rows, cols, maxDisp);
    } else {
        // Apply box filter
        hammingCost<<<costGridSize, maxDisp, 3*maxDisp*sizeof(uint32_t), stream1>>>(d_census0, d_census1, d_rawcost, rows, cols, maxDisp);
        gpuErrCheck(cudaDeviceSynchronize());
        boxFilterHorizontal<<<rows, maxDisp, 0, stream1>>>(d_rawcost, d_hsum, rows, cols, maxDisp, bfWidth);
        gpuErrCheck(cudaDeviceSynchronize());
        boxFilterVertical<<<cols, maxDisp, 0, stream1>>>(d_hsum, d_cost, rows, cols, maxDisp, bfHeight);
    }
#ifdef PRINT_RUNTIME
    cudaEventRecord(stop);
    gpuErrCheck(cudaDeviceSynchronize());
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Runtime of cost calculation + box filter: %f ms\n", runtime);
#endif

    // Cost aggregation
    gpuErrCheck(cudaDeviceSynchronize());
    int P1 = p1 * bfWidth * bfHeight;
    int P2 = p2 * bfWidth * bfHeight;
    aggrLeft2Right<<<rows, maxDisp, maxDisp*sizeof(cost_t), stream1>>>(d_cost, d_L0, P1, P2, rows, cols, maxDisp);
    aggrRight2Left<<<rows, maxDisp, maxDisp*sizeof(cost_t), stream2>>>(d_cost, d_L1, P1, P2, rows, cols, maxDisp);
    aggrTop2Bottom<<<cols, maxDisp, maxDisp*sizeof(cost_t), stream3>>>(d_cost, d_L2, P1, P2, rows, cols, maxDisp);
    gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
    cudaEventRecord(start);
#endif
    aggrBottom2Top<<<cols, maxDisp, maxDisp*sizeof(cost_t), stream1>>>(d_cost, d_LAll, d_L0, d_L1, d_L2, P1, P2, rows, cols, maxDisp);
#ifdef PRINT_RUNTIME
    cudaEventRecord(stop);
    gpuErrCheck(cudaDeviceSynchronize());
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Runtime of cost aggregation: %f ms\n", runtime);
#endif

    // Winner takes all
    dim3 WTAGridSize;
    WTAGridSize.x = cols;
    WTAGridSize.y = rows;
    int WTAThrNum = ((maxDisp+WARP_SIZE-1)/WARP_SIZE) * WARP_SIZE;
    gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
    cudaEventRecord(start);
#endif
    winnerTakesAll<<<WTAGridSize, WTAThrNum, maxDisp*sizeof(cost_t), stream1>>>(d_LAll, d_leftDisp, d_rightDisp, cols, maxDisp, uniqRatio);
#ifdef PRINT_RUNTIME
    cudaEventRecord(stop);
    gpuErrCheck(cudaDeviceSynchronize());
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Runtime of winner takes all: %f ms\n", runtime);
#endif

    if (lrMaxDiff != 255) {
        gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
        cudaEventRecord(start);
#endif
        lrConsistencyCheck<<<(size+8*WARP_SIZE-1)/(8*WARP_SIZE), 8*WARP_SIZE, 0, stream1>>>(d_leftDisp, d_rightDisp, rows, cols, lrMaxDiff);
#ifdef PRINT_RUNTIME
        cudaEventRecord(stop);
        gpuErrCheck(cudaDeviceSynchronize());
        cudaEventElapsedTime(&runtime, start, stop);
        printf("Runtime of left-right consistency check: %f ms\n", runtime);
#endif
    }

    float *d_finalDisp = d_leftDisp;
    // Apply median filter
    if (mfSize != 1) {
        int mfSharedMemSize = 2*WARP_SIZE * mfSize*mfSize * sizeof(float);
        gpuErrCheck(cudaDeviceSynchronize());
#ifdef PRINT_RUNTIME
    cudaEventRecord(start);
#endif
        medianFilter<<<(size+2*WARP_SIZE-1)/(2*WARP_SIZE), 2*WARP_SIZE, mfSharedMemSize, stream1>>>(d_finalDisp, d_filteredDisp, rows, cols, mfSize);
#ifdef PRINT_RUNTIME
    cudaEventRecord(stop);
    gpuErrCheck(cudaDeviceSynchronize());
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Runtime of median filter: %f ms\n", runtime);
#endif
        d_finalDisp = d_filteredDisp;
    }

    // Convert disparity into depth, then clamp to [minDepth, maxDepth)
    gpuErrCheck(cudaDeviceSynchronize());
    disp2Depth<<<(size+8*WARP_SIZE-1)/(8*WARP_SIZE), 8*WARP_SIZE, 0, stream1>>>(d_finalDisp, d_depth, size, focalLen, baselineLen);
    correctDepthRange<<<(size+8*WARP_SIZE-1)/(8*WARP_SIZE), 8*WARP_SIZE, 0, stream1>>>(d_depth, size, minDepth, maxDepth);

    // GPU -> 调用方主机缓冲（米，0=无效）
    gpuErrCheck(cudaDeviceSynchronize());
    gpuErrCheck(cudaMemcpy(out_host, d_depth, sizeof(float)*size, cudaMemcpyDeviceToHost));
}

void DepthSensorEngine::setInfraredNoiseParameters(float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma) {
    speckleShape = _speckleShape;
    speckleScale = _speckleScale;
    gaussianMu = _gaussianMu;
    gaussianSigma = _gaussianSigma;
}

void DepthSensorEngine::setPenalties(uint8_t _p1, uint8_t _p2) {
    p1 = _p1;
    p2 = _p2;
}

void DepthSensorEngine::setCensusWindowSize(uint8_t _censusWidth, uint8_t _censusHeight) {
    censusWidth = _censusWidth;
    censusHeight = _censusHeight;
}

void DepthSensorEngine::setMatchingBlockSize(uint8_t _bfWidth, uint8_t _bfHeight) {
    bfWidth = _bfWidth;
    bfHeight = _bfHeight;
}

void DepthSensorEngine::setUniquenessRatio(uint8_t _uniqRatio) {
    uniqRatio = _uniqRatio;
}

void DepthSensorEngine::setLrMaxDiff(uint8_t _lrMaxDiff) {
    lrMaxDiff = _lrMaxDiff;
}

DepthSensorEngine::~DepthSensorEngine() {
    gpuErrCheck(cudaStreamDestroy(stream1));
    gpuErrCheck(cudaStreamDestroy(stream2));
    gpuErrCheck(cudaStreamDestroy(stream3));

    gpuErrCheck(cudaFree(d_irNoiseStates0));
    gpuErrCheck(cudaFree(d_irNoiseStates1));

    gpuErrCheck(cudaFree(d_mapLx));
    gpuErrCheck(cudaFree(d_mapLy));
    gpuErrCheck(cudaFree(d_mapRx));
    gpuErrCheck(cudaFree(d_mapRy));

    gpuErrCheck(cudaFree(d_rawim0));
    gpuErrCheck(cudaFree(d_rawim1));
    gpuErrCheck(cudaFree(d_noisyim0));
    gpuErrCheck(cudaFree(d_noisyim1));
    if (!rectified) {
        gpuErrCheck(cudaFree(d_recim0));
        gpuErrCheck(cudaFree(d_recim1));
    }

    gpuErrCheck(cudaFree(d_census0));
    gpuErrCheck(cudaFree(d_census1));
    
    if (bfWidth * bfHeight != 1) {
        gpuErrCheck(cudaFree(d_rawcost));
        gpuErrCheck(cudaFree(d_hsum));
    }
    gpuErrCheck(cudaFree(d_cost));

    gpuErrCheck(cudaFree(d_L0));
    gpuErrCheck(cudaFree(d_L1));
    gpuErrCheck(cudaFree(d_L2));
    gpuErrCheck(cudaFree(d_LAll));

    gpuErrCheck(cudaFree(d_leftDisp));
    gpuErrCheck(cudaFree(d_rightDisp));

    if (mfSize != 1) {
        gpuErrCheck(cudaFree(d_filteredDisp));
    }

#ifdef DISP_ONLY
    delete h_disp;
#else
    gpuErrCheck(cudaFree(d_depth));
    if (registration) {
        gpuErrCheck(cudaFree(d_a1));
        gpuErrCheck(cudaFree(d_a2));
        gpuErrCheck(cudaFree(d_a3));
        gpuErrCheck(cudaFree(d_rgbDepth));
        if (dilation) {
            gpuErrCheck(cudaFree(d_dilatedDepth));
        }
    }
    delete h_depth;
#endif

    gpuErrCheck(cudaDeviceSynchronize());
}

}
