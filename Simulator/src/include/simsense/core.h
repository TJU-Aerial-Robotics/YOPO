#ifndef SIMSENSE_CORE_H
#define SIMSENSE_CORE_H

// 本文件是 simsense 的 core.h 在 Simulator 工程内的“去 pybind”原生 C++ 版本。
// 原版接口用 py::array_t（依赖 Python 解释器），这里改为 raw 指针，便于直接在 C++ 里调用。
// 仅保留无 registration 的双目深度路径（理想校正双目：内参相同、纯平移、无畸变）。

#include <stdint.h>
#include <iostream>
#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <simsense/config.h>

namespace simsense {

// 主传感器类（原生 C++ 接口）
class DepthSensorEngine {
public:
    // 无 registration 构造。rectified=true 时不需要 rectify 映射（map_* 可传 nullptr）。
    DepthSensorEngine(
        uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen, float _minDepth, float _maxDepth, uint64_t infraredNoiseSeed,
        float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma, bool _rectified, uint8_t _censusWidth, uint8_t _censusHeight,
        uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1, uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff,
        uint8_t _mfSize, const float *map_lx = nullptr, const float *map_ly = nullptr, const float *map_rx = nullptr, const float *map_ry = nullptr
    );

    // 计算深度：left/right 为 rows*cols 的 uint8 主机缓冲；out_host 为 rows*cols 的 float 主机缓冲（米，0=无效）。
    void compute(const uint8_t *left_host, const uint8_t *right_host, float *out_host);

    void setInfraredNoiseParameters(float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma);
    void setPenalties(uint8_t _p1, uint8_t _p2);
    void setCensusWindowSize(uint8_t _censusWidth, uint8_t _censusHeight);
    void setMatchingBlockSize(uint8_t _bfWidth, uint8_t _bfHeight);
    void setUniquenessRatio(uint8_t _uniqRatio);
    void setLrMaxDiff(uint8_t _lrMaxDiff);

    ~DepthSensorEngine();

protected:
    cudaStream_t stream1, stream2, stream3;
    curandState_t *d_irNoiseStates0, *d_irNoiseStates1;
    float *d_mapLx, *d_mapLy, *d_mapRx, *d_mapRy, *d_a1, *d_a2, *d_a3;
    uint8_t *d_rawim0, *d_rawim1, *d_noisyim0, *d_noisyim1, *d_recim0, *d_recim1;
    uint32_t *d_census0, *d_census1;
    cost_t *d_rawcost, *d_hsum, *d_cost, *d_L0, *d_L1, *d_L2, *d_LAll;
    float *d_leftDisp, *d_filteredDisp, *d_depth, *d_rgbDepth, *d_dilatedDepth, *h_disp, *h_depth;
    uint16_t *d_rightDisp;
    float speckleShape, speckleScale, gaussianMu, gaussianSigma;
    uint8_t censusWidth, censusHeight, bfWidth, bfHeight, p1, p2, uniqRatio, lrMaxDiff, mfSize;
    uint32_t rows, cols, size, maxDisp, rgbRows, rgbCols, rgbSize;
    float focalLen, baselineLen, minDepth, maxDepth, b1, b2, b3;
    bool rectified, registration, dilation;
};

}

#endif
