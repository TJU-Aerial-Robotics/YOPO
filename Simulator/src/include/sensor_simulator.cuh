#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include "cuda_toolkit/se3.cuh"
#include <cmath>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h> // For pcl::getMinMax3D
#include <pcl/point_cloud.h>   // For pcl::PointCloud
#include <pcl/point_types.h>   // For pcl::PointXYZ
#include <chrono>

namespace simsense { class DepthSensorEngine; } // 前置声明，避免在头里引入 simsense/core.h

namespace raycast
{
    struct Vector3f
    {
        float x, y, z;
        __device__ __host__ Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
        __device__ __host__ Vector3f(float x_val, float y_val, float z_val)
            : x(x_val), y(y_val), z(z_val) {}
    };

    struct Vector3i
    {
        int x, y, z;
        __device__ __host__ Vector3i() : x(0), y(0), z(0) {}
        __device__ __host__ Vector3i(int x_val, int y_val, int z_val)
            : x(x_val), y(y_val), z(z_val) {}
    };

    struct CameraParams
    {
        float fx = 80.0f; // focal length x
        float fy = 80.0f; // focal length y
        float cx = 80.0f; // principal point x (image center)
        float cy = 45.0f; // principal point y (image center)
        int image_width = 160;
        int image_height = 90;
        float max_depth_dist{20};
        bool normalize_depth{false};
    };

    // 伪双目的“结构光”纹理参数：把一个世界空间的伪随机散斑“贴”到场景表面上，
    // 强度只与世界命中点有关，因此左右相机看同一表面点得到同一灰度，可被双目匹配。
    struct TextureParams
    {
        bool enable = true;
        float cell = 0.03f;       // 世界散斑格尺寸 (m)，越小斑点越细。图像斑点≈fx*cell/depth (640宽下~3px@3m)
        float amplitude = 180.0f; // 散斑对比度（灰度幅值）
        float base = 128.0f;      // 基准灰度
        float background = 0.0f;  // 未命中（天空/超距）像素灰度
        unsigned int seed = 12345u;
        // 天空/远处弱纹理：对无命中射线按【世界视线方向】采散斑。无穷远点在左右相机里
        // 视线平行 -> 同一像素方向相同 -> 该纹理在视差≈0 处匹配(=远/无效)，但给背景提供了
        // 可竞争的纹理，避免“无纹理空洞把近处障碍边缘撑胖”。零渲染开销（不需打更远）。
        bool sky_texture = true;
        float sky_scale = 200.0f;      // 天空散斑角频率（越大斑点越细）
        float sky_amplitude = 50.0f;   // 天空散斑对比度（弱一些，模拟远处弱纹理）
    };

    // simsense 双目匹配参数（SGM + 红外噪声），供 DepthSensorEngine 构造。
    struct StereoMatcherParams
    {
        float min_depth = 0.0f;
        float max_depth = 20.0f;
        int max_disp = 192;            // fx*B/min_depth; 192 覆盖最近≈0.25m
        int census_width = 7;
        int census_height = 7;
        int block_width = 7;
        int block_height = 7;
        int p1_penalty = 1;
        int p2_penalty = 16;
        int uniqueness_ratio = 15;
        int lr_max_diff = 1;
        int median_filter_size = 3;
        // 红外噪声 (Landau): shape*scale≈1 保持亮度; shape 越小噪声越大
        float ir_speckle_shape = 500.0f;
        float ir_speckle_scale = 0.002f;
        float ir_gaussian_mu = 0.0f;
        float ir_gaussian_sigma = 1.0f;
        int ir_noise_seed = 0;
    };

    struct LidarParams
    {
        int vertical_lines = 16;            // 纵向16线
        float vertical_angle_start = -15.0; // 起始垂直角度
        float vertical_angle_end = 15.0;    // 结束垂直角度
        int horizontal_num = 360;           // 水平360点
        float horizontal_resolution = 1.0;  // 水平分辨率为1度
        float max_lidar_dist{50};
    };

    class GridMap
    {
        public:
            GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold);
            ~GridMap() {};
            void freeGridMap() {cudaFree(map_cuda_);}
            __host__ __device__ Vector3i Pos2Vox(const Vector3f &pos);
            __host__ __device__ Vector3f Vox2Pos(const Vector3i &vox);
            __host__ __device__ int Vox2Idx(const Vector3i &vox);
            __host__ __device__ Vector3i Idx2Vox(int idx);
            __device__ int symmetricIndex(int index, int length);
            __device__ int mapQuery(const Vector3f &pos);

            float raycast_step_; // raycast step
        private:
            // map param
            int *map_cuda_;
            float resolution_;                                           // grid resolution
            float origin_x_, origin_y_, origin_z_;                       // origin coordinates
            int grid_size_x_, grid_size_y_, grid_size_z_, grid_size_yz_; // grid sizes
            int occupy_threshold_;                                        // occupancy threshold
    };

    __global__ void cameraRaycastKernel(float *depth_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc);
    __global__ void cameraRaycastIRKernel(float *depth_values, unsigned char *ir_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc, TextureParams tex);
    __global__ void lidarRaycastKernel(Vector3f* point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc);

    void renderDepthImage(GridMap *grid_map, CameraParams *camera_param, cudaMat::SE3<float>& T_wc, cv::Mat &depth_image);
    // 渲染一路视角，同时输出深度图 (CV_32FC1) 与带散斑纹理的伪红外灰度图 (CV_8UC1)。
    void renderDepthAndIR(GridMap *grid_map, CameraParams *camera_param, cudaMat::SE3<float>& T_wc, const TextureParams &tex, cv::Mat &depth_image, cv::Mat &ir_image);

    // 与 renderDepthImage 并列：渲染左右伪红外 -> 直接用 simsense 计算类 RealSense 深度。
    // 不发布中间红外图。engine 为预先构造好的 simsense 引擎（理想校正双目，rectified=true）。
    // 输出 depth_image 为 CV_32FC1（米，0=无效）。
    void renderStereoImage(GridMap *grid_map, CameraParams *camera_param,
                           cudaMat::SE3<float>& T_wc_left, cudaMat::SE3<float>& T_wc_right,
                           const TextureParams &tex, simsense::DepthSensorEngine *engine,
                           cv::Mat &depth_image);

    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wc, pcl::PointCloud<pcl::PointXYZ>& lidar_points);
}
#endif // CUDA_UTILS_CUH
