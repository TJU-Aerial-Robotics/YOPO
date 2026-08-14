#include "sensor_simulator.cuh"
#include <simsense/core.h>   // 矢量化后的 simsense 引擎（去 pybind 原生 C++ 版）

namespace raycast
{
    GridMap::GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold = 1){
        const float epsilon = 0.001f;   // 避免数值误差导致 (1)建图空行 (2)边缘点被忽略
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        float length = max_pt(0) - min_pt(0) + 2 * epsilon;  // 保证各个边界最大值能被取到
        float width  = max_pt(1) - min_pt(1) + 2 * epsilon;
        float height = max_pt(2) - min_pt(2) + 2 * epsilon;
        Vector3f origin(min_pt(0), min_pt(1), min_pt(2));
        Vector3f map_size(length, width, height);
        origin_x_ = origin.x;
        origin_y_ = origin.y;
        origin_z_ = origin.z;

        Vector3i grid_size;
        grid_size.x = ceil(map_size.x / resolution);
        grid_size.y = ceil(map_size.y / resolution);
        grid_size.z = ceil(map_size.z / resolution);
        int grid_total_size = grid_size.x * grid_size.y * grid_size.z;

        resolution_   = resolution;
        grid_size_x_  = grid_size.x, 
        grid_size_y_  = grid_size.y, 
        grid_size_z_  = grid_size.z, 
        grid_size_yz_ = grid_size.y * grid_size.z;
        occupy_threshold_ = occupy_threshold;
        raycast_step_ = resolution;

        std::vector<int> h_map(grid_total_size, 0);
        // 点云全位于体素边界，有时候会有全空的行，加个很小的偏移
        for (size_t i = 0; i < cloud->points.size(); i++) {
            Vector3f point(cloud->points[i].x + epsilon, cloud->points[i].y + epsilon, cloud->points[i].z + epsilon);
            int idx = Vox2Idx(Pos2Vox(point));
            if (idx < grid_total_size) {
                h_map[idx]++;
            }
        }
        cudaMalloc((void **)&map_cuda_, grid_total_size * sizeof(int));
        cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f &pos)
    {
        Vector3i vox;
        vox.x = floor((pos.x - origin_x_) / resolution_);
        vox.y = floor((pos.y - origin_y_) / resolution_);
        vox.z = floor((pos.z - origin_z_) / resolution_);
        return vox;
    }

    __host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i &vox)
    {
        Vector3f pos;
        pos.x = (vox.x + 0.5f) * resolution_ + origin_x_;
        pos.y = (vox.y + 0.5f) * resolution_ + origin_y_;
        pos.z = (vox.z + 0.5f) * resolution_ + origin_z_;
        return pos;
    }

    __host__ __device__ int GridMap::Vox2Idx(const Vector3i &vox)
    {
        return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
    }

    __host__ __device__ Vector3i GridMap::Idx2Vox(int idx)
    {
        return Vector3i(idx / grid_size_yz_, (idx % grid_size_yz_) / grid_size_z_, idx % grid_size_z_);
    }

    __device__ int GridMap::symmetricIndex(int index, int length)
    {
        index = index % (2 * length - 2);
        if (index < 0)
        {
            index += (2 * length - 2);
        }

        if (index >= length)
        {
            index = 2 * length - 2 - index;
        }
        return index;
    }

    // -1: z越界; 0: 空闲; 1: 占据
    __device__  int GridMap::mapQuery(const Vector3f &pos){
        Vector3i vox = Pos2Vox(pos);
        vox.x = symmetricIndex(vox.x, grid_size_x_);
        vox.y = symmetricIndex(vox.y, grid_size_y_);

        if (vox.z >= grid_size_z_)
            return 0;
        if (vox.z <= 0)
            return 1;

        int idx = Vox2Idx(vox);
        if (map_cuda_[idx] > occupy_threshold_)
            return 1;
        return 0;        
    }

    // 整数哈希 (Wang hash)，把一个整数打散成均匀分布
    __device__ __forceinline__ unsigned int wangHash(unsigned int s)
    {
        s = (s ^ 61u) ^ (s >> 16);
        s *= 9u;
        s = s ^ (s >> 4);
        s *= 0x27d4eb2du;
        s = s ^ (s >> 15);
        return s;
    }

    // 世界空间伪随机散斑：按 cell 把世界点量化到立方格，每格一个随机灰度。
    // 只依赖世界坐标 -> 左右视角对同一表面点得到相同灰度 -> 可双目匹配（模拟结构光）。
    __device__ __forceinline__ unsigned char speckle3D(const float3 &p, const TextureParams &tex)
    {
        int ix = (int)floorf(p.x / tex.cell);
        int iy = (int)floorf(p.y / tex.cell);
        int iz = (int)floorf(p.z / tex.cell);
        unsigned int h = wangHash(((unsigned int)(ix * 73856093)) ^
                                  ((unsigned int)(iy * 19349663)) ^
                                  ((unsigned int)(iz * 83492791)) ^ tex.seed);
        float r = (h & 0xFFFFFFu) / (float)0xFFFFFFu; // [0,1)
        float val = tex.base + tex.amplitude * (r - 0.5f);
        val = fminf(255.0f, fmaxf(0.0f, val));
        return (unsigned char)(val + 0.5f);
    }

    // 天空/远处弱纹理：只依赖【世界视线方向】，左右相机同一像素方向相同 -> 视差≈0 处匹配
    // (=远/无效)。给无纹理背景提供可竞争纹理，避免近处障碍边缘被前景膨胀“撑胖”。
    __device__ __forceinline__ unsigned char skyTexture(const float3 &dir, const TextureParams &tex)
    {
        int ix = (int)floorf(dir.x * tex.sky_scale);
        int iy = (int)floorf(dir.y * tex.sky_scale);
        int iz = (int)floorf(dir.z * tex.sky_scale);
        unsigned int h = wangHash(((unsigned int)(ix * 73856093)) ^
                                  ((unsigned int)(iy * 19349663)) ^
                                  ((unsigned int)(iz * 83492791)) ^ (tex.seed + 777u));
        float r = (h & 0xFFFFFFu) / (float)0xFFFFFFu;
        float val = tex.base + tex.sky_amplitude * (r - 0.5f);
        val = fminf(255.0f, fmaxf(0.0f, val));
        return (unsigned char)(val + 0.5f);
    }

    // 与 cameraRaycastKernel 相同的射线步进，但额外输出散斑纹理灰度图。
    // 深度仍用 voxel 量化点（避免摩尔纹）；纹理用连续命中点（更细的斑点）。
    __global__ void cameraRaycastIRKernel(float* depth_values, unsigned char* ir_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc, TextureParams tex)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        if (u < camera_param.image_width && v < camera_param.image_height)
        {
            float y = -(u - camera_param.cx) / camera_param.fx;
            float z = -(v - camera_param.cy) / camera_param.fy;
            float x = 1.0f;

            float length = sqrtf(x * x + y * y + z * z);
            x /= length; y /= length; z /= length;

            float dx = 1.0 * grid_map.raycast_step_;
            float dy = (y / x) * dx;
            float dz = (z / x) * dx;

            // 增量世界点更新：把每步的 SE3 变换提到循环外（world点 = 相机光心 + scale*step_w）
            float3 step_w = T_wc.rotate(make_float3(dx, dy, dz));
            float3 point_w = T_wc * make_float3(0.0f, 0.0f, 0.0f); // 相机光心(世界系)
            float point_x = 0.0f;
            float depth = 0.0f;
            unsigned char ir = (unsigned char)tex.background;

            while (1)
            {
                point_x += dx;
                point_w.x += step_w.x;
                point_w.y += step_w.y;
                point_w.z += step_w.z;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                if (occupied == 1)
                {
                    // 纹理：连续命中点（步进精度 ~0.5*resolution，比 voxel 更细）
                    ir = speckle3D(point_w, tex);
                    // 深度：voxel 量化点（与原 renderDepthImage 保持一致）
                    Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                    Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                    float3 occ_point_w_ = make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                    float3 occ_point_c_ = T_wc.inv() * occ_point_w_;
                    depth = occ_point_c_.x;
                    break;
                }

                if (point_x >= camera_param.max_depth_dist){
                    depth = camera_param.max_depth_dist;
                    // 远/无命中：给一层按视线方向的弱纹理（匹配在视差≈0=远），背景不再是无纹理空洞。
                    if (tex.sky_texture) {
                        float3 raydir_w = T_wc.rotate(make_float3(x, y, z));
                        ir = skyTexture(raydir_w, tex);
                    }
                    break;
                }
            }

            if (camera_param.normalize_depth)
                depth = depth / camera_param.max_depth_dist;
            int idx = v * camera_param.image_width + u;
            depth_values[idx] = depth;
            ir_values[idx] = ir;
        }
    }

    __global__ void cameraRaycastKernel(float* depth_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);

        if (u < camera_param.image_width && v < camera_param.image_height)
        {
            // 计算射线方向
            float y = -(u - camera_param.cx) / camera_param.fx;
            float z = -(v - camera_param.cy) / camera_param.fy;
            float x = 1.0f;

            // 归一化射线方向
            float length = sqrtf(x * x + y * y + z * z);
            x /= length;
            y /= length;
            z /= length;

            // 计算每个轴的增量比例 (x方向固定步长避免近距离处畸变; 0.5是瞎设的防止过于稀疏导致错误)
            float dx = 0.5 * grid_map.raycast_step_;
            float dy = (y / x) * dx;
            float dz = (z / x) * dx;

            // 增量世界点更新：把每步的 SE3 变换提到循环外（world点 = 相机光心 + scale*step_w）
            float3 step_w = T_wc.rotate(make_float3(dx, dy, dz));
            float3 point_w = T_wc * make_float3(0.0f, 0.0f, 0.0f); // 相机光心(世界系)
            float point_x = 0.0f;
            float depth = 0.0f;

            while (1)
            {
                point_x += dx;
                point_w.x += step_w.x;
                point_w.y += step_w.y;
                point_w.z += step_w.z;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                if (occupied == 1)
                {
                    // depth = point_x;  // 直接这样赋值会有一点误差
                    // 栅格化避免平面变曲面 (有些冗余，但在机体系栅格化会有类似摩尔纹的东西)
                    Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                    Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                    float3 occ_point_w_ = make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                    float3 occ_point_c_ = T_wc.inv() * occ_point_w_;
                    depth = occ_point_c_.x;
                    break;
                }

                if (point_x >= camera_param.max_depth_dist){
                    depth = camera_param.max_depth_dist;
                    break;
                }
            }

            // 将深度值存储到输出数组中
            if (camera_param.normalize_depth)
                depth = depth / camera_param.max_depth_dist;
            depth_values[v * camera_param.image_width + u] = depth;
        }
    }

    void renderDepthImage(GridMap* grid_map, CameraParams* camera_param, cudaMat::SE3<float>& T_wc, cv::Mat& depth_image)
    {   
        float* depth_values;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));

        // 在GPU上启动核函数
        cameraRaycastKernel<<<camera_param->image_height, camera_param->image_width>>>(depth_values, *grid_map, *camera_param, T_wc);
        
        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);

        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(depth_values);
        return;
    }

    void renderDepthAndIR(GridMap* grid_map, CameraParams* camera_param, cudaMat::SE3<float>& T_wc, const TextureParams& tex, cv::Mat& depth_image, cv::Mat& ir_image)
    {
        float* depth_values;
        unsigned char* ir_values;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));
        cudaMallocManaged(&ir_values, num_elements * sizeof(unsigned char));

        cameraRaycastIRKernel<<<camera_param->image_height, camera_param->image_width>>>(depth_values, ir_values, *grid_map, *camera_param, T_wc, tex);

        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);
        ir_image.create(camera_param->image_height, camera_param->image_width, CV_8UC1);

        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ir_image.data, ir_values, num_elements * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cudaFree(depth_values);
        cudaFree(ir_values);
        return;
    }

    // 与 renderDepthImage 并列：渲染左右伪红外 -> 直接喂 simsense 算深度（无中间红外发布）。
    void renderStereoImage(GridMap* grid_map, CameraParams* camera_param,
                           cudaMat::SE3<float>& T_wc_left, cudaMat::SE3<float>& T_wc_right,
                           const TextureParams& tex, simsense::DepthSensorEngine* engine,
                           cv::Mat& depth_image)
    {
        // 渲染左右两路伪红外（深度真值这里不需要，丢弃）
        cv::Mat depth_dummy, ir_left, ir_right;
        renderDepthAndIR(grid_map, camera_param, T_wc_left, tex, depth_dummy, ir_left);
        renderDepthAndIR(grid_map, camera_param, T_wc_right, tex, depth_dummy, ir_right);

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);

        // 直接调用 simsense 引擎计算深度（rectified=true，已在构造时配置）
        engine->compute(ir_left.data, ir_right.data, reinterpret_cast<float*>(depth_image.data));
        return;
    }

    __global__ void lidarRaycastKernel(Vector3f* point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc)
    {
        int h = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);
        if (h < lidar_param.horizontal_num && v < lidar_param.vertical_lines)
        {   
            float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
            float vertical_angle = lidar_param.vertical_angle_start + v * vertical_resolution;
            float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
            float cos_vert = std::cos(vertical_angle * M_PI / 180.0);
            float horizontal_angle = h * lidar_param.horizontal_resolution;
            float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
            float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);
            // 计算射线方向
            Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);

            // 计算每个轴的增量比例
            float dx = ray_direction.x * grid_map.raycast_step_;
            float dy = ray_direction.y * grid_map.raycast_step_;
            float dz = ray_direction.z * grid_map.raycast_step_;

            // 递增射线方向上的每个轴
            int scale = 0;
            Vector3f point_value(0, 0, 0);

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                float ray_length = sqrtf(point_x * point_x + point_y * point_y + point_z * point_z);

                if (occupied == 1)
                {
                    point_value = Vector3f(point_x, point_y, point_z);
                    Vector3i vox_body = grid_map.Pos2Vox(point_value);  // 栅格化避免平面变曲面
                    point_value = grid_map.Vox2Pos(vox_body);
                    break;
                }

                if (ray_length > lidar_param.max_lidar_dist){
                    break;
                }
            }

            // 将点云值存储到输出数组中，(0, 0, 0)为无效值
            point_values[v * lidar_param.horizontal_num + h] = point_value;
        }
    }

    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wc, pcl::PointCloud<pcl::PointXYZ>& lidar_points){
        Vector3f* point_values;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&point_values, num_elements * sizeof(Vector3f));

        // 在GPU上启动核函数
        lidarRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(point_values, *grid_map, *lidar_param, T_wc);
        
        cudaDeviceSynchronize();

        std::vector<Vector3f> cpu_points(num_elements);
        cudaMemcpy(cpu_points.data(), point_values, num_elements * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        
        lidar_points.points.clear();
        lidar_points.points.reserve(num_elements);
        
        for (const auto& point : cpu_points) {
            if (point.x != 0 || point.y != 0 || point.z != 0) {
                lidar_points.points.emplace_back(point.x, point.y, point.z);
            }
        }
        cudaFree(point_values);
        return;
    }


    
}