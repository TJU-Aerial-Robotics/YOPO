#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "sensor_simulator.cuh"
#include <simsense/core.h>   // 内置 simsense 双目匹配引擎（原生 C++ 版）
#include <chrono>
#include "maps.hpp"

using namespace raycast;

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // 读取camera参数
        camera = new CameraParams();
        camera->fx = config["camera"]["fx"].as<float>();
        camera->fy = config["camera"]["fy"].as<float>();
        camera->cx = config["camera"]["cx"].as<float>();
        camera->cy = config["camera"]["cy"].as<float>();
        camera->image_width = config["camera"]["image_width"].as<int>();
        camera->image_height = config["camera"]["image_height"].as<int>();
        camera->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        camera->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
        quat_bc = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

        // 读取lidar参数
        lidar = new LidarParams();
        lidar->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        render_stereo = config["render_stereo"] ? config["render_stereo"].as<bool>() : false;
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();
        depth_pub_duration = ros::Duration(1 / depth_fps);
        lidar_pub_duration = ros::Duration(1 / lidar_fps);
        
        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = config["odom_topic"].as<std::string>();
        std::string depth_topic = config["depth_topic"].as<std::string>();
        std::string lidar_topic = config["lidar_topic"].as<std::string>();
        if (config["stereo_topic"]) stereo_depth_topic_ = config["stereo_topic"].as<std::string>();
        if (config["camera_info_topic"]) camera_info_topic_ = config["camera_info_topic"].as<std::string>();

        // 读取伪双目参数（用世界散斑纹理模拟结构光，供 simsense 计算类 RealSense 深度）
        // 双目相机与上面的深度真值相机解耦，使用独立的分辨率/内参。
        stereo_camera = new CameraParams();
        if (config["stereo"]) {
            YAML::Node st = config["stereo"];
            baseline = st["baseline"].as<float>();       // 双目基线 (m)

            YAML::Node sc = st["camera"];
            stereo_camera->fx = sc["fx"].as<float>();
            stereo_camera->fy = sc["fy"].as<float>();
            stereo_camera->cx = sc["cx"].as<float>();
            stereo_camera->cy = sc["cy"].as<float>();
            stereo_camera->image_width = sc["image_width"].as<int>();
            stereo_camera->image_height = sc["image_height"].as<int>();
            stereo_camera->max_depth_dist = sc["max_depth_dist"].as<float>();
            stereo_camera->normalize_depth = false;      // 双目只用红外图，深度真值走 camera

            // 散斑纹理与 simsense 匹配参数：见 sensor_simulator.cuh 的 TextureParams / StereoMatcherParams
            StereoMatcherParams mp;
            if (render_stereo) {
                // 理想校正双目：focalLen=fx, baselineLen=baseline, rectified=true, 无 rectify 映射(nullptr)
                stereo_engine = new simsense::DepthSensorEngine(
                    (uint32_t)stereo_camera->image_height, (uint32_t)stereo_camera->image_width,
                    stereo_camera->fx, baseline, mp.min_depth, mp.max_depth, (uint64_t)mp.ir_noise_seed,
                    mp.ir_speckle_shape, mp.ir_speckle_scale, mp.ir_gaussian_mu, mp.ir_gaussian_sigma, /*rectified=*/true,
                    (uint8_t)mp.census_width, (uint8_t)mp.census_height, (uint32_t)mp.max_disp,
                    (uint8_t)mp.block_width, (uint8_t)mp.block_height, (uint8_t)mp.p1_penalty, (uint8_t)mp.p2_penalty,
                    (uint8_t)mp.uniqueness_ratio, (uint8_t)mp.lr_max_diff, (uint8_t)mp.median_filter_size);
                printf("Stereo (C++ simsense) ready: %dx%d baseline=%.3f max_disp=%d\n",
                       stereo_camera->image_width, stereo_camera->image_height, baseline, mp.max_disp);
            }
        }

        // 读取地图参数
        bool use_random_map = config["random_map"].as<bool>();
        float resolution = config["resolution"].as<float>();
        int occupy_threshold = config["occupy_threshold"].as<int>();
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("mock_map", 1);
        int seed = config["seed"].as<int>();
        int sizeX = config["x_length"].as<int>();
        int sizeY = config["y_length"].as<int>();
        int sizeZ = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();
        double scale = 1 / resolution;
        sizeX = sizeX * scale;
        sizeY = sizeY * scale;
        sizeZ = sizeZ * scale;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_random_map) {
            printf("1.Generate Random Map... \n");
            mocka::Maps::BasicInfo info;
            info.sizeX      = sizeX;
            info.sizeY      = sizeY;
            info.sizeZ      = sizeZ;
            info.seed       = seed;
            info.scale      = scale;
            info.cloud      = cloud;

            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(type);
        }
        else {
            printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
            if (pcl::io::loadPLYFile(ply_file, *cloud) == -1) {
                PCL_ERROR("Couldn't read PLY file \n");
            }
        }
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "world";

        std::cout<<"Pointloud size:"<<cloud->points.size()<<std::endl;
        printf("2.Mapping... \n");
        grid_map = new GridMap(cloud, resolution, occupy_threshold);
        
        ros::Time next_depth_pub_time = ros::Time::now();
        ros::Time next_lidar_pub_time = ros::Time::now();

        // ROS
        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        stereo_depth_pub_ = nh_.advertise<sensor_msgs::Image>(stereo_depth_topic_, 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        camera_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>(camera_info_topic_, 1, true);
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this, ros::TransportHints().tcpNoDelay());
        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);

        buildCameraInfo();
        publishStaticCameraTF();
        camera_info_.header.stamp = ros::Time::now();
        camera_info_pub_.publish(camera_info_);   // 先发一帧, 保证不渲染深度时内参也可用

        printf("3.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);

    void buildCameraInfo();

    void publishStaticCameraTF();

    void renderDepthCallback(const ros::Time stamp);

    void renderLidarCallback(const ros::Time stamp);

    void timerMapCallback(const ros::TimerEvent &);

private:
    bool render_depth{false};
    bool render_lidar{false};
    bool render_stereo{false};
    float baseline{0.15f};        // 双目基线 (m)
    TextureParams tex;            // 散斑纹理参数
    std::string stereo_depth_topic_{"/stereo/depth"};
    std::string camera_info_topic_{"/camera_info"};
    static constexpr const char* CAMERA_FRAME = "camera_link";   // 光学系: z 前 x 右 y 下 (深度图的 frame_id)
    static constexpr const char* BODY_FRAME = "odom";            // 机体系: x 前 y 左 z 上
    Eigen::Vector3f t_bc{0.0f, 0.0f, 0.05f};                     // 仅为 rviz 显示抬升, 真实相机无此偏移(渲染用 odom 位置)
    simsense::DepthSensorEngine* stereo_engine{nullptr};  // 内置双目匹配引擎
    Eigen::Quaternionf quat;
    Eigen::Quaternionf quat_bc, quat_wc;
    Eigen::Vector3f pos;

    CameraParams* camera;
    CameraParams* stereo_camera;   // 双目独立相机（高分辨率），与 camera 解耦
    LidarParams* lidar;
    GridMap* grid_map;
    sensor_msgs::PointCloud2 output;

    ros::NodeHandle nh_;
    ros::Publisher image_pub_, stereo_depth_pub_, point_cloud_pub_;
    ros::Publisher camera_info_pub_;
    ros::Publisher pcl_pub;
    sensor_msgs::CameraInfo camera_info_;
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
    ros::Subscriber odom_sub_;
    ros::Timer timer_depth_, timer_lidar_, timer_map_;

    ros::Time next_depth_pub_time, next_lidar_pub_time;
    ros::Duration depth_pub_duration, lidar_pub_duration;
    double depth_time{0.0}, lidar_time{0.0};
    int depth_count{0}, lidar_count{0};
    // mocka::Maps map;
};



// 相机内参: 直接取自 config 的 camera 段, 与实际渲染的深度图保证一致
void SensorSimulator::buildCameraInfo() {
    camera_info_.header.frame_id = CAMERA_FRAME;
    camera_info_.width = camera->image_width;
    camera_info_.height = camera->image_height;
    camera_info_.distortion_model = "plumb_bob";
    camera_info_.D.assign(5, 0.0);                       // 仿真无畸变
    camera_info_.K = {camera->fx, 0.0, camera->cx,       // [fx  0 cx]
                      0.0, camera->fy, camera->cy,       // [ 0 fy cy]
                      0.0, 0.0, 1.0};                    // [ 0  0  1]
    camera_info_.R = {1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0};
    camera_info_.P = {camera->fx, 0.0, camera->cx, 0.0,  // [fx  0 cx Tx]
                      0.0, camera->fy, camera->cy, 0.0,  // [ 0 fy cy Ty]
                      0.0, 0.0, 1.0, 0.0};               // [ 0  0  1  0]
}

// 静态 TF: 机体系 -> 相机光学系. 光学系相对机体系为 x_opt=-y_body, y_opt=-z_body, z_opt=x_body,
// 即四元数 (x,y,z,w)=(-0.5,0.5,-0.5,0.5); 再左乘 config 里的相机俯仰 quat_bc.
void SensorSimulator::publishStaticCameraTF() {
    const Eigen::Quaternionf q_cam2opt(0.5f, -0.5f, 0.5f, -0.5f);   // (w, x, y, z)
    Eigen::Quaternionf q_bo = (quat_bc * q_cam2opt).normalized();

    geometry_msgs::TransformStamped tf;
    tf.header.stamp = ros::Time::now();
    tf.header.frame_id = BODY_FRAME;
    tf.child_frame_id = CAMERA_FRAME;
    tf.transform.translation.x = t_bc.x();
    tf.transform.translation.y = t_bc.y();
    tf.transform.translation.z = t_bc.z();
    tf.transform.rotation.w = q_bo.w();
    tf.transform.rotation.x = q_bo.x();
    tf.transform.rotation.y = q_bo.y();
    tf.transform.rotation.z = q_bo.z();
    static_tf_broadcaster_.sendTransform(tf);
//    printf("Published static transform: %s -> %s\n", BODY_FRAME, CAMERA_FRAME);
}

void SensorSimulator::renderDepthCallback(const ros::Time stamp) {
    if (!render_depth && !render_stereo)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    // 左相机（= 原始相机位姿）
    cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(), pos.x(), pos.y(), pos.z());

    // 1) 深度真值（规划用，低分辨率 camera）
    if (render_depth) {
        cv::Mat depth_image;
        renderDepthImage(grid_map, camera, T_wc, depth_image);

        sensor_msgs::Image ros_image;
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = stamp;
        cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        cv_image.image = depth_image;
        cv_image.toImageMsg(ros_image);
        ros_image.header.frame_id = CAMERA_FRAME;
        image_pub_.publish(ros_image);

        camera_info_.header.stamp = stamp;
        camera_info_pub_.publish(camera_info_);
    }

    // 2) 伪双目深度（直接在 C++ 里用 simsense 算，不发布中间红外图）
    if (render_stereo) {
        // 右相机：沿相机“右”方向平移 baseline。本相机系里 +图像x 对应 -y_cam，
        // 故相机系偏移 = (0, -B, 0)，转到世界系叠加到左相机位置。
        Eigen::Vector3f off = quat_wc * Eigen::Vector3f(0.0f, -baseline, 0.0f);
        Eigen::Vector3f pos_r = pos + off;
        cudaMat::SE3<float> T_wc_r(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(), pos_r.x(), pos_r.y(), pos_r.z());

        cv::Mat stereo_depth;
        renderStereoImage(grid_map, stereo_camera, T_wc, T_wc_r, tex, stereo_engine, stereo_depth);

        sensor_msgs::Image ros_image;
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = stamp;
        cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        cv_image.image = stereo_depth;
        cv_image.toImageMsg(ros_image);
        ros_image.header.frame_id = CAMERA_FRAME;
        stereo_depth_pub_.publish(ros_image);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    depth_time += elapsed.count();
    depth_count++;
    // std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl;
}

void SensorSimulator::timerMapCallback(const ros::TimerEvent&) {
    if (pcl_pub.getNumSubscribers() > 0)
        pcl_pub.publish(output);    
}

void SensorSimulator::renderLidarCallback(const ros::Time stamp) {
    if (!render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time += elapsed.count();
    lidar_count++;
    // std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;
    quat_wc = quat * quat_bc;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    ros::Time tnow = ros::Time::now();

    // 避免仿真odom消息中断，导致时间差太大
    if (fabs((tnow - next_depth_pub_time).toSec()) > 10 * depth_pub_duration.toSec())
        next_depth_pub_time = tnow;
    if (fabs((tnow - next_lidar_pub_time).toSec()) > 10 * lidar_pub_duration.toSec())
        next_lidar_pub_time = tnow;

    if (tnow >= next_depth_pub_time){
        next_depth_pub_time += depth_pub_duration;
        renderDepthCallback(msg->header.stamp);
    }
    if (tnow >= next_lidar_pub_time){
        next_lidar_pub_time += lidar_pub_duration;
        renderLidarCallback(msg->header.stamp);
    }
    ros::Duration render_duration = ros::Time::now() - tnow;
    if (render_duration > depth_pub_duration || render_duration > lidar_pub_duration){
        // Performance reference: should take < 1 ms on 3060 GPU & Ubuntu 20.04
        ROS_WARN("Current Rendering time: %.2f ms, delay too much!", 1000 * render_duration.toSec());
        std::cout << "Average Depth Rendering time: " << (depth_time / (depth_count + 1e-8)) * 1000 << " ms" << std::endl;
        std::cout << "Average Lidar Rendering time: " << (lidar_time / (lidar_count + 1e-8)) * 1000 << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    SensorSimulator sensor_simulator(nh);
    return 0;
}
