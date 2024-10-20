
#pragma once

#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <memory>

#include "std_msgs/Empty.h"
// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"
#include "flightlib/sensors/sgm_gpu/sgm_gpu.h"

using namespace flightlib;

namespace flightros {

class FlightPilot {
   public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	FlightPilot(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
	~FlightPilot();

	// callbacks
	void mainLoopCallback(const ros::TimerEvent& event);
	void spawnTreeCallback(const std_msgs::Empty::ConstPtr& msg);
	void clearTreeCallback(const std_msgs::Empty::ConstPtr& msg);
	void savePointcloudCallback(const std_msgs::Empty::ConstPtr& msg);
	void poseCallback(const nav_msgs::Odometry::ConstPtr& msg);

	// unity
	bool setUnity(const bool render);
	bool connectUnity(void);
	bool disconnectUnity();
	bool loadParams(const YAML::Node& cfg);
	bool configCamera(const YAML::Node& cfg);
	void computeDepthImage(const cv::Mat& left_frame, const cv::Mat& right_frame, cv::Mat* const depth);
	bool spawnTreesAndSavePointCloud();

   private:
	// ros nodes
	ros::NodeHandle nh_;
	ros::NodeHandle pnh_;

	// publisher & subscriber
	std::string odom_topic_;
	ros::Publisher stereo_pub, left_img_pub, depth_pub, cam_info_pub;
	ros::Subscriber state_est_sub_, spawn_tree_sub_, clear_tree_sub_, save_pc_sub_;

	// main loop timer
	ros::Timer timer_main_loop_;
	ros::Time timestamp;
	Scalar main_loop_freq_{50.0};

	// unity & quadrotor
	Vector<3> quad_size_;
	QuadState quad_state_;
	std::shared_ptr<Quadrotor> quad_ptr_;
	std::shared_ptr<UnityBridge> unity_bridge_ptr_;
	SceneID scene_id_{UnityScene::WAREHOUSE};
	bool unity_ready_{false};
	bool unity_render_{false};
	RenderMessage_t unity_output_;
	uint16_t receive_id_{0};
	FrameID frameID{0};

	// camera param
	Scalar stereo_baseline_;
	Scalar fov_;
	int width_;
	int height_;
	bool use_depth, use_stereo;
	std::shared_ptr<RGBCamera> rgb_camera_left, rgb_camera_right;
	std::shared_ptr<sgm_gpu::SgmGpu> sgm_;

	// tree generation
	int ply_id_{0};
	Scalar avg_tree_spacing_;
	Vector<3> bounding_box_, bounding_box_origin_;
	Scalar pointcloud_resolution_;
	std::string ply_path_;
};
}  // namespace flightros