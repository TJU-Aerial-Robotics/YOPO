#pragma once

// std lib
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <iostream>
// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/math.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/grad_traj_optimization/traj_optimization_bridge.h"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/sgm_gpu/sgm_gpu.h"

namespace flightlib {

enum Ctl : int {
	kNObs = 13,  // observation dim
	kNAct = 9,   // action dim
};

class QuadrotorEnv final : public EnvBase {
   public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	QuadrotorEnv();
	QuadrotorEnv(const std::string &cfg_path);
	~QuadrotorEnv();

	/* set functions */
	bool reset(Ref<Vector<>> obs, const bool random = true) override;
	void setState(ConstRef<Vector<>> state);
	void setGoal(ConstRef<Vector<>> goal);
	void setMapID(int id);
	void setDAggerMode(bool dagger_mode) { dagger_mode_ = dagger_mode; }
	bool step(const Ref<Vector<>> act, Ref<Vector<>> obs, Ref<Vector<>> reward) override;
	void setInputCloud(const pcl::PointCloud<pcl::PointXYZ> &point_in);
	void setESDF(const std::vector<float> &esdf_map, const pcl::PointCloud<pcl::PointXYZ> &point_in);
	void addKdtree(std::shared_ptr<pcl::search::KdTree<pcl::PointXYZ>> kdtree) { kdtrees.push_back(kdtree); }
	void addESDFMap(std::shared_ptr<sdf_tools::SignedDistanceField> esdf_map) { esdf_maps.push_back(esdf_map); }
	void addMapSize(Eigen::Vector3d map_boundary_min, Eigen::Vector3d map_boundary_max) {
		min_map_boundaries.push_back(map_boundary_min);
		max_map_boundaries.push_back(map_boundary_max);
	}

	/* get functions */
	bool getObs(Ref<Vector<>> obs) override;
	bool getAct(Ref<Vector<>> act) const;
	bool getRGBImage(Ref<ImgVector<>> img, const bool rgb);
	bool getDepthImage(Ref<DepthImgVector<>> img) override;
	bool getStereoImage(Ref<DepthImgVector<>> img);
	int getMapNum() { return kdtrees.size(); }
	void getCostAndGradient(ConstRef<Vector<>> dp, int id, float &cost, Ref<Vector<>> grad);
	inline std::vector<std::string> getRewardNames() { return reward_names_; }
	void getWorldBox(Ref<Vector<>> world_box) {
		world_box << world_box_(0), world_box_(1), world_box_(2), world_box_(3), world_box_(4), world_box_(5);  // xyz_min, xyz_max
	};

	/* other functions */
	void collisionCheck(float dis = 0.2);
	bool configCamera(const YAML::Node &cfg_node);
	bool loadParam(const YAML::Node &cfg);
	void computeDepthImage(const cv::Mat &left_frame, const cv::Mat &right_frame, cv::Mat *const depth);
	bool isTerminalState(Scalar &reward) override;
	void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) { bridge->addQuadrotor(quadrotor_ptr_); }
	void runControlAndUpdateState(Eigen::Vector3f p_ref, Eigen::Vector3f v_ref, Eigen::Vector3f a_ref);

   private:
	// quadrotor state and observation
	std::shared_ptr<Quadrotor> quadrotor_ptr_;
	QuadState quad_state_;
	Vector<kNObs> quad_obs_;
	Vector<kNAct> quad_act_;
	Logger logger_{"QaudrotorEnv"};
	Eigen::Vector3f desired_p_, desired_v_, desired_a_;

	// map
	Matrix<3, 2> world_box_;
	Vector<3> center_, scale_;
	std::vector<std::shared_ptr<pcl::search::KdTree<pcl::PointXYZ>>> kdtrees;
	std::vector<std::shared_ptr<sdf_tools::SignedDistanceField>> esdf_maps;
	std::vector<Eigen::Vector3d> min_map_boundaries, max_map_boundaries;

	// camera params
	Scalar fov_;
	int width_, height_;
	Scalar stereo_baseline_;
	std::shared_ptr<sgm_gpu::SgmGpu> sgm_;
	cv::Mat rgb_img_, gray_img_, depth_img_;
	std::shared_ptr<RGBCamera> rgb_camera_left, rgb_camera_right;

	// trajectory optimization
	int map_idx_{0};
	Vector<3> goal_;
	TrajOptimizationBridge *traj_opt_bridge;

	// data collection
	Scalar roll_var_, pitch_var_;

	// others
	int steps_;
	YAML::Node cfg_;
	bool is_collision_;
	Scalar nearest_obstacle{10};
	std::vector<std::string> reward_names_;
	bool collect_data_, dagger_mode_{false};
};

}  // namespace flightlib