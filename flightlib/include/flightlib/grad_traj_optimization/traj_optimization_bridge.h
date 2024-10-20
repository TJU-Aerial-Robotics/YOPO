#ifndef _TRAJ_OPTIMIZATION_BRIDGE_H_
#define _TRAJ_OPTIMIZATION_BRIDGE_H_

#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud.h>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>

#include "flightlib/grad_traj_optimization/grad_traj_optimizer.h"
#include "flightlib/grad_traj_optimization/opt_utile.h"

namespace traj_opt {
std::shared_ptr<sdf_tools::SignedDistanceField> SdfConstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::Vector3d &map_boundary_min_sdf,
                                                                Eigen::Vector3d &map_boundary_max_sdf);
}

class TrajOptimizationBridge {
   private:
	YAML::Node cfg_;
	// state of uav in world frame
	Eigen::Vector3d pos_;
	Eigen::Vector3d vel_;
	Eigen::Vector3d acc_;
	Eigen::Quaterniond q_wb_;
	Eigen::Vector3d goal_;
	Eigen::MatrixXd pred_coeff_;

	// std::string ply_file;
	double resolution;
	std::shared_ptr<sdf_tools::SignedDistanceField> sdf_;
	Eigen::Vector3d map_boundary_min_, map_boundary_max_;

	// x_pva, y_pva, z_pva in world frame
	std::vector<double> dp_;  // df refers to the initial_state currently
	std::vector<double> df_;  // dp refers to the end_state currently

	double goal_length;
	int horizon_num, vertical_num, radio_num, vel_num;
	double horizon_fov, vertical_fov, radio_range, vel_fov, vel_prefile;
	std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lattice_nodes;

	void loadParam(YAML::Node &cfg);

   public:
	TrajOptimizationBridge();
	~TrajOptimizationBridge();

	void setMap(std::shared_ptr<sdf_tools::SignedDistanceField> sdf_for_traj_optimization, Eigen::Vector3d &map_boundary_min,
	            Eigen::Vector3d &map_boundary_max);

	// in world frame
	void setState(Eigen::Vector3d pos, Eigen::Quaterniond q, Eigen::Vector3d vel, Eigen::Vector3d acc);

	void setGoal(Eigen::Vector3d goal);

	// dp in body frame, grad in body frame
	void getCostAndGradient(const std::vector<double> &dp, int id, double &cost, std::vector<double> &grad);

	void getNextStateAndCost(const std::vector<double> &dp, double &cost, Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Vector3d &acc,
	                         double sim_t);

	void solveBVP(const std::vector<double> &dp);

	void getNextState(Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Vector3d &acc, double sim_t);
};

#endif