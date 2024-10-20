#include "flightlib/grad_traj_optimization/traj_optimization_bridge.h"

namespace traj_opt {

std::shared_ptr<sdf_tools::SignedDistanceField> SdfConstruction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::Vector3d &map_boundary_min_sdf,
                                                                Eigen::Vector3d &map_boundary_max_sdf) {
	pcl::PointXYZ min, max;
	pcl::getMinMax3D(*cloud, min, max);

	// sdf collision map parameter
	const double x_size = max.x - min.x;
	const double z_size = max.z - min.z;
	const double y_size = max.y - min.y;
	Eigen::Translation3d origin_translation(min.x, min.y, min.z);
	Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
	const Eigen::Isometry3d origin_transform = origin_translation * origin_rotation;
	const std ::string frame                 = "world";
	map_boundary_min_sdf                     = Eigen::Vector3d(min.x, min.y, min.z);
	map_boundary_max_sdf                     = Eigen::Vector3d(max.x, max.y, max.z);

	// create map
	sdf_tools ::COLLISION_CELL cell;
	cell.occupancy                           = 0.0;
	cell.component                           = 0;
	const sdf_tools::COLLISION_CELL oob_cell = cell;
	double resolution_sdf                    = 0.2;
	sdf_tools::CollisionMapGrid collision_map(origin_transform, frame, resolution_sdf, x_size, y_size, z_size, oob_cell);

	// add obstacles set in launch file
	std::cout << "Generate map..." << std::endl;
	sdf_tools::COLLISION_CELL obstacle_cell(1.0);

	// add the generated obstacles into collision map （flightmare点云直接建图行列偶尔有全空的情况, 但不影响）
	// 点云分辨率改为0.1地图分辨率0.2，就不存在空行的问题; 地图分辨率0.2时，用pt.x + 0.001, pt.y + 0.001, pt.z + 0.001可避免空行且不会造成地图偏移
	for (int i = 0; i < cloud->points.size(); i++) {
		pcl::PointXYZ pt = cloud->points[i];
		collision_map.Set(pt.x, pt.y, pt.z, obstacle_cell);
	}

	// Build the signed distance field
	float oob_value                                                                       = INFINITY;
	std::pair<sdf_tools::SignedDistanceField, std::pair<double, double>> sdf_with_extrema = collision_map.ExtractSignedDistanceField(oob_value);

	sdf_tools::SignedDistanceField sdf_for_traj_optimization = sdf_with_extrema.first;
	cout << "Signed Distance Field Build!" << endl;
	return std::make_shared<sdf_tools::SignedDistanceField>(sdf_for_traj_optimization);
}

}  // namespace traj_opt

TrajOptimizationBridge::TrajOptimizationBridge() {
	std::string cfg_path = getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/traj_opt.yaml");
	cfg_                 = YAML::LoadFile(cfg_path);
	loadParam(cfg_);

	resolution = 0.2;  // Must be the same as the map in SdfConstruction()
	df_.resize(9);
	dp_.resize(9);

	getLatticeGuiding(lattice_nodes, horizon_num, vertical_num, radio_num, vel_num, horizon_fov, vertical_fov, radio_range, vel_fov, vel_prefile);
}

TrajOptimizationBridge::~TrajOptimizationBridge() {}

// Explanation: In grad_traj_optimization of the current project, dp refers to the end_state, and df refers to the initial_state
void TrajOptimizationBridge::getCostAndGradient(const std::vector<double> &dp, int id, double &cost, std::vector<double> &grad) {
	// dp: the predicted X_pva, Y_pva, Z_pva in Body Frame
	if (dp_.size() != dp.size()) {
		std::cout << "Error: size of dp dose not match !" << std::endl;
		return;
	}
	std::vector<double> dp_b = dp;
	// Transform to world frame.
	Eigen::Vector3d Pb, Vb, Ab, Pw, Vw, Aw;
	for (int i = 0; i < 3; i++) {
		Pb(i) = dp_b[3 * i];
		Vb(i) = dp_b[3 * i + 1];
		Ab(i) = dp_b[3 * i + 2];
	}
	Eigen::Matrix3d Rwb = q_wb_.toRotationMatrix();
	Pw                  = Rwb * Pb + pos_;
	Vw                  = Rwb * Vb;
	Aw                  = Rwb * Ab;
	for (int i = 0; i < 3; i++) {
		dp_[3 * i]     = Pw(i);
		dp_[3 * i + 1] = Vw(i);
		dp_[3 * i + 2] = Aw(i);
	}

	// ----------------------------main optimization procedure--------------------------
	GradTrajOptimizer grad_traj_opt(cfg_);
	grad_traj_opt.setSignedDistanceField(sdf_, resolution);
	grad_traj_opt.setBoundary(map_boundary_min_, map_boundary_max_);
	grad_traj_opt.setGoal(goal_);

	double cost_;
	std::vector<double> grad_w, grad_b;
	grad_traj_opt.getCostAndGradient(df_, dp_, cost_, grad_w);

	Eigen::Vector3d grad_pb, grad_vb, grad_ab, grad_pw, grad_vw, grad_aw;
	for (int i = 0; i < 3; i++) {
		grad_pw(i) = grad_w[3 * i];
		grad_vw(i) = grad_w[3 * i + 1];
		grad_aw(i) = grad_w[3 * i + 2];
	}
	grad_pb = Rwb.transpose() * grad_pw;
	grad_vb = Rwb.transpose() * grad_vw;
	grad_ab = Rwb.transpose() * grad_aw;
	grad_b.resize(grad_w.size());
	for (int i = 0; i < 3; i++) {
		grad_b[3 * i]     = grad_pb(i);
		grad_b[3 * i + 1] = grad_vb(i);
		grad_b[3 * i + 2] = grad_ab(i);
	}

	cost = cost_;
	grad = grad_b;  // x_pva, y_pva, z_pva
}

void TrajOptimizationBridge::setMap(std::shared_ptr<sdf_tools::SignedDistanceField> sdf_for_traj_optimization, Eigen::Vector3d &map_boundary_min,
                                    Eigen::Vector3d &map_boundary_max) {
	map_boundary_min_ = map_boundary_min;
	map_boundary_max_ = map_boundary_max;
	sdf_              = sdf_for_traj_optimization;
}

void TrajOptimizationBridge::setState(Eigen::Vector3d pos, Eigen::Quaterniond q, Eigen::Vector3d vel, Eigen::Vector3d acc) {
	pos_  = pos;
	q_wb_ = q;
	vel_  = vel;
	acc_  = acc;
	for (int i = 0; i < 3; i++) {
		df_[3 * i]     = pos(i);
		df_[3 * i + 1] = vel(i);
		df_[3 * i + 2] = acc(i);
	}
}

// Explanation: In grad_traj_optimization of the current project, dp refers to the end_state, and df refers to the initial_state
void TrajOptimizationBridge::getNextStateAndCost(const std::vector<double> &dp, double &cost, Eigen::Vector3d &pos, Eigen::Vector3d &vel,
                                                 Eigen::Vector3d &acc, double sim_t) {
	// dp: xyz_pva (in Body Frame)
	if (dp_.size() != dp.size()) {
		std::cout << "Error: size of dp dose not match !" << std::endl;
		return;
	}
	std::vector<double> dp_b = dp;
	Eigen::Vector3d Pb, Vb, Ab, Pw, Vw, Aw;
	for (int i = 0; i < 3; i++) {
		Pb(i) = dp_b[3 * i];
		Vb(i) = dp_b[3 * i + 1];
		Ab(i) = dp_b[3 * i + 2];
	}
	Eigen::Matrix3d Rwb = q_wb_.toRotationMatrix();
	Pw                  = Rwb * Pb + pos_;
	Vw                  = Rwb * Vb;
	Aw                  = Rwb * Ab;
	for (int i = 0; i < 3; i++) {
		dp_[3 * i]     = Pw(i);
		dp_[3 * i + 1] = Vw(i);
		dp_[3 * i + 2] = Aw(i);
	}

	GradTrajOptimizer grad_traj_opt(cfg_);
	grad_traj_opt.setSignedDistanceField(sdf_, resolution);
	grad_traj_opt.setBoundary(map_boundary_min_, map_boundary_max_);
	grad_traj_opt.setGoal(goal_);

	double cost_;
	std::vector<double> grad_w;
	grad_traj_opt.getCostAndGradient(df_, dp_, cost_, grad_w);     // Df is set here
	grad_traj_opt.getCoefficientFromDerivative(pred_coeff_, dp_);  // get coefficient by Dp and Df

	cost = cost_;
	getPositionFromCoeff(pos, pred_coeff_, 0, sim_t);
	getVelocityFromCoeff(vel, pred_coeff_, 0, sim_t);
	getAccelerationFromCoeff(acc, pred_coeff_, 0, sim_t);
}

/**
    set dp and get the coeffs for getNextState() function.
    dp is in the world frame because this func only used in real flight and
    the prediction must be tramsformed to world frame in python to avoid the attitude inconsistency caused by latency
*/
void TrajOptimizationBridge::solveBVP(const std::vector<double> &dp) {
	// dp: xyz_pva given by python（in World Frame, 除了位置没加机身的偏移）
	if (dp_.size() != dp.size()) {
		std::cout << "Error: size of dp dose not match !" << std::endl;
		return;
	}

	std::vector<double> dp_w = dp;
	Eigen::Vector3d Pb, Vb, Ab, Pw, Vw, Aw;
	for (int i = 0; i < 3; i++) {
		Pw(i) = dp_w[3 * i];
		Vw(i) = dp_w[3 * i + 1];
		Aw(i) = dp_w[3 * i + 2];
	}

	Pw = Pw + pos_;
	// Eigen::Matrix3d Rwb = q_wb_.toRotationMatrix();
	// Pw = Rwb * Pb + pos_;
	// Vw = Rwb * Vb;
	// Aw = Rwb * Ab;

	double traj_time = 2 * cfg_["radio_range"].as<double>() / cfg_["vel_max"].as<double>();
	pred_coeff_      = solveCoeffFromBoundaryState(pos_, vel_, acc_, Pw, Vw, Aw, traj_time);
}

// get the next state in world frame
void TrajOptimizationBridge::getNextState(Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Vector3d &acc, double sim_t) {
	getPositionFromCoeff(pos, pred_coeff_, 0, sim_t);
	getVelocityFromCoeff(vel, pred_coeff_, 0, sim_t);
	getAccelerationFromCoeff(acc, pred_coeff_, 0, sim_t);
}

void TrajOptimizationBridge::setGoal(Eigen::Vector3d goal) {
	Eigen::Vector3d goal_dir  = (goal - pos_) / (goal - pos_).norm();
	Eigen::Vector3d temp_goal = goal;
	temp_goal                 = pos_ + goal_length * goal_dir;
	goal_                     = temp_goal;
}

void TrajOptimizationBridge::loadParam(YAML::Node &cfg) {
	horizon_num  = cfg["horizon_num"].as<int>();
	vertical_num = cfg["vertical_num"].as<int>();
	vel_num      = cfg["vel_num"].as<int>();
	radio_num    = cfg["radio_num"].as<int>();
	horizon_fov  = cfg["horizon_camera_fov"].as<double>() * (horizon_num - 1) / horizon_num;
	vertical_fov = cfg["vertical_camera_fov"].as<double>() * (vertical_num - 1) / vertical_num;
	vel_fov      = cfg["vel_fov"].as<double>();
	radio_range  = cfg["radio_range"].as<double>();
	vel_prefile  = cfg["vel_prefile"].as<double>();
	goal_length  = cfg["goal_length"].as<double>();
}