#include "flightlib/envs/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv() : QuadrotorEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path) : EnvBase() {
	// load configuration file
	YAML::Node cfg_ = YAML::LoadFile(cfg_path);
	loadParam(cfg_);

	quadrotor_ptr_ = std::make_shared<Quadrotor>();

	// 1、define a bounding box (z is defined manually, different from spawn box)
	// x_min, x_max, y_min, y_max, z_min, z_max
	world_box_ << center_(0) - 0.5 * scale_(0), center_(0) + 0.5 * scale_(0), center_(1) - 0.5 * scale_(1), center_(1) + 0.5 * scale_(1), 0.0, 5.0;
	if (!quadrotor_ptr_->setWorldBox(world_box_)) {
		logger_.error("cannot set wolrd box");
	};

	// 2、define input and output dimension for the environment
	obs_dim_ = kNObs;
	act_dim_ = kNAct;
	rew_dim_ = 1;

	// 3、define planner
	traj_opt_bridge = new TrajOptimizationBridge();

	// 5、add camera
	sgm_.reset(new sgm_gpu::SgmGpu(width_, height_));

	if (!configCamera(cfg_)) {
		logger_.error("Cannot config RGB Camera. Something wrong with the config file");
	}

	Vector<3> quad_size(0.01, 0.01, 0.01);
	quadrotor_ptr_->setSize(quad_size);
	is_collision_ = false;
	steps_        = 0;
}

QuadrotorEnv::~QuadrotorEnv() { delete traj_opt_bridge; }

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
	quad_state_.setZero();
	quad_act_.setZero();
	is_collision_    = false;
	steps_           = 0;
	nearest_obstacle = 10;

	// Dagger Training
	if (random && !collect_data_) {
		// 1.reset position.
		do {
			is_collision_           = false;
			quad_state_.x(QS::POSX) = 0.40 * scale_(0) * uniform_dist_(random_gen_) + center_(0);
			quad_state_.x(QS::POSY) = 0.40 * scale_(1) * uniform_dist_(random_gen_) + center_(1);
			quad_state_.x(QS::POSZ) = 0.20 * scale_(2) * uniform_dist_(random_gen_) + center_(2);
			collisionCheck(1.5);
		} while (is_collision_);

		// 2.reset orientation
		float roll  = 0.0;
		float pitch = 0.0;
		float yaw   = 3.14159 * uniform_dist_(random_gen_);
		Eigen::Quaternionf q_;
		q_ = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
		     Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
		quad_state_.q(q_);
	}

	// Offline Data Collection
	if (collect_data_) {
		// 1.reset position.
		do {
			is_collision_           = false;
			quad_state_.x(QS::POSX) = 0.5 * scale_(0) * uniform_dist_(random_gen_) + center_(0);
			quad_state_.x(QS::POSY) = 0.5 * scale_(1) * uniform_dist_(random_gen_) + center_(1);
			quad_state_.x(QS::POSZ) = 0.5 * scale_(2) * uniform_dist_(random_gen_) + center_(2);
			collisionCheck(0.5);
		} while (is_collision_);

		// 2.reset orientation
		float roll  = (norm_dist_(random_gen_) * sqrt(roll_var_)) + 0.0;
		float pitch = (norm_dist_(random_gen_) * sqrt(pitch_var_)) + 0.0;
		float yaw   = 3.14159 * uniform_dist_(random_gen_);
		Eigen::Quaternionf q_;
		q_ = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
		     Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
		quad_state_.q(q_);
	}

	// reset quadrotor with random states
	quadrotor_ptr_->reset(quad_state_);
	// Currently, since there is no controller, the desired state is equal to the actual state.
	desired_p_ = Eigen::Vector3f(quad_state_.p(0), quad_state_.p(1), quad_state_.p(2));
	desired_v_ = Eigen::Vector3f(quad_state_.v(0), quad_state_.v(1), quad_state_.v(2));
	desired_a_ = Eigen::Vector3f(quad_state_.a(0), quad_state_.a(1), quad_state_.a(2));

	// obtain observations
	getObs(obs);
	return true;
}

void QuadrotorEnv::setState(ConstRef<Vector<>> state) {
	if (state.rows() != 13) {
		std::cout << "ERROR: state must be 13 dim (P_xyz, V_xyz, A_xyz, Q_wxyz)" << std::endl;
		return;
	}
	quad_state_.setZero();
	quad_state_.x(QS::POSX) = state(0);
	quad_state_.x(QS::POSY) = state(1);
	quad_state_.x(QS::POSZ) = state(2);
	quad_state_.x(QS::VELX) = state(3);
	quad_state_.x(QS::VELY) = state(4);
	quad_state_.x(QS::VELZ) = state(5);
	quad_state_.x(QS::ACCX) = state(6);
	quad_state_.x(QS::ACCY) = state(7);
	quad_state_.x(QS::ACCZ) = state(8);
	quad_state_.x(QS::ATTW) = state(9);
	quad_state_.x(QS::ATTX) = state(10);
	quad_state_.x(QS::ATTY) = state(11);
	quad_state_.x(QS::ATTZ) = state(12);
	quadrotor_ptr_->reset(quad_state_);
}

void QuadrotorEnv::setGoal(ConstRef<Vector<>> goal) {
	if (goal.rows() != 3) {
		std::cout << "ERROR: goal must be 3 dim (xyz)" << std::endl;
		return;
	}
	goal_ = goal;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
	// The actual position.
	Eigen::Vector3f Pw(quad_state_.p(0), quad_state_.p(1), quad_state_.p(2));
	Eigen::Quaternionf Qw = quad_state_.q();
	// The desired state, same with the real flight.
	Eigen::Matrix3f Rwb = quad_state_.R();
	Eigen::Vector3f Vw(desired_v_(0), desired_v_(1), desired_v_(2));
	Eigen::Vector3f Vb = Rwb.inverse() * Vw;
	Eigen::Vector3f Aw(desired_a_(0), desired_a_(1), desired_a_(2));
	Eigen::Vector3f Ab = Rwb.inverse() * Aw;

	// Observation: p, q_wxyz in the world frame (for training); v, a in the body frame (for testing).
	quad_obs_ << Pw(0), Pw(1), Pw(2), Vb(0), Vb(1), Vb(2), Ab(0), Ab(1), Ab(2), Qw.w(), Qw.x(), Qw.y(), Qw.z();
	obs.segment<kNObs>(0) = quad_obs_;
	return true;
}

bool QuadrotorEnv::getDepthImage(Ref<DepthImgVector<>> depth_img) {
	if (!rgb_camera_left || !rgb_camera_left->getEnabledLayers()[1]) {
		logger_.error("No RGB Camera or depth map is not enabled. Cannot retrieve depth images.");
		return false;
	}
	rgb_camera_left->getDepthMap(depth_img_);

	depth_img = Map<DepthImgVector<>>((float_t *)depth_img_.data, depth_img_.rows * depth_img_.cols);
	return true;
}

bool QuadrotorEnv::getStereoImage(Ref<DepthImgVector<>> stereo_img) {
	if (!rgb_camera_left || !rgb_camera_right) {
		logger_.error("No Stereo Camera enabled. Cannot retrieve depth map.");
		return false;
	}
	cv::Mat left_img, right_img;
	rgb_camera_left->getRGBImage(left_img);
	rgb_camera_right->getRGBImage(right_img);

	// compute disparity image
	cv::Mat stereo_(height_, width_, CV_32FC1);
	computeDepthImage(left_img, right_img, &stereo_);

	// fix the nan of stereo by gt depth (make it closer to RealSense 435, as the depth from 435 is better than from SGM directly)
	if (rgb_camera_left->getEnabledLayers()[1]) {
		if (rgb_camera_left->getDepthMap(depth_img_)) {
			cv::Mat mask, mask1, mask2;
			cv::compare(stereo_, 0, mask1, cv::CMP_EQ);   // 将 A 中为 0 的位置置为 255，其余位置置为 0
			cv::compare(stereo_, 20, mask2, cv::CMP_GT);  // 将 A 中大于 20 的位置置为 255，其余位置置为 0
			mask = mask1 | mask2;                         // 将两个掩码进行逻辑或操作
			depth_img_.copyTo(stereo_, mask);             // 将 B 中 mask 为 255 的位置的值复制到 A 中
		}
	}

	stereo_img = Map<DepthImgVector<>>((float_t *)stereo_.data, stereo_.rows * stereo_.cols);
	return true;
}

void QuadrotorEnv::setMapID(int id) {
	// -1 represent using the latest map in imitation learning
	if (id < 0)
		map_idx_ = kdtrees.size() - 1;
	else
		map_idx_ = id;
}

void QuadrotorEnv::getCostAndGradient(ConstRef<Vector<>> endstate, int traj_id, float &cost, Ref<Vector<>> grad) {
	if (endstate.rows() != 9) {
		std::cout << "ERROR: endstate must be 9 dim (X_pva, Y_pva, Z_pva)" << std::endl;
		return;
	}
	std::vector<double> endstate_, grad_;
	double cost_;
	for (size_t i = 0; i < endstate.rows(); i++) {
		endstate_.push_back(static_cast<double>(endstate(i)));
	}

	traj_opt_bridge->setMap(esdf_maps[map_idx_], min_map_boundaries[map_idx_], max_map_boundaries[map_idx_]);
	traj_opt_bridge->setState(quad_state_.p.cast<double>(), quad_state_.q().cast<double>(), quad_state_.v.cast<double>(),
	                          quad_state_.a.cast<double>());
	traj_opt_bridge->setGoal(goal_.cast<double>());

	traj_opt_bridge->getCostAndGradient(endstate_, traj_id, cost_, grad_);

	for (size_t i = 0; i < grad_.size(); i++) {
		grad(i) = static_cast<float>(grad_[i]);
	}
	cost = static_cast<float>(cost_);
}

bool QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs, Ref<Vector<>> reward) {
	// python：setGoal -> step
	if (!act.allFinite() || act.rows() != 9 || reward.rows() != 1) {
		std::cout << "ERROR: endstate must be 9 dim" << std::endl;
		return false;
	}

	steps_++;

	std::vector<double> endstate_;
	for (size_t i = 0; i < act.rows(); i++) {
		endstate_.push_back(static_cast<double>(act(i)));
	}

	traj_opt_bridge->setMap(esdf_maps[map_idx_], min_map_boundaries[map_idx_], max_map_boundaries[map_idx_]);
	traj_opt_bridge->setState(quad_state_.p.cast<double>(), quad_state_.q().cast<double>(), quad_state_.v.cast<double>(),
	                          quad_state_.a.cast<double>());
	traj_opt_bridge->setGoal(goal_.cast<double>());

	double cost_;
	Eigen::Vector3d next_pos, next_vel, next_acc;
	traj_opt_bridge->getNextStateAndCost(endstate_, cost_, next_pos, next_vel, next_acc, sim_dt_);
	desired_p_ = next_pos.cast<float>();
	desired_v_ = next_vel.cast<float>();
	desired_a_ = next_acc.cast<float>();
	reward(0)  = static_cast<float>(cost_);

	// calculate the state based on the desired state.
	runControlAndUpdateState(desired_p_, desired_v_, desired_a_);
	quadrotor_ptr_->getState(&quad_state_);

	getObs(obs);
	return true;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
	// 1.if out of boundary
	if (quad_state_.x(QS::POSX) <= (world_box_(0)) || quad_state_.x(QS::POSY) <= (world_box_(1)) || quad_state_.x(QS::POSZ) <= (world_box_(2)) ||
	    quad_state_.x(QS::POSX) >= (world_box_(3)) || quad_state_.x(QS::POSY) >= (world_box_(4)) || quad_state_.x(QS::POSZ) >= (world_box_(5))) {
		reward = 0;
		// std::cout<<"越界"<<std::endl;
		return true;
	}

	// 2.if collision
	if (is_collision_) {
		reward = -1;
		// std::cout<<"碰撞"<<std::endl;
		return true;
	}

	// 3.prevent uav being trapped
	if (steps_ > 10 && quad_state_.v.norm() < 0.6 && dagger_mode_) {
		reward = 0;
		return true;
	}

	// 4.if reach the goal
	Eigen::Vector3f Pw(quad_state_.p(0), quad_state_.p(1), quad_state_.p(2));
	Eigen::Vector3f Gw(goal_(0), goal_(1), goal_(2));
	float dist = (Pw - Gw).norm();
	if (dist < 4) {
		reward = 0;
		// std::cout<<"到达"<<std::endl;
		return true;
	}

	reward = 0.0;
	return false;
}

void QuadrotorEnv::collisionCheck(float dis) {
	pcl::PointXYZ drone_;
	drone_.x = quad_state_.x(QS::POSX);
	drone_.y = quad_state_.x(QS::POSY);
	drone_.z = quad_state_.x(QS::POSZ);

	int K = 1;
	std::vector<int> indices(K);
	std::vector<float> distances(K);  // the square of the distances to the neighboring points.
	kdtrees[map_idx_]->nearestKSearch(drone_, K, indices, distances);

	nearest_obstacle = distances[0];
	if (distances[0] < dis * dis)
		is_collision_ = true;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
	// camera
	width_  = cfg["rgb_camera_left"]["width"].as<int>();
	height_ = cfg["rgb_camera_left"]["height"].as<int>();
	fov_    = cfg["rgb_camera_left"]["fov"].as<Scalar>();
	// train or test
	collect_data_ = cfg["quadrotor_env"]["collect_data"].as<bool>();
	sim_dt_       = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
	// data_collection
	roll_var_  = cfg["data_collection"]["roll_var"].as<Scalar>();
	pitch_var_ = cfg["data_collection"]["pitch_var"].as<Scalar>();
	// world box
	for (int i = 0; i < 3; ++i) {
		scale_(i)  = cfg["quadrotor_env"]["bounding_box"][i].as<Scalar>();
		center_(i) = cfg["quadrotor_env"]["bounding_box_origin"][i].as<Scalar>();
	}
	return true;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
	if (quad_act_.allFinite()) {
		act = quad_act_;
		return true;
	}
	return false;
}

bool QuadrotorEnv::configCamera(const YAML::Node &cfg) {
	if (!cfg["rgb_camera_left"]) {
		logger_.error("Cannot config RGB Camera");
		return false;
	}

	if (!cfg["rgb_camera_left"]["on"].as<bool>()) {
		logger_.warn("Camera is off. Please turn it on.");
		return false;
	}

	if (quadrotor_ptr_->getNumCamera() >= 2) {
		logger_.warn("Camera has been added. Skipping the camera configuration.");
		return false;
	}

	// create left camera --------------------------------------------
	rgb_camera_left = std::make_shared<RGBCamera>();

	// load camera settings
	std::vector<Scalar> t_BC_vec = cfg["rgb_camera_left"]["t_BC"].as<std::vector<Scalar>>();
	std::vector<Scalar> r_BC_vec = cfg["rgb_camera_left"]["r_BC"].as<std::vector<Scalar>>();

	Vector<3> t_BC(t_BC_vec.data());
	Matrix<3, 3> r_BC = (AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ()) * AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
	                     AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()))
	                        .toRotationMatrix();

	// Flightmare's FOV is vertical, while usually is horizontal, so convert the input horizontal FOV to vertical FOV.
	Scalar rgb_fov_deg_    = cfg["rgb_camera_left"]["fov"].as<Scalar>();
	double hor_fov_radians = (M_PI * (rgb_fov_deg_ / 180.0));
	Scalar img_rows_       = cfg["rgb_camera_left"]["height"].as<Scalar>();
	Scalar img_cols_       = cfg["rgb_camera_left"]["width"].as<Scalar>();
	double flightmare_fov  = 2. * std::atan(std::tan(hor_fov_radians / 2) * img_rows_ / img_cols_);
	flightmare_fov         = (flightmare_fov / M_PI) * 180.0;
	rgb_camera_left->setFOV(flightmare_fov);
	rgb_camera_left->setWidth(cfg["rgb_camera_left"]["width"].as<int>());
	rgb_camera_left->setHeight(cfg["rgb_camera_left"]["height"].as<int>());
	rgb_camera_left->setRelPose(t_BC, r_BC);
	rgb_camera_left->enableOpticalFlow(cfg["rgb_camera_left"]["enable_opticalflow"].as<bool>());
	rgb_camera_left->enableSegmentation(cfg["rgb_camera_left"]["enable_segmentation"].as<bool>());
	rgb_camera_left->enableDepth(cfg["rgb_camera_left"]["enable_depth"].as<bool>());

	// add camera to the quadrotor
	quadrotor_ptr_->addRGBCamera(rgb_camera_left);

	// create right camera --------------------------------------------
	bool have_right_camera = cfg["rgb_camera_right"]["on"].as<bool>();
	if (have_right_camera) {
		rgb_camera_right = std::make_shared<RGBCamera>();

		// load camera settings
		std::vector<Scalar> t_BC_vec_r = cfg["rgb_camera_right"]["t_BC"].as<std::vector<Scalar>>();
		std::vector<Scalar> r_BC_vec_r = cfg["rgb_camera_right"]["r_BC"].as<std::vector<Scalar>>();

		Vector<3> t_BC_r(t_BC_vec_r.data());
		Matrix<3, 3> r_BC_r =
		    (AngleAxis(r_BC_vec_r[2] * M_PI / 180.0, Vector<3>::UnitZ()) * AngleAxis(r_BC_vec_r[1] * M_PI / 180.0, Vector<3>::UnitY()) *
		     AngleAxis(r_BC_vec_r[0] * M_PI / 180.0, Vector<3>::UnitX()))
		        .toRotationMatrix();

		rgb_camera_right->setFOV(flightmare_fov);
		rgb_camera_right->setWidth(cfg["rgb_camera_left"]["width"].as<int>());
		rgb_camera_right->setHeight(cfg["rgb_camera_left"]["height"].as<int>());
		rgb_camera_right->setRelPose(t_BC_r, r_BC_r);
		rgb_camera_right->enableOpticalFlow(false);
		rgb_camera_right->enableSegmentation(false);
		rgb_camera_right->enableDepth(false);

		// add camera to the quadrotor
		quadrotor_ptr_->addRGBCamera(rgb_camera_right);
		stereo_baseline_ = fabs(t_BC(1) - t_BC_r(1));
	}
	// adapt parameters
	img_width_  = rgb_camera_left->getWidth();
	img_height_ = rgb_camera_left->getHeight();
	rgb_img_    = cv::Mat::zeros(img_height_, img_width_, CV_MAKETYPE(CV_8U, rgb_camera_left->getChannels()));
	depth_img_  = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
	return true;
}

void QuadrotorEnv::computeDepthImage(const cv::Mat &left_frame, const cv::Mat &right_frame, cv::Mat *const depth) {
	cv::Mat disparity(height_, width_, CV_8UC1);
	sgm_->computeDisparity(left_frame, right_frame, disparity);
	disparity.convertTo(disparity, CV_32FC1);

	// compute depth from disparity
	float f = (width_ / 2.0) / std::tan((M_PI * (fov_ / 180.0)) / 2.0);
	//  depth = stereo_baseline_ * f / disparity
	for (int r = 0; r < height_; ++r) {
		for (int c = 0; c < width_; ++c) {
			if (disparity.at<float>(r, c) == 0.0f) {
				depth->at<float>(r, c) = 0.0f;
			} else if (disparity.at<float>(r, c) == 255.0f) {
				depth->at<float>(r, c) = 0.0f;
			} else {
				depth->at<float>(r, c) = static_cast<float>(stereo_baseline_) * f / disparity.at<float>(r, c);
			}
		}
	}
}

bool QuadrotorEnv::getRGBImage(Ref<ImgVector<>> img, const bool rgb) {
	if (!rgb_camera_left) {
		logger_.error("No Camera! Cannot retrieve Images.");
		return false;
	}

	rgb_camera_left->getRGBImage(rgb_img_);

	if (rgb_img_.rows != height_ || rgb_img_.cols != width_) {
		logger_.error("Image resolution mismatch. Aborting.. Image rows %d != %d, Image cols %d != %d", rgb_img_.rows, height_, rgb_img_.cols,
		              width_);
		return false;
	}

	if (!rgb) {
		// converting rgb image to gray image
		cvtColor(rgb_img_, gray_img_, CV_RGB2GRAY);
		// map cv::Mat data to Eiegn::Vector
		img = Map<ImgVector<>>(gray_img_.data, gray_img_.rows * gray_img_.cols);
	} else {
		img = Map<ImgVector<>>(rgb_img_.data, rgb_img_.rows * rgb_img_.cols * rgb_camera_left->getChannels());
	}
	return true;
}

void QuadrotorEnv::runControlAndUpdateState(Eigen::Vector3f p_ref, Eigen::Vector3f v_ref, Eigen::Vector3f a_ref) {
	Eigen::Vector3f p_cur;
	p_cur = quad_state_.p;

	Eigen::Vector3f dir_vel  = v_ref;
	Eigen::Vector3f dir_goal = goal_ - p_cur;
	Eigen::Vector3f dir_des  = dir_vel.normalized() + dir_goal.normalized();
	float yaw_ref            = atan2(dir_des(1), dir_des(0));
	Vector<3> rpy_cur;
	get_euler_from_R(rpy_cur, quad_state_.R());
	yaw_ref = calculate_yaw(rpy_cur(2), yaw_ref, sim_dt_);  // limit the yaw (required by controller)

	Eigen::Quaternionf q_ref;
	quadrotor_ptr_->runSimpleFlight(a_ref, yaw_ref, q_ref);
	quadrotor_ptr_->setState(p_ref, v_ref, q_ref, a_ref, sim_dt_);
}

}  // namespace flightlib