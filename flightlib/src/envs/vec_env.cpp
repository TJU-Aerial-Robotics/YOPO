#include "flightlib/envs/vec_env.hpp"

namespace flightlib {

template<typename EnvBase>
VecEnv<EnvBase>::VecEnv() : VecEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/vec_env.yaml")) {}

template<typename EnvBase>
VecEnv<EnvBase>::VecEnv(const YAML::Node& cfg_node) : cfg_(cfg_node) {
	init();
}

template<typename EnvBase>
VecEnv<EnvBase>::VecEnv(const std::string& cfgs, const bool from_file) {
	// load environment configuration
	if (from_file)
		cfg_ = YAML::LoadFile(cfgs);
	else
		cfg_ = YAML::Load(cfgs);

	init();
}

template<typename EnvBase>
void VecEnv<EnvBase>::init(void) {
	// note that the cfg are input from python, and many have changed from vec_end.yaml
	unity_render_          = cfg_["env"]["render"].as<bool>();
	supervised_mode_       = cfg_["env"]["supervised"].as<bool>();
	dagger_mode_           = cfg_["env"]["imitation"].as<bool>();
	seed_                  = cfg_["env"]["seed"].as<int>();
	num_envs_              = cfg_["env"]["num_envs"].as<int>();
	num_threads_           = cfg_["env"]["num_threads"].as<int>();
	scene_id_              = cfg_["env"]["scene_id"].as<SceneID>();
	ply_path_              = getenv("FLIGHTMARE_PATH") + cfg_["env"]["ply_path"].as<std::string>();
	avg_tree_spacing_      = cfg_["unity"]["avg_tree_spacing"].as<Scalar>();
	pointcloud_resolution_ = cfg_["unity"]["pointcloud_resolution"].as<Scalar>();
	for (int i = 0; i < 3; ++i) {
		bounding_box_(i)        = cfg_["unity"]["bounding_box"][i].as<Scalar>();
		bounding_box_origin_(i) = cfg_["unity"]["bounding_box_origin"][i].as<Scalar>();
	}

	// set threads
	omp_set_num_threads(cfg_["env"]["num_threads"].as<int>());

	// create & setup environments
	const bool render = false;
	for (int i = 0; i < num_envs_; i++) {
		envs_.push_back(std::make_unique<EnvBase>());
	}

	// set Unity (init unity_bridge_ptr_ and add quadrotors to envs)
	setUnity(unity_render_);

	obs_dim_    = envs_[0]->getObsDim();
	act_dim_    = envs_[0]->getActDim();
	rew_dim_    = envs_[0]->getRewDim();
	img_width_  = envs_[0]->getImgWidth();
	img_height_ = envs_[0]->getImgHeight();

	// if supervised_mode, then generate map from .ply
	if (supervised_mode_)
		generateMaps();
	// set dagger_mode to each env
	for (int i = 0; i < num_envs_; i++)
		envs_[i]->setDAggerMode(dagger_mode_);

	std::cout << "Vectorized Environment:\n"
	          << "dagger mode       =        [" << dagger_mode_ << "]\n"
	          << "supervised mode   =        [" << supervised_mode_ << "]\n"
	          << "obs dim           =        [" << obs_dim_ << "]\n"
	          << "act dim           =        [" << act_dim_ << "]\n"
	          << "img width         =        [" << img_width_ << "]\n"
	          << "img height        =        [" << img_height_ << "]\n"
	          << "num_envs          =        [" << num_envs_ << "]\n"
	          << "num_thread        =        [" << num_threads_ << "]\n"
	          << "scene_id          =        [" << scene_id_ << "]" << std::endl;
}

template<typename EnvBase>
VecEnv<EnvBase>::~VecEnv() {}

// ======================	set functions	===================== //

template<typename EnvBase>
bool VecEnv<EnvBase>::reset(Ref<MatrixRowMajor<>> obs) {
	if (obs.rows() != num_envs_ || obs.cols() != obs_dim_) {
		logger_.error("Input matrix dimensions do not match with that of the environment.");
		return false;
	}

	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->reset(obs.row(i));
	}

	if (unity_render_ && unity_ready_) {
		frameID = 1;
		FrameID frameID_rt;
		unity_bridge_ptr_->getRender(frameID);
		unity_bridge_ptr_->handleOutput(frameID_rt);
		while (frameID != frameID_rt)
			unity_bridge_ptr_->handleOutput(frameID_rt);
	}
	return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::setState(ConstRef<MatrixRowMajor<>> state) {
	if (state.rows() != num_envs_ || state.cols() != 13) {  // 13: pvaq
		logger_.error("Input state dimensions do not match with state.");
		return false;
	}

	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->setState(state.row(i));
	}

	return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::setGoal(ConstRef<MatrixRowMajor<>> goal) {
	if (goal.rows() != num_envs_ || goal.cols() != 3) {
		logger_.error("Input goal dimensions do not match with 3.");
		return false;
	}

	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->setGoal(goal.row(i));
	}

	return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> reward, Ref<BoolVector<>> done) {
	if (act.rows() != num_envs_ || act.cols() != act_dim_ || obs.rows() != num_envs_ || obs.cols() != obs_dim_ || reward.rows() != num_envs_ ||
	    reward.cols() != rew_dim_ || done.rows() != num_envs_ || done.cols() != 1) {
		logger_.error("Input matrix dimensions do not match with that of the environment.");
		return false;
	}

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < num_envs_; i++) {
		perAgentStep(i, act, obs, reward, done);
	}

	if (unity_render_ && unity_ready_) {
		frameID++;
		FrameID frameID_rt;
		unity_bridge_ptr_->getRender(frameID);
		unity_bridge_ptr_->handleOutput(frameID_rt);
		while (frameID != frameID_rt)
			unity_bridge_ptr_->handleOutput(frameID_rt);
	}
	return true;
}

template<typename EnvBase>
void VecEnv<EnvBase>::perAgentStep(int agent_id, Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> reward,
                                   Ref<BoolVector<>> done) {
	envs_[agent_id]->step(act.row(agent_id), obs.row(agent_id), reward.row(agent_id));

	// use larger collision threshold in training and lower in testing
	if (dagger_mode_)
		envs_[agent_id]->collisionCheck(0.3);
	else
		envs_[agent_id]->collisionCheck(0.1);

	Scalar terminal_reward = 0;
	done(agent_id)         = envs_[agent_id]->isTerminalState(terminal_reward);

	if (done[agent_id]) {
		envs_[agent_id]->reset(obs.row(agent_id));
	}
}

template<typename EnvBase>
void VecEnv<EnvBase>::setMapID(ConstRef<IntVector<>> id) {
	if (id.rows() != num_envs_) {
		logger_.error("Input matrix dimensions do not match with that of the environment.");
		return;
	}

	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->setMapID(id(i));
	}
}

template<typename EnvBase>
void VecEnv<EnvBase>::setSeed(const int seed) {
	int seed_inc = seed;
	for (int i = 0; i < num_envs_; i++)
		envs_[i]->setSeed(seed_inc++);
}

// ======================	set functions	===================== //

template<typename EnvBase>
void VecEnv<EnvBase>::getObs(Ref<MatrixRowMajor<>> obs) {
	for (int i = 0; i < num_envs_; i++)
		envs_[i]->getObs(obs.row(i));
}

template<typename EnvBase>
bool VecEnv<EnvBase>::getRGBImage(Ref<ImgMatrixRowMajor<>> img, const bool rgb_image) {
	bool valid_img = true;
	for (int i = 0; i < num_envs_; i++) {
		valid_img &= envs_[i]->getRGBImage(img.row(i), rgb_image);
	}
	return valid_img;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::getStereoImage(Ref<DepthImgMatrixRowMajor<>> stereo_img) {
	bool valid_img = true;
	for (int i = 0; i < num_envs_; i++) {
		valid_img &= envs_[i]->getStereoImage(stereo_img.row(i));
	}
	return valid_img;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::getDepthImage(Ref<DepthImgMatrixRowMajor<>> depth_img) {
	bool valid_img = true;
	for (int i = 0; i < num_envs_; i++) {
		valid_img &= envs_[i]->getDepthImage(depth_img.row(i));
	}
	return valid_img;
}


template<typename EnvBase>
void VecEnv<EnvBase>::getCostAndGradient(ConstRef<MatrixRowMajor<>> dp, ConstRef<IntVector<>> traj_id, Ref<Vector<>> cost,
                                         Ref<MatrixRowMajor<>> grad) {
#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->getCostAndGradient(dp.row(i), traj_id(i), cost(i), grad.row(i));
	}
}


// ======================	unity functions	===================== //

template<typename EnvBase>
bool VecEnv<EnvBase>::setUnity(bool render) {
	unity_render_ = render;
	if (unity_render_ && unity_bridge_ptr_ == nullptr) {
		// create unity bridge
		unity_bridge_ptr_ = UnityBridge::getInstance();
		// add objects to Unity
		for (int i = 0; i < num_envs_; i++) {
			envs_[i]->addObjectsToUnity(unity_bridge_ptr_);
		}
		logger_.info("Flightmare Bridge is created.");
	}
	return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::spawnTrees() {
	if (!unity_ready_ || unity_bridge_ptr_ == nullptr)
		return false;
	bool spawned = unity_bridge_ptr_->spawnTrees(bounding_box_, bounding_box_origin_, avg_tree_spacing_);
	return spawned;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::savePointcloud(int ply_id) {
	if (!unity_ready_ || unity_bridge_ptr_ == nullptr)
		return false;
	Vector<3> min_corner = bounding_box_origin_ - 0.5 * bounding_box_;
	Vector<3> max_corner = bounding_box_origin_ + 0.5 * bounding_box_;
	unity_bridge_ptr_->generatePointcloud(min_corner, max_corner, ply_id, ply_path_, scene_id_, pointcloud_resolution_);
	return true;
}

template<typename EnvBase>
bool VecEnv<EnvBase>::spawnTreesAndSavePointcloud(int ply_id_in, float spacing) {
	Scalar avg_tree_spacing = avg_tree_spacing_;
	if (spacing > 0)
		avg_tree_spacing = spacing;
	int ply_id = envs_[0]->getMapNum();
	if (ply_id_in >= 0)
		ply_id = ply_id_in;

	if (!unity_ready_ || unity_bridge_ptr_ == nullptr)
		return false;

	bool spawned = unity_bridge_ptr_->spawnTrees(bounding_box_, bounding_box_origin_, avg_tree_spacing);

	Vector<3> min_corner = bounding_box_origin_ - 0.5 * bounding_box_;
	Vector<3> max_corner = bounding_box_origin_ + 0.5 * bounding_box_;
	unity_bridge_ptr_->generatePointcloud(min_corner, max_corner, ply_id, ply_path_, scene_id_, pointcloud_resolution_);

	usleep(1 * 1e6);  // waitting 1s for generating completely

	// KDtree, for collision detection
	pcl::search::KdTree<pcl::PointXYZ> kdtree;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	std::cout << "Map Path: " << ply_path_ + "pointcloud-" + std::to_string(ply_id) + ".ply" << std::endl;
	pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path_ + "pointcloud-" + std::to_string(ply_id) + ".ply", *cloud);
	kdtree.setInputCloud(cloud);  // 0.3s

	// ESDF, for gradient calculation (map_id is required)
	Eigen::Vector3d map_boundary_min, map_boundary_max;
	std::shared_ptr<sdf_tools::SignedDistanceField> esdf_map = traj_opt::SdfConstruction(cloud, map_boundary_min, map_boundary_max);

	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->addKdtree(std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>(kdtree));
		envs_[i]->addESDFMap(esdf_map);
		envs_[i]->addMapSize(map_boundary_min, map_boundary_max);
	}

	return true;
}

// For supervised learning
template<typename EnvBase>
void VecEnv<EnvBase>::generateMaps() {
	std::vector<std::string> ply_files;
	for (const auto& entry : std::filesystem::directory_iterator(ply_path_)) {
		if (entry.is_regular_file() && entry.path().extension() == ".ply") {
			ply_files.push_back(entry.path().string());
		}
	}

	// Sort according to the number of the filename.
	std::sort(ply_files.begin(), ply_files.end(), [this](const std::string& a, const std::string& b) {
		return extract_number(std::filesystem::path(a).filename().string()) < extract_number(std::filesystem::path(b).filename().string());
	});

	for (auto ply_file : ply_files) {
		std::cout << "load ply file: " << ply_file << std::endl;
		pcl::search::KdTree<pcl::PointXYZ> kdtree;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile<pcl::PointXYZ>(ply_file, *cloud);
		// KDtree, for collision detection
		kdtree.setInputCloud(cloud);  // 0.3s
		// ESDF, for gradient calculation
		Eigen::Vector3d map_boundary_min, map_boundary_max;
		std::shared_ptr<sdf_tools::SignedDistanceField> esdf_map = traj_opt::SdfConstruction(cloud, map_boundary_min, map_boundary_max);

		std::cout << "pc min:" << map_boundary_min.transpose() << " pc max:" << map_boundary_max.transpose() << std::endl;
		for (int i = 0; i < num_envs_; i++) {
			envs_[i]->addKdtree(std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>(kdtree));
			envs_[i]->addESDFMap(esdf_map);
			envs_[i]->addMapSize(map_boundary_min, map_boundary_max);
		}
	}
}

template<typename EnvBase>
bool VecEnv<EnvBase>::connectUnity(void) {
	if (unity_bridge_ptr_ == nullptr)
		return false;
	unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
	return unity_ready_;
}

template<typename EnvBase>
void VecEnv<EnvBase>::render(void) {
	if (unity_render_ && unity_ready_) {
		frameID++;
		FrameID frameID_rt;
		unity_bridge_ptr_->getRender(frameID);
		unity_bridge_ptr_->handleOutput(frameID_rt);
		while (frameID != frameID_rt)
			unity_bridge_ptr_->handleOutput(frameID_rt);
	}
}

template<typename EnvBase>
void VecEnv<EnvBase>::disconnectUnity(void) {
	if (unity_bridge_ptr_ != nullptr) {
		unity_bridge_ptr_->disconnectUnity();
		unity_ready_ = false;
	} else {
		logger_.warn("Flightmare Unity Bridge is not initialized.");
	}
}

template<typename EnvBase>
void VecEnv<EnvBase>::close() {
	for (int i = 0; i < num_envs_; i++) {
		envs_[i]->close();
	}
}

// ======================	other functions	===================== //

// Extract the number from the filename (e.g., pointcloud-1.ply)
template<typename EnvBase>
int VecEnv<EnvBase>::extract_number(const std::string& filename) {
	std::regex number_regex("pointcloud-(\\d+)\\.ply");
	std::smatch match;
	if (std::regex_search(filename, match, number_regex)) {
		return std::stoi(match[1]);
	}
	return -1;  // If no number is found, return -1
}

// IMPORTANT. Otherwise:
// Segmentation fault (core dumped)
template class VecEnv<QuadrotorEnv>;

}  // namespace flightlib
