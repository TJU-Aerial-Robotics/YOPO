//
// This is inspired by RaiGym, thanks.
//
#pragma once

// std
#include <omp.h>
#include <time.h>

#include <filesystem>
#include <memory>
#include <regex>

// pcl
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env.hpp"
#include "flightlib/grad_traj_optimization/traj_optimization_bridge.h"

namespace flightlib {

template<typename EnvBase>
class VecEnv {
   public:
	VecEnv();
	VecEnv(const std::string& cfgs, const bool from_file = true);
	VecEnv(const YAML::Node& cfgs_node);
	~VecEnv();

	/* unity functions */
	bool connectUnity();
	void disconnectUnity();
	bool setUnity(bool render);
	void render();
	bool spawnTrees();
	bool savePointcloud(int ply_id);
	bool spawnTreesAndSavePointcloud(int ply_id_in = -1, float spacing = -1);
	void close();
	void setSeed(const int seed);

	/* set functions */
	bool reset(Ref<MatrixRowMajor<>> obs);
	bool step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> reward, Ref<BoolVector<>> done);
	void perAgentStep(int agent_id, Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> reward, Ref<BoolVector<>> done);
	bool setState(ConstRef<MatrixRowMajor<>> state);  // World Frame
	bool setGoal(ConstRef<MatrixRowMajor<>> goal);    // World Frame
	void setMapID(ConstRef<IntVector<>> id);

	/* get functions */
	inline int getObsDim(void) { return obs_dim_; };
	inline int getActDim(void) { return act_dim_; };
	inline int getRewDim(void) const { return rew_dim_; };
	inline int getImgHeight(void) const { return img_height_; };
	inline int getImgWidth(void) const { return img_width_; };
	inline int getNumOfEnvs(void) { return envs_.size(); };
	inline std::vector<std::string> getRewardNames(void) { return envs_[0]->getRewardNames(); };
	inline void getWorldBox(Ref<Vector<>> world_box) { envs_[0]->getWorldBox(world_box); };
	void getObs(Ref<MatrixRowMajor<>> obs);
	bool getRGBImage(Ref<ImgMatrixRowMajor<>> img, const bool rgb_image);
	bool getStereoImage(Ref<DepthImgMatrixRowMajor<>> img);
	bool getDepthImage(Ref<DepthImgMatrixRowMajor<>> img);
	void getCostAndGradient(ConstRef<MatrixRowMajor<>> dp, ConstRef<IntVector<>> traj_id, Ref<Vector<>> cost,
	                        Ref<MatrixRowMajor<>> grad);  // Body Frame

	/* other functions */
	void init(void);
	void generateMaps();
	int extract_number(const std::string& filename);

   private:
	// create objects
	Logger logger_{"VecEnv"};
	std::vector<std::unique_ptr<EnvBase>> envs_;

	// Flightmare(Unity3D)
	std::shared_ptr<UnityBridge> unity_bridge_ptr_;
	SceneID scene_id_{UnityScene::WAREHOUSE};
	bool unity_ready_{false};
	bool unity_render_{false};
	FrameID frameID{1};
	RenderMessage_t unity_output_;
	uint16_t receive_id_{0};

	// scenario generation
	Scalar avg_tree_spacing_;
	Vector<3> bounding_box_, bounding_box_origin_;
	Scalar pointcloud_resolution_;

	// other variables
	std::string ply_path_;
	bool dagger_mode_{false}, supervised_mode_{false};
	int seed_, num_envs_, obs_dim_, act_dim_, rew_dim_, num_threads_;
	int img_width_, img_height_;

	// yaml configurations
	YAML::Node cfg_;
};

}  // namespace flightlib
