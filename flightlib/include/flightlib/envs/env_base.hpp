//
// This is inspired by RaiGym, thanks.
//
#pragma once

#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "flightlib/common/types.hpp"

namespace flightlib {

class EnvBase {
   public:
	EnvBase() : obs_dim_(0), act_dim_(0), rew_dim_(0), sim_dt_(0.0), img_width_(0), img_height_(0) {};
	virtual ~EnvBase() {};

	// (pure virtual) public methods (has to be implemented by child classes)
	virtual bool reset(Ref<Vector<>> obs, const bool random = true)                     = 0;
	virtual bool step(const Ref<Vector<>> act, Ref<Vector<>> obs, Ref<Vector<>> reward) = 0;
	virtual bool getObs(Ref<Vector<>> obs)                                              = 0;

	// (virtual) public methods (implementations are optional.)
	virtual bool getDepthImage(Ref<DepthImgVector<>> img) { return false; }
	virtual void close() {};
	// virtual void render();
	virtual void updateExtraInfo() {};
	virtual bool isTerminalState(Scalar &reward) {
		reward = 0.f;
		return false;
	}

	// auxilirary functions
	inline void setSeed(const int seed) { std::srand(seed); };
	inline int getObsDim() { return obs_dim_; };
	inline int getActDim() { return act_dim_; };
	inline int getRewDim() { return rew_dim_; };
	inline int getImgWidth() { return img_width_; };
	inline int getImgHeight() { return img_height_; };
	inline Scalar getSimTimeStep() { return sim_dt_; };
	inline int getExtraInfoDim() { return extra_info_.size(); };

	// public variables
	std::unordered_map<std::string, float> extra_info_;

   protected:
	// observation and action dimenstions (for Reinforcement learning)
	int obs_dim_;
	int act_dim_;
	int rew_dim_;

	int img_width_;
	int img_height_;

	// control time step
	Scalar sim_dt_{0.02};

	// random variable generator
	std::normal_distribution<Scalar> norm_dist_{0.0, 1.0};
	std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
	std::random_device rd_;
	std::mt19937 random_gen_{rd_()};
};

}  // namespace flightlib