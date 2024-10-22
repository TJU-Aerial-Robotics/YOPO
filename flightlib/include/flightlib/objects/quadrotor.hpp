#pragma once

/*
    Explanation:
        For efficiency, we do not use the built-in dynamics model and instead
        directly use the desired attitude given by the controller as actual state.
        Because the proposed approach (YOPO) only focuses on the trajectory performance,
        while control is preformed by an external controller.
*/

#include <stdlib.h>

#include "flightlib/common/types.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/sensors/imu.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

class Quadrotor {
   public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Quadrotor();
	~Quadrotor();

	// reset
	bool reset(void);
	bool reset(const QuadState& state);

	// run the quadrotor
	bool setState(const Ref<Vector<>> p, const Ref<Vector<>> v, const Quaternion q_, const Ref<Vector<>> a_, const Scalar ctl_dt);
	void runSimpleFlight(const Eigen::Vector3f& ref_acc, float ref_yaw, Eigen::Quaternionf& quat_des);

	// public get functions
	bool getState(QuadState* const state) const;
	Vector<3> getSize(void) const;
	Vector<3> getPosition(void) const;
	Quaternion getQuaternion(void) const;
	int getNumCamera() const;
	bool getCollision() const;
	std::vector<std::shared_ptr<RGBCamera>> getCameras(void) const;
	bool getCamera(const size_t cam_id, std::shared_ptr<RGBCamera> camera) const;
	float getSimT() { return state_.t; }

	// public set functions
	bool setState(const QuadState& state);
	bool addRGBCamera(std::shared_ptr<RGBCamera> camera);
	inline void setSize(const Ref<Vector<3>> size) { size_ = size; };
	inline void setCollision(const bool collision) { collision_ = collision; };

	// constrain world box
	bool setWorldBox(const Ref<Matrix<3, 2>> box);
	bool constrainInWorldBox(const QuadState& old_state);

   private:
	// quadrotor sensors
	IMU imu_;
	std::vector<std::shared_ptr<RGBCamera>> rgb_cameras_;

	// quad state
	QuadState state_;
	Vector<3> size_;
	bool collision_;

	// auxiliary variables
	Matrix<3, 2> world_box_;
};

}  // namespace flightlib
