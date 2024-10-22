#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

Quadrotor::Quadrotor()
    : world_box_((Matrix<3, 2>() << -100, 100, -100, 100, -100, 100).finished()), size_(1.0, 1.0, 1.0), collision_(false) {
	reset();
}

Quadrotor::~Quadrotor() {}

bool Quadrotor::setState(const Ref<Vector<>> p_, const Ref<Vector<>> v_, const Quaternion q_, const Ref<Vector<>> a_, const Scalar ctl_dt) {
	QuadState old_state = state_;
	state_.p            = p_;
	state_.v            = v_;
	state_.q(q_);
	state_.a = a_;
	state_.t += ctl_dt;
	constrainInWorldBox(old_state);
	return true;
}

bool Quadrotor::reset(void) {
	state_.setZero();
	return true;
}

bool Quadrotor::reset(const QuadState &state) {
	if (!state.valid())
		return false;
	state_ = state;
	return true;
}

/*
    There is no controller (or using an ideal controller). The attitude is simply obtained from the desired acceleration.
    This is because our algorithm is only concerned with the quality of the trajectory, while control is performed by external controller.
*/
void Quadrotor::runSimpleFlight(const Eigen::Vector3f &ref_acc, float ref_yaw, Eigen::Quaternionf &quat_des) {
	float mass_ = 1.0;
	float ONE_G = 9.8;
	// float M_PI             = 3.1415925;
	Eigen::Vector3f force_ = mass_ * ONE_G * Eigen::Vector3f(0, 0, 1);
	force_.noalias() += mass_ * ref_acc;

	// Limit control angle to theta degree
	float theta = M_PI / 4;
	float c     = cos(theta);
	Eigen::Vector3f f;
	f.noalias() = force_ - mass_ * ONE_G * Eigen::Vector3f(0, 0, 1);
	if (Eigen::Vector3f(0, 0, 1).dot(force_ / force_.norm()) < c) {
		float nf         = f.norm();
		float A          = c * c * nf * nf - f(2) * f(2);
		float B          = 2 * (c * c - 1) * f(2) * mass_ * ONE_G;
		float C          = (c * c - 1) * mass_ * mass_ * ONE_G * ONE_G;
		float s          = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
		force_.noalias() = s * f + mass_ * ONE_G * Eigen::Vector3f(0, 0, 1);
	}

	Eigen::Vector3f b1c, b2c, b3c;
	Eigen::Vector3f b1d(cos(ref_yaw), sin(ref_yaw), 0);

	if (force_.norm() > 1e-6)
		b3c.noalias() = force_.normalized();
	else
		b3c.noalias() = Eigen::Vector3f(0, 0, 1);

	b2c.noalias() = b3c.cross(b1d).normalized();
	b1c.noalias() = b2c.cross(b3c).normalized();

	Eigen::Matrix3f R;
	R << b1c, b2c, b3c;

	quat_des = Eigen::Quaternionf(R);
}

bool Quadrotor::setState(const QuadState &state) {
	if (!state.valid())
		return false;
	state_ = state;
	return true;
}

bool Quadrotor::setWorldBox(const Ref<Matrix<3, 2>> box) {
	if (box(0, 0) >= box(0, 1) || box(1, 0) >= box(1, 1) || box(2, 0) >= box(2, 1)) {
		return false;
	}
	world_box_ = box;
	return true;
}

bool Quadrotor::constrainInWorldBox(const QuadState &old_state) {
	if (!old_state.valid())
		return false;

	// violate world box constraint in the x-axis
	if (state_.x(QS::POSX) < world_box_(0, 0) || state_.x(QS::POSX) > world_box_(0, 1)) {
		state_.x(QS::POSX) = old_state.x(QS::POSX);
		state_.x(QS::VELX) = 0.0;
	}

	// violate world box constraint in the y-axis
	if (state_.x(QS::POSY) < world_box_(1, 0) || state_.x(QS::POSY) > world_box_(1, 1)) {
		state_.x(QS::POSY) = old_state.x(QS::POSY);
		state_.x(QS::VELY) = 0.0;
	}

	// violate world box constraint in the x-axis
	if (state_.x(QS::POSZ) <= world_box_(2, 0) || state_.x(QS::POSZ) > world_box_(2, 1)) {
		//
		state_.x(QS::POSZ) = world_box_(2, 0);

		// reset velocity to zero
		state_.x(QS::VELX) = 0.0;
		state_.x(QS::VELY) = 0.0;

		// reset acceleration to zero
		state_.a << 0.0, 0.0, 0.0;
		// reset angular velocity to zero
		state_.w << 0.0, 0.0, 0.0;
	}
	return true;
}

bool Quadrotor::getState(QuadState *const state) const {
	if (!state_.valid())
		return false;

	*state = state_;
	return true;
}

bool Quadrotor::addRGBCamera(std::shared_ptr<RGBCamera> camera) {
	rgb_cameras_.push_back(camera);
	return true;
}

Vector<3> Quadrotor::getSize(void) const { return size_; }

Vector<3> Quadrotor::getPosition(void) const { return state_.p; }

std::vector<std::shared_ptr<RGBCamera>> Quadrotor::getCameras(void) const { return rgb_cameras_; };

bool Quadrotor::getCamera(const size_t cam_id, std::shared_ptr<RGBCamera> camera) const {
	if (cam_id <= rgb_cameras_.size()) {
		return false;
	}

	camera = rgb_cameras_[cam_id];
	return true;
}

bool Quadrotor::getCollision() const { return collision_; }

int Quadrotor::getNumCamera() const { return rgb_cameras_.size(); }

}  // namespace flightlib
