#include "flightlib/common/math.hpp"

#include "iostream"

namespace flightlib {

Matrix<3, 3> skew(const Vector<3>& v) { return (Matrix<3, 3>() << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0).finished(); }

Matrix<4, 4> Q_left(const Quaternion& q) {
	return (Matrix<4, 4>() << q.w(), -q.x(), -q.y(), -q.z(), q.x(), q.w(), -q.z(), q.y(), q.y(), q.z(), q.w(), -q.x(), q.z(), -q.y(), q.x(), q.w())
	    .finished();
}

Matrix<4, 4> Q_right(const Quaternion& q) {
	return (Matrix<4, 4>() << q.w(), -q.x(), -q.y(), -q.z(), q.x(), q.w(), q.z(), -q.y(), q.y(), -q.z(), q.w(), q.x(), q.z(), q.y(), -q.x(), q.w())
	    .finished();
}

Matrix<4, 3> qFromQeJacobian(const Quaternion& q) {
	return (Matrix<4, 3>() << -1.0 / q.w() * q.vec().transpose(), Matrix<3, 3>::Identity()).finished();
}

Matrix<4, 4> qConjugateJacobian() { return Matrix<4, 1>(1, -1, -1, -1).asDiagonal(); }

Matrix<3, 3> qeRotJacobian(const Quaternion& q, const Matrix<3, 1>& t) {
	return 2.0 * (Matrix<3, 3>() << (q.y() + q.z() * q.x() / q.w()) * t.y() + (q.z() - q.y() * q.x() / q.w()) * t.z(),        // entry 0,0
	              -2.0 * q.y() * t.x() + (q.x() + q.z() * q.y() / q.w()) * t.y() + (q.w() - q.y() * q.y() / q.w()) * t.z(),   // entry 0,1
	              -2.0 * q.z() * t.x() + (-q.w() + q.z() * q.z() / q.w()) * t.y() + (q.x() - q.y() * q.z() / q.w()) * t.z(),  // entry 0,2

	              (q.y() - q.z() * q.x() / q.w()) * t.x() + (-2.0 * q.x()) * t.y() + (-q.w() + q.x() * q.x() / q.w()) * t.z(),  // entry 1,0
	              (q.x() - q.z() * q.y() / q.w()) * t.x() + (q.z() + q.x() * q.y() / q.w()) * t.z(),                            // entry 1,1
	              (q.w() - q.z() * q.z() / q.w()) * t.x() + (-2.0 * q.z()) * t.y() + (q.y() + q.x() * q.z() / q.w()) * t.z(),   // entry 1,2

	              (q.z() + q.y() * q.x() / q.w()) * t.x() + (q.w() - q.x() * q.x() / q.w()) * t.y() + (-2.0 * q.x()) * t.z(),   // entry 2,0
	              (-q.w() + q.y() * q.y() / q.w()) * t.x() + (q.z() - q.x() * q.y() / q.w()) * t.y() + (-2.0 * q.y()) * t.z(),  // entry 2,1
	              (q.x() + q.y() * q.z() / q.w()) * t.x() + (q.y() - q.x() * q.z() / q.w()) * t.y()                             // entry 2,2
	              )
	                 .finished();
}

Matrix<3, 3> qeInvRotJacobian(const Quaternion& q, const Matrix<3, 1>& t) {
	return 2.0 * (Matrix<3, 3>() << (q.y() - q.z() * q.x() / q.w()) * t.y() + (q.z() + q.y() * q.x() / q.w()) * t.z(),       // entry 0,0
	              -2.0 * q.y() * t.x() + (q.x() - q.z() * q.y() / q.w()) * t.y() - (q.w() - q.y() * q.y() / q.w()) * t.z(),  // entry 0,1
	              -2.0 * q.z() * t.x() + (q.w() - q.z() * q.z() / q.w()) * t.y() + (q.x() + q.y() * q.z() / q.w()) * t.z(),  // entry 0,2

	              (q.y() + q.z() * q.x() / q.w()) * t.x() - 2.0 * q.x() * t.y() + (q.w() - q.x() * q.x() / q.w()) * t.z(),   // entry 1,0
	              (q.x() + q.z() * q.y() / q.w()) * t.x() + (q.z() - q.x() * q.y() / q.w()) * t.z(),                         // entry 1,1
	              -(q.w() - q.z() * q.z() / q.w()) * t.x() - 2.0 * q.z() * t.y() + (q.y() - q.x() * q.z() / q.w()) * t.z(),  // entry 1,2

	              (q.z() - q.y() * q.x() / q.w()) * t.x() - (q.w() - q.x() * q.x() / q.w()) * t.y() - 2.0 * q.x() * t.z(),  // entry 2,0
	              (q.w() - q.y() * q.y() / q.w()) * t.x() + (q.z() + q.x() * q.y() / q.w()) * t.y() - 2.0 * q.y() * t.z(),  // entry 2,1
	              (q.x() - q.y() * q.z() / q.w()) * t.x() + (q.y() + q.x() * q.z() / q.w()) * t.y()                         // entry 2,2
	              )
	                 .finished();
}

void matrixToTripletList(const SparseMatrix& matrix, std::vector<SparseTriplet>* const list, const int row_offset, const int col_offset) {
	list->reserve((size_t)matrix.nonZeros() + list->size());

	for (int i = 0; i < matrix.outerSize(); i++) {
		for (typename SparseMatrix::InnerIterator it(matrix, i); it; ++it) {
			list->emplace_back(it.row() + row_offset, it.col() + col_offset, it.value());
		}
	}
}

void matrixToTripletList(const Matrix<Dynamic, Dynamic>& matrix, std::vector<SparseTriplet>* const list, const int row_offset, const int col_offset) {
	const SparseMatrix sparse = matrix.sparseView();
	matrixToTripletList(sparse, list, row_offset, col_offset);
}

void insert(const SparseMatrix& from, SparseMatrix* const into, const int row_offset, const int col_offset) {
	std::vector<SparseTriplet> v;

	matrixToTripletList(*into, &v);
	matrixToTripletList(from, &v, row_offset, col_offset);

	into->setFromTriplets(v.begin(), v.end(), [](const Scalar& older, const Scalar& newer) { return newer; });
}

void insert(const Matrix<>& from, SparseMatrix* const into, const int row_offset, const int col_offset) {
	const SparseMatrix from_sparse = from.sparseView();
	insert(from_sparse, into, row_offset, col_offset);
}

inline void insert(const Matrix<>& from, Matrix<>* const into, const int row_offset, const int col_offset) {
	into->block(row_offset, col_offset, from.rows(), from.cols()) = from;
}

void quaternionToEuler(const Quaternion& quat, Ref<Vector<3>> euler) {
	euler.x() = std::atan2(2 * quat.w() * quat.x() + 2 * quat.y() * quat.z(),
	                       quat.w() * quat.w() - quat.x() * quat.x() - quat.y() * quat.y() + quat.z() * quat.z());
	euler.y() = -std::asin(2 * quat.x() * quat.z() - 2 * quat.w() * quat.y());
	euler.z() = std::atan2(2 * quat.w() * quat.z() + 2 * quat.x() * quat.y(),
	                       quat.w() * quat.w() + quat.x() * quat.x() - quat.y() * quat.y() - quat.z() * quat.z());
}


std::vector<Scalar> transformationRos2Unity(const Matrix<4, 4>& ros_tran_mat) {
	/// [ Transformation Matrix ] from ROS coordinate system (right hand)
	/// to Unity coordinate system (left hand)
	Matrix<4, 4> tran_mat = Matrix<4, 4>::Zero();
	tran_mat(0, 0)        = 1.0;
	tran_mat(1, 2)        = 1.0;
	tran_mat(2, 1)        = 1.0;
	tran_mat(3, 3)        = 1.0;
	//
	Matrix<4, 4> unity_tran_mat = tran_mat * ros_tran_mat * tran_mat.transpose();
	// std::vector<Scalar> unity_tran_mat_vec(
	//   unity_tran_mat.data(),
	//   unity_tran_mat.data() + unity_tran_mat.rows() * unity_tran_mat.cols());
	std::vector<Scalar> tran_unity;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			tran_unity.push_back(unity_tran_mat(i, j));
		}
	}
	return tran_unity;
}

std::vector<Scalar> quaternionRos2Unity(const Quaternion& ros_quat) {
	/// [ Quaternion ] from ROS coordinate system (right hand)
	/// to Unity coordinate system (left hand)
	Matrix<3, 3> rot_mat = Matrix<3, 3>::Zero();
	rot_mat(0, 0)        = 1.0;
	rot_mat(1, 2)        = 1.0;
	rot_mat(2, 1)        = 1.0;
	//
	Matrix<3, 3> unity_rot_mat = rot_mat * ros_quat.toRotationMatrix() * rot_mat.transpose();
	Quaternion unity_quat(unity_rot_mat);
	std::vector<Scalar> unity_quat_vec{unity_quat.x(), unity_quat.y(), unity_quat.z(), unity_quat.w()};
	return unity_quat_vec;
}

std::vector<Scalar> positionRos2Unity(const Vector<3>& ros_pos_vec) {
	/// [ Position Vector ] from ROS coordinate system (right hand)
	/// to Unity coordinate system (left hand)
	std::vector<Scalar> unity_position{ros_pos_vec(0), ros_pos_vec(2), ros_pos_vec(1)};
	return unity_position;
}

std::vector<Scalar> scalarRos2Unity(const Vector<3>& ros_scalar) {
	/// [ Object Scalar Vector ] from ROS coordinate system (right hand)
	/// to Unity coordinate system (left hand)
	std::vector<Scalar> unity_scalar{ros_scalar(0), ros_scalar(2), ros_scalar(1)};
	return unity_scalar;
}

// rpy顺序
void get_euler_from_R(Vector<3>& e, const Matrix<3, 3>& R) {
	float phi   = atan2(R(2, 1), R(2, 2));
	float theta = asin(-R(2, 0));
	float psi   = atan2(R(1, 0), R(0, 0));
	float pi    = M_PI;

	if (fabs(theta - pi / 2.0f) < 1.0e-3) {
		phi = 0.0f;
		psi = atan2(R(1, 2), R(0, 2));
	} else if (fabs(theta + pi / 2.0f) < 1.0e-3) {
		phi = 0.0f;
		psi = atan2(-R(1, 2), -R(0, 2));
	}
	e(0) = phi;
	e(1) = theta;
	e(2) = psi;
}

double wrapMinusPiToPi(const double angle) {
	if (angle >= -M_PIl && angle <= M_PIl) {
		return angle;
	}
	double wrapped_angle = angle + M_PIl;
	wrapped_angle        = wrapZeroToTwoPi(wrapped_angle);
	wrapped_angle -= M_PIl;
	return wrapped_angle;
}

double wrapZeroToTwoPi(const double angle) {
	if (angle >= 0.0 && angle <= 2.0 * M_PIl) {
		return angle;
	}
	double wrapped_angle = fmod(angle, 2.0 * M_PIl);
	if (wrapped_angle < 0.0) {
		wrapped_angle += 2.0 * M_PIl;
	}
	return wrapped_angle;
}

// calculate and constrain the yaw angle per sim_t
float calculate_yaw(float yaw_cur, float yaw_ref, float sim_t)  // yaw [-pi,pi]
{
	float PI                  = 3.1415926;
	float YAW_DOT_MAX_PER_SEC = 0.3 * PI;
	float max_yaw_change      = YAW_DOT_MAX_PER_SEC * sim_t;
	float yaw_temp            = yaw_ref;
	float last_yaw_           = yaw_cur;
	float yaw                 = 0;

	if (yaw_temp - last_yaw_ > PI) {
		if (yaw_temp - last_yaw_ - 2 * PI < -max_yaw_change) {
			yaw = last_yaw_ - max_yaw_change;
			if (yaw < -PI)
				yaw += 2 * PI;
		} else {
			yaw = yaw_temp;
		}
	} else if (yaw_temp - last_yaw_ < -PI) {
		if (yaw_temp - last_yaw_ + 2 * PI > max_yaw_change) {
			yaw = last_yaw_ + max_yaw_change;
			if (yaw > PI)
				yaw -= 2 * PI;
		} else {
			yaw = yaw_temp;
		}
	} else {
		if (yaw_temp - last_yaw_ < -max_yaw_change) {
			yaw = last_yaw_ - max_yaw_change;
			if (yaw < -PI)
				yaw += 2 * PI;
		} else if (yaw_temp - last_yaw_ > max_yaw_change) {
			yaw = last_yaw_ + max_yaw_change;
			if (yaw > PI)
				yaw -= 2 * PI;
		} else {
			yaw = yaw_temp;
		}
	}

	return yaw;
}

}  // namespace flightlib