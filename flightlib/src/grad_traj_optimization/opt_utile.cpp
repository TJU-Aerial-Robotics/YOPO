#include "flightlib/grad_traj_optimization/opt_utile.h"

/*
    Front-End Guiding Path:
    We evenly sample vertical_num * horizon_num * radio_num * vel_num primitives here with different position, length, and velocity direction.
    But in practical, only vertical_num * horizon_num primitives are sampled (radio_num = vel_num = 1).
*/
void getLatticeGuiding(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &lattice_nodes, int horizon_num, int vertical_num, int radio_num,
                       int vel_num, double horizon_fov, double vertical_fov, double radio_range, double vel_fov, double vel_prefile) {
	double direction_diff, altitude_diff, radio_diff, vel_dir_diff;
	if (horizon_num == 1)
		direction_diff = 0;
	else
		direction_diff = (horizon_fov / 180.0 * M_PI) / (horizon_num - 1);
	if (vertical_num == 1)
		altitude_diff = 0;
	else
		altitude_diff = (vertical_fov / 180.0 * M_PI) / (vertical_num - 1);
	radio_diff = radio_range / radio_num;
	if (vel_num == 1)
		vel_dir_diff = 0;
	else
		vel_dir_diff = (vel_fov / 180.0f * M_PI) / (vel_num - 1);
	// if (vel_num == 1)   // be 0 looks like better
	//     vel_prefile = 0.0;
	lattice_nodes.clear();

	for (int h = 0; h < radio_num; h++) {
		for (int i = 0; i < vertical_num; i++) {
			for (int j = 0; j < horizon_num; j++) {
				for (int k = 0; k < vel_num; k++) {
					double search_radio = (h + 1) * radio_diff;
					double alpha        = -direction_diff * (horizon_num - 1) / 2 + j * direction_diff;  // 位置偏航角（从右往左）
					double beta         = -altitude_diff * (vertical_num - 1) / 2 + i * altitude_diff;   // 高度偏移角（从下往上）
					double gamma        = -vel_dir_diff * (vel_num - 1) / 2 + k * vel_dir_diff;          // 速度偏航
					Eigen::Vector3d lattice_node_pos(cos(beta) * cos(alpha) * search_radio, cos(beta) * sin(alpha) * search_radio,
					                                 sin(beta) * search_radio);
					Eigen::Vector3d lattice_node_vel(cos(alpha + gamma) * vel_prefile, sin(alpha + gamma) * vel_prefile, 0.0);
					std::pair<Eigen::Vector3d, Eigen::Vector3d> lattice_node(lattice_node_pos, lattice_node_vel);
					lattice_nodes.push_back(lattice_node);
				}
			}
		}
	}
}

Eigen::MatrixXd solveCoeffFromBoundaryState(const Eigen::Vector3d &Pos_init, const Eigen::Vector3d &Vel_init, const Eigen::Vector3d &Acc_init,
                                            const Eigen::Vector3d &Pos_end, const Eigen::Vector3d &Vel_end, const Eigen::Vector3d &Acc_end,
                                            double Time) {
	// Solution of Boundary Value Problem for a 5th-Order Polynomial
	Eigen::MatrixXd PolyCoeff(1, 3 * 6);
	Eigen::VectorXd Px(6), Py(6), Pz(6);
	
	Eigen::MatrixXd A_inv(6,6);
	A_inv << 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 1.0 / 2.0, 0,
            -10.0 / pow(Time, 3), 10.0 / pow(Time, 3), -6.0 / pow(Time, 2), -4.0 / pow(Time, 2), -3.0 / (2.0 * Time), 1.0 / (2.0 * Time),
             15.0 / pow(Time, 4), -15.0 / pow(Time, 4), 8.0 / pow(Time, 3), 7.0 / pow(Time, 3), 3.0 / (2.0 * pow(Time, 2)), -1.0 / pow(Time, 2),
            -6.0 / pow(Time, 5), 6.0 / pow(Time, 5), -3.0 / pow(Time, 4), -3.0 / pow(Time, 4), -1.0 / (2.0 * pow(Time, 3)), 1.0 / (2.0 * pow(Time, 3));
    

	/*   Produce the dereivatives in X, Y and Z axis directly.  */
	Eigen::VectorXd Dx = Eigen::VectorXd::Zero(6);
	Eigen::VectorXd Dy = Eigen::VectorXd::Zero(6);
	Eigen::VectorXd Dz = Eigen::VectorXd::Zero(6);

    Dx(0) = Pos_init(0); Dx(1) = Pos_end(0); Dx(2) = Vel_init(0); Dx(3) = Vel_end(0); Dx(4) = Acc_init(0); Dx(5) = Acc_end(0);
    Dy(0) = Pos_init(1); Dy(1) = Pos_end(1); Dy(2) = Vel_init(1); Dy(3) = Vel_end(1); Dy(4) = Acc_init(1); Dy(5) = Acc_end(1);
    Dz(0) = Pos_init(2); Dz(1) = Pos_end(2); Dz(2) = Vel_init(2); Dz(3) = Vel_end(2); Dz(4) = Acc_init(2); Dz(5) = Acc_end(2);

    Px = A_inv * Dx;
    Py = A_inv * Dy;
    Pz = A_inv * Dz;

	PolyCoeff.block(0, 0, 1, 6)  = Px.segment(0, 6).transpose();
	PolyCoeff.block(0, 6, 1, 6)  = Py.segment(0, 6).transpose();
	PolyCoeff.block(0, 12, 1, 6) = Pz.segment(0, 6).transpose();

	return PolyCoeff;
}

void getPositionFromCoeff(Eigen::Vector3d &pos, Eigen::MatrixXd coeff, int index, double time) {
	int s    = index;
	double t = time;
	float x  = coeff(s, 0) + coeff(s, 1) * t + coeff(s, 2) * pow(t, 2) + coeff(s, 3) * pow(t, 3) + coeff(s, 4) * pow(t, 4) + coeff(s, 5) * pow(t, 5);
	float y  = coeff(s, 6) + coeff(s, 7) * t + coeff(s, 8) * pow(t, 2) + coeff(s, 9) * pow(t, 3) + coeff(s, 10) * pow(t, 4) + coeff(s, 11) * pow(t, 5);
	float z  = coeff(s, 12) + coeff(s, 13) * t + coeff(s, 14) * pow(t, 2) + coeff(s, 15) * pow(t, 3) + coeff(s, 16) * pow(t, 4) + coeff(s, 17) * pow(t, 5);

	pos(0) = x;
	pos(1) = y;
	pos(2) = z;
}

void getVelocityFromCoeff(Eigen::Vector3d &vel, Eigen::MatrixXd coeff, int index, double time) {
	int s    = index;
	double t = time;
	float vx = coeff(s, 1) + 2 * coeff(s, 2) * pow(t, 1) + 3 * coeff(s, 3) * pow(t, 2) + 4 * coeff(s, 4) * pow(t, 3) + 5 * coeff(s, 5) * pow(t, 4);
	float vy = coeff(s, 7) + 2 * coeff(s, 8) * pow(t, 1) + 3 * coeff(s, 9) * pow(t, 2) + 4 * coeff(s, 10) * pow(t, 3) + 5 * coeff(s, 11) * pow(t, 4);
	float vz = coeff(s, 13) + 2 * coeff(s, 14) * pow(t, 1) + 3 * coeff(s, 15) * pow(t, 2) + 4 * coeff(s, 16) * pow(t, 3) + 5 * coeff(s, 17) * pow(t, 4);

	vel(0) = vx;
	vel(1) = vy;
	vel(2) = vz;
}

void getAccelerationFromCoeff(Eigen::Vector3d &acc, Eigen::MatrixXd coeff, int index, double time) {
	int s    = index;
	double t = time;
	float ax = 2 * coeff(s, 2) + 6 * coeff(s, 3) * pow(t, 1) + 12 * coeff(s, 4) * pow(t, 2) + 20 * coeff(s, 5) * pow(t, 3);
	float ay = 2 * coeff(s, 8) + 6 * coeff(s, 9) * pow(t, 1) + 12 * coeff(s, 10) * pow(t, 2) + 20 * coeff(s, 11) * pow(t, 3);
	float az = 2 * coeff(s, 14) + 6 * coeff(s, 15) * pow(t, 1) + 12 * coeff(s, 16) * pow(t, 2) + 20 * coeff(s, 17) * pow(t, 3);

	acc(0) = ax;
	acc(1) = ay;
	acc(2) = az;
}