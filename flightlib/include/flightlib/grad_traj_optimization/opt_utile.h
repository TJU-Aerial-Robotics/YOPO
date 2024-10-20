#ifndef _LATTICE_NODE_H_
#define _LATTICE_NODE_H_
#include <Eigen/Eigen>

// body frame
void getLatticeGuiding(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &lattice_nodes, int horizon_num, int vertical_num, int radio_num,
                       int vel_num, double horizon_fov, double vertical_fov, double radio_range, double vel_fov, double vel_prefile);

void getPositionFromCoeff(Eigen::Vector3d &pos, Eigen::MatrixXd coeff, int index, double time);

void getVelocityFromCoeff(Eigen::Vector3d &vel, Eigen::MatrixXd coeff, int index, double time);

void getAccelerationFromCoeff(Eigen::Vector3d &acc, Eigen::MatrixXd coeff, int index, double time);

Eigen::MatrixXd solveCoeffFromBoundaryState(const Eigen::Vector3d &Pos_init, const Eigen::Vector3d &Vel_init, const Eigen::Vector3d &Acc_init,
                                            const Eigen::Vector3d &Pos_end, const Eigen::Vector3d &Vel_end, const Eigen::Vector3d &Acc_end,
                                            double Time);
#endif