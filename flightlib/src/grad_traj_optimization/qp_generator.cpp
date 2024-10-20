/*
    Currently, only 5-order single-segment polynomial is used,
    but the functionality for piecewise polynomials is retained (i.e. m, num_f, num_p and other variables).
*/

#include "flightlib/grad_traj_optimization/qp_generator.h"

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace Eigen;

TrajectoryGenerator::TrajectoryGenerator() {}

TrajectoryGenerator::~TrajectoryGenerator() {}

void TrajectoryGenerator::QPGeneration(const Eigen::VectorXd &Time) {
	m = Time.size();

	// 阶乘
	const static auto Factorial = [](int x) {
		int fac = 1;

		for (int i = x; i > 0; i--)
			fac = fac * i;

		return fac;
	};

	/*   Produce Mapping Matrix A to the entire trajectory.   */
	MatrixXd Ab;                                // 每一段矩阵的A（论文中M）
	MatrixXd A = MatrixXd::Zero(m * 6, m * 6);  // m是段数

	// Ab 的组成为6行，第1行Tk位置，第二行Tk+1位置，第三行Tk速度，第四行Tk+1速度，五六为加速度
	// 采用5次多项式，所以每段轨迹有6个多项式系数（列）
	for (int k = 0; k < m; k++) {
		Ab = Eigen::MatrixXd::Zero(6, 6);
		for (int i = 0; i < 3; i++) {
			Ab(2 * i, i) = Factorial(i);
			for (int j = i; j < 6; j++)
				Ab(2 * i + 1, j) = Factorial(j) / Factorial(j - i) * pow(Time(k), j - i);
		}
		A.block(k * 6, k * 6, 6, 6) = Ab;
	}

	_A = A;

	/*   Produce the Minimum Snap cost function, the Hessian Matrix   */
	MatrixXd H = MatrixXd::Zero(m * 6, m * 6);
	// min snap 的cost function（四阶导数的积分 和 系数的关系 的矩阵），论文中间的矩阵Q
	for (int k = 0; k < m; k++) {
		for (int i = 3; i < 6; i++) {
			for (int j = 3; j < 6; j++) {
				H(k * 6 + i, k * 6 + j) = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * pow(Time(k), (i + j - 5));
			}
		}
	}

	_Q = H;  // Only minumum snap term is considered here inf the Hessian matrix

	StackOptiDep();
}

void TrajectoryGenerator::StackOptiDep() {
	double num_f = 3;      // 4 + 4 : only start and target position has fixed derivatives
	double num_p = 3 * m;  // All other derivatives are free
	double num_d = 6 * m;

	MatrixXd Ct;
	Ct       = MatrixXd::Zero(num_d, num_f + num_p);
	Ct(0, 0) = 1;
	Ct(2, 1) = 1;
	Ct(4, 2) = 1;  // Stack the start point

	Ct(6 * (m - 1) + 1, 3) = 1;  // Stack the end point
	Ct(6 * (m - 1) + 3, 4) = 1;
	Ct(6 * (m - 1) + 5, 5) = 1;

	if (m > 1) {
		Ct(1, 6)                       = 1;
		Ct(3, 7)                       = 1;
		Ct(5, 8)                       = 1;
		Ct(6 * (m - 1) + 0, 3 * m + 0) = 1;
		Ct(6 * (m - 1) + 2, 3 * m + 1) = 1;
		Ct(6 * (m - 1) + 4, 3 * m + 2) = 1;
		for (int j = 2; j < m; j++) {
			Ct(6 * (j - 1) + 0, 6 + 3 * (j - 2) + 0) = 1;
			Ct(6 * (j - 1) + 1, 6 + 3 * (j - 1) + 0) = 1;
			Ct(6 * (j - 1) + 2, 6 + 3 * (j - 2) + 1) = 1;
			Ct(6 * (j - 1) + 3, 6 + 3 * (j - 1) + 1) = 1;
			Ct(6 * (j - 1) + 4, 6 + 3 * (j - 2) + 2) = 1;
			Ct(6 * (j - 1) + 5, 6 + 3 * (j - 1) + 2) = 1;
		}
	}

	_C = Ct.transpose();
	_L = _A.inverse() * Ct;

	MatrixXd B = _A.inverse();
	_R         = _C * B.transpose() * _Q * (_A.inverse()) * Ct;

	_Rff.resize(3, 3);
	_Rfp.resize(3, 3 * m);
	_Rpf.resize(3 * m, 3);
	_Rpp.resize(3 * m, 3 * m);

	_Rff = _R.block(0, 0, 3, 3);
	_Rfp = _R.block(0, 3, 3, 3 * m);
	_Rpf = _R.block(3, 0, 3 * m, 3);
	_Rpp = _R.block(3, 3, 3 * m, 3 * m);
}

Eigen::MatrixXd TrajectoryGenerator::getA() { return _A; }
Eigen::MatrixXd TrajectoryGenerator::getQ() { return _Q; }
Eigen::MatrixXd TrajectoryGenerator::getC() { return _C; }
Eigen::MatrixXd TrajectoryGenerator::getL() { return _L; }
Eigen::MatrixXd TrajectoryGenerator::getR() { return _R; }

Eigen::MatrixXd TrajectoryGenerator::getRff() { return _Rff; }
Eigen::MatrixXd TrajectoryGenerator::getRpp() { return _Rpp; }
Eigen::MatrixXd TrajectoryGenerator::getRpf() { return _Rpf; }
Eigen::MatrixXd TrajectoryGenerator::getRfp() { return _Rfp; }
