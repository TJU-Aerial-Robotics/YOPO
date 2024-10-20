/*
    This code is modified from https://github.com/HKUST-Aerial-Robotics/grad_traj_optimization, thanks to their excellent work.
*/

#ifndef _GRAD_TRAJ_OPTIMIZER_H_
#define _GRAD_TRAJ_OPTIMIZER_H_

#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>

#include "flightlib/grad_traj_optimization/qp_generator.h"
#include "sdf_tools/collision_map.hpp"
#include "sdf_tools/sdf.hpp"

#define GDTB getDistanceToBoundary

using namespace std;

class GradTrajOptimizer {
   public:
	GradTrajOptimizer(const YAML::Node &cfg);

	void getCoefficient(Eigen::MatrixXd &coeff) { coeff = this->coeff; };

	double getCost() { return this->min_f; }

	void getSegmentTime(Eigen::VectorXd &T) { T = this->Time; }

	void setSignedDistanceField(std::shared_ptr<sdf_tools::SignedDistanceField> s, double res);

	void setGoal(Eigen::Vector3d goal);

	void setBoundary(Eigen::Vector3d min, Eigen::Vector3d max);

	void getCostAndGradient(const std::vector<double> &df, const std::vector<double> &dp, double &cost, std::vector<double> &grad) const;

	/** convert derivatives of end points to polynomials coefficient */
	void getCoefficientFromDerivative(Eigen::MatrixXd &coeff, const std::vector<double> &_dp) const;

   private:
	/** signed distance field */
	double resolution;
	Eigen::Vector3d map_boundary_min, map_boundary_max;
	std::shared_ptr<sdf_tools::SignedDistanceField> sdf;
	mutable Eigen::VectorXd boundary;  // min x max x... min z,max z

	/** coefficient of polynomials*/
	mutable Eigen::MatrixXd coeff;

	/** important matrix and variables*/
	Eigen::MatrixXd A;
	Eigen::MatrixXd C;
	Eigen::MatrixXd L;
	Eigen::MatrixXd R;
	Eigen::MatrixXd Rff;
	Eigen::MatrixXd Rpp;
	Eigen::MatrixXd Rpf;
	Eigen::MatrixXd Rfp;
	Eigen::VectorXd Time;
	Eigen::MatrixXd V;
	mutable Eigen::MatrixXd Df;
	Eigen::MatrixXd Dp;
	Eigen::MatrixXd path;
	Eigen::Vector3d goal;

	/** tractory params */
	double sgm_time;
	int num_dp;
	int num_df;
	int num_point;
	double min_f;

	// weight of cost
	double w_smooth;
	double w_collision;
	double w_vel;
	double w_acc;
	double w_goal;
	double w_long;

	// params of cost
	double d0;
	double r;
	double alpha;

	double v0;
	double rv;
	double alphav;

	double a0;
	double ra;
	double alphaa;

	/** get distance and gradient in signed distance field ,by position query*/
	void getDistanceAndGradient(Eigen::Vector3d &pos, double &dist, Eigen::Vector3d &grad) const;
	void getPositionFromCoeff(Eigen::Vector3d &pos, const Eigen::MatrixXd &coeff, const int &index, const double &time) const;
	void getVelocityFromCoeff(Eigen::Vector3d &vel, const Eigen::MatrixXd &coeff, const int &index, const double &time) const;
	void getAccelerationFromCoeff(Eigen::Vector3d &acc, const Eigen::MatrixXd &coeff, const int &index, const double &time) const;

	/** penalty and gradient */
	void getDistancePenalty(const double &distance, double &cost) const;
	void getDistancePenaltyGradient(const double &distance, double &grad) const;
	void getVelocityPenalty(const double &distance, double &cost) const;
	void getVelocityPenaltyGradient(const double &vel, double &grad) const;
	void getAccelerationPenalty(const double &distance, double &cost) const;
	void getAccelerationPenaltyGradient(const double &acc, double &grad) const;

	void getTimeMatrix(const double &t, Eigen::MatrixXd &T) const;
	void constrains(double &n, double min, double max) const;
	double getDistanceToBoundary(const double &x, const double &y, const double &z) const;
	void recaluculateGradient(const double &x, const double &y, const double &z, Eigen ::Vector3d &grad) const;
};

#endif