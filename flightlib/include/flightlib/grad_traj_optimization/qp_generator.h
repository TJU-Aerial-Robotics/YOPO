#ifndef _TRAJECTORY_GENERATOR_H_
#define _TRAJECTORY_GENERATOR_H_
#include <Eigen/Eigen>
#include <vector>


class TrajectoryGenerator {
   private:
	int m = 1;           // number of segments in the trajectory
	Eigen::MatrixXd _A;  // Mapping matrix
	Eigen::MatrixXd _Q;  // Hessian matrix
	Eigen::MatrixXd _C;  // Selection matrix
	Eigen::MatrixXd _L;  // A.inv() * C.transpose()

	Eigen::MatrixXd _R;
	Eigen::MatrixXd _Rff;
	Eigen::MatrixXd _Rpp;
	Eigen::MatrixXd _Rpf;
	Eigen::MatrixXd _Rfp;

	Eigen::VectorXd _Pxi;
	Eigen::VectorXd _Pyi;
	Eigen::VectorXd _Pzi;

	Eigen::VectorXd _Dx;
	Eigen::VectorXd _Dy;
	Eigen::VectorXd _Dz;

   public:
	Eigen::MatrixXd _Path;
	Eigen::VectorXd _Time;

	TrajectoryGenerator();

	~TrajectoryGenerator();

	void QPGeneration(const Eigen::VectorXd &Time);

	void StackOptiDep();  // Stack the optimization's dependency, the intermediate matrix and initial derivatives

	Eigen::MatrixXd getA();
	Eigen::MatrixXd getQ();
	Eigen::MatrixXd getC();
	Eigen::MatrixXd getL();

	Eigen::MatrixXd getR();
	Eigen::MatrixXd getRpp();
	Eigen::MatrixXd getRff();
	Eigen::MatrixXd getRfp();
	Eigen::MatrixXd getRpf();
};

#endif
