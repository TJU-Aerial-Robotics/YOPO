#include "flightlib/grad_traj_optimization/grad_traj_optimizer.h"

GradTrajOptimizer::GradTrajOptimizer(const YAML::Node &cfg) {
	//-------------------------get parameter from server--------------------
	this->w_smooth    = cfg["ws"].as<double>();
	this->w_goal      = cfg["wg"].as<double>();
	this->w_long      = cfg["wl"].as<double>();
	this->w_vel       = cfg["wv"].as<double>();
	this->w_acc       = cfg["wa"].as<double>();
	this->w_collision = cfg["wc"].as<double>();

	this->alpha  = cfg["alpha"].as<double>();
	this->d0     = cfg["d0"].as<double>();
	this->r      = cfg["r"].as<double>();
	this->alphav = cfg["alphav"].as<double>();
	this->v0     = cfg["v0"].as<double>();
	this->rv     = cfg["rv"].as<double>();
	this->alphaa = cfg["alphaa"].as<double>();
	this->a0     = cfg["a0"].as<double>();
	this->ra     = cfg["ra"].as<double>();

	this->sgm_time = 2 * cfg["radio_range"].as<double>() / cfg["vel_max"].as<double>();

	//------------------------generate optimization dependency------------------
	Time    = Eigen::VectorXd::Zero(1);
	Time(0) = sgm_time;

	TrajectoryGenerator generator;
	generator.QPGeneration(Time);
	R   = generator.getR();
	Rff = generator.getRff();
	Rpp = generator.getRpp();
	Rpf = generator.getRpf();
	Rfp = generator.getRfp();
	L   = generator.getL();
	A   = generator.getA();
	C   = generator.getC();

	int m = Time.size();                      // number of segments in the trajectory
	Dp    = Eigen::MatrixXd::Zero(3, m * 3);  // optimized x_pva, y_pva, z_pva (end state)
	Df    = Eigen::MatrixXd::Zero(3, m * 3);  // fixed x_pva, y_pva, z_pva (init state)

	V = Eigen::MatrixXd::Zero(6, 6);
	for (int i = 0; i < 5; ++i)
		V(i, i + 1) = i + 1;

	num_dp    = Dp.cols();
	num_df    = Df.cols();
	num_point = Time.rows() + 1;
	boundary  = Eigen::VectorXd::Zero(6);
}

void GradTrajOptimizer::setSignedDistanceField(std::shared_ptr<sdf_tools::SignedDistanceField> s, double res) {
	this->sdf        = s;
	this->resolution = res;
}

void GradTrajOptimizer::constrains(double &n, double min, double max) const {
	if (n > max)
		n = max;
	else if (n < min)
		n = min;
}

void GradTrajOptimizer::setGoal(Eigen::Vector3d goal) { this->goal = goal; }

void GradTrajOptimizer::setBoundary(Eigen::Vector3d min, Eigen::Vector3d max) {
	this->map_boundary_min = min;
	this->map_boundary_max = max;
	boundary(0)            = map_boundary_min(0);
	boundary(1)            = map_boundary_max(0);
	boundary(2)            = map_boundary_min(1);
	boundary(3)            = map_boundary_max(1);
	boundary(4)            = map_boundary_min(2);
	boundary(5)            = map_boundary_max(2);
}


void GradTrajOptimizer::getCoefficientFromDerivative(Eigen::MatrixXd &coefficient, const std::vector<double> &_dp) const {
	coefficient.resize(num_point - 1, 18);

	for (int i = 0; i < 3; ++i) {
		//-----------------------merge df and dp -> d(df,dp)-----------------------
		Eigen::VectorXd df(num_df);
		Eigen::VectorXd dp(num_dp);
		Eigen::VectorXd d(num_df + num_dp);

		df = Df.row(i);
		for (int j = 0; j < num_dp; j++) {
			dp(j) = _dp[j + num_dp * i];
		}

		d.segment(0, 3)      = df;
		d.segment(3, num_dp) = dp;

		// ----------------------convert derivative to coefficient------------------
		Eigen::VectorXd coe(6 * (num_point - 1));
		coe = L * d;

		for (int j = 0; j < (num_point - 1); j++) {
			coefficient.block(j, 6 * i, 1, 6) = coe.segment(6 * j, 6).transpose();
		}
	}
}

void GradTrajOptimizer::getCostAndGradient(const std::vector<double> &df, const std::vector<double> &dp, double &cost,
                                           std::vector<double> &_grad) const {
	cost               = 0;
	double cost_smooth = 0;
	double cost_colli  = 0;
	double cost_goal   = 0;
	double cost_long   = 0;
	double cost_vel    = 0;  // deprecated
	double cost_acc    = 0;  // deprecated

	Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(3, num_dp);
	Eigen::MatrixXd g_smooth = Eigen::MatrixXd::Zero(3, num_dp);
	Eigen::MatrixXd g_colli  = Eigen::MatrixXd::Zero(3, num_dp);
	Eigen::MatrixXd g_goal   = Eigen::MatrixXd::Zero(3, num_dp);
	Eigen::MatrixXd g_long   = Eigen::MatrixXd::Zero(3, num_dp);
	Eigen::MatrixXd g_vel    = Eigen::MatrixXd::Zero(3, num_dp);  // deprecated
	Eigen::MatrixXd g_acc    = Eigen::MatrixXd::Zero(3, num_dp);  // deprecated

	for (int i = 0; i < num_df; ++i) {
		Df(0, i) = df[i];
		Df(1, i) = df[i + num_df];
		Df(2, i) = df[i + 2 * num_df];
	}

	// ------------------------- 1. get smoothness cost -----------------------------
	{
		// 平滑度的Cost，基于MinimalSnap，有：Js = d * R * d，其中d = [dF,dP]
		Eigen::VectorXd dfx = Df.block(0, 0, 1, 3).transpose();
		Eigen::VectorXd dfy = Df.block(1, 0, 1, 3).transpose();
		Eigen::VectorXd dfz = Df.block(2, 0, 1, 3).transpose();

		Eigen::VectorXd dpx = Eigen::VectorXd::Zero(num_dp);
		Eigen::VectorXd dpy = Eigen::VectorXd::Zero(num_dp);
		Eigen::VectorXd dpz = Eigen::VectorXd::Zero(num_dp);

		for (int i = 0; i < num_dp; ++i) {
			dpx(i) = dp[i];
			dpy(i) = dp[i + num_dp];
			dpz(i) = dp[i + 2 * num_dp];
		}

		Eigen::VectorXd dx    = Eigen::VectorXd::Zero(num_dp + num_df);
		Eigen::VectorXd dy    = Eigen::VectorXd::Zero(num_dp + num_df);
		Eigen::VectorXd dz    = Eigen::VectorXd::Zero(num_dp + num_df);
		dx.segment(0, 3)      = dfx;
		dx.segment(3, num_dp) = dpx;
		dy.segment(0, 3)      = dfy;
		dy.segment(3, num_dp) = dpy;
		dz.segment(0, 3)      = dfz;
		dz.segment(3, num_dp) = dpz;

		// ------------------- 1.1 get smoothness cost,fs= d'Rd ---------------------
		cost_smooth = double(dx.transpose() * R * dx) + double(dy.transpose() * R * dy) + (dz.transpose() * R * dz);

		//-------------------- 1.2 get smoothness gradient --------------------------
		Eigen::MatrixXd gx_smooth = 2 * Rfp.transpose() * dfx + 2 * Rpp * dpx;
		Eigen::MatrixXd gy_smooth = 2 * Rfp.transpose() * dfy + 2 * Rpp * dpy;
		Eigen::MatrixXd gz_smooth = 2 * Rfp.transpose() * dfz + 2 * Rpp * dpz;

		g_smooth.row(0) = gx_smooth.transpose();
		g_smooth.row(1) = gy_smooth.transpose();
		g_smooth.row(2) = gz_smooth.transpose();
	}

	// -------------------------- 2. get collision cost -----------------------------
	{
		Eigen::MatrixXd coe;
		getCoefficientFromDerivative(coe, dp);

		Eigen::MatrixXd Ldp(6, num_dp);

		// only single-segment polynomial here
		for (int s = 0; s < Time.size(); s++) {
			Ldp = L.block(6 * s, 3, 6, num_dp);

			// discrete time step
			double dt = Time(s) / 30.0;
			for (double t = 1e-3; t < Time(s); t += dt) {
				// get position, velocity
				Eigen::Vector3d pos, vel;
				getPositionFromCoeff(pos, coe, s, t);
				getVelocityFromCoeff(vel, coe, s, t);

				// get information from signed distance field
				double dist = 0, gd = 0, cd = 0;
				Eigen::Vector3d grad;
				getDistanceAndGradient(pos, dist, grad);  // 在sdf地图中的距离障碍物dist和grad（梯度方向）
				getDistancePenalty(dist, cd);             // 计算障碍物距离惩罚cost
				getDistancePenaltyGradient(dist, gd);     // 计算障碍物距离惩罚的梯度值
				// time Matrix T
				Eigen::MatrixXd T(1, 6);
				getTimeMatrix(t, T);

				// ------------------------ 2.1 collision cost-------------------------
				cost_colli += cd * dt;  // 碰撞cost = 障碍物距离惩罚c * 速度norm * 时间间隔

				// ------------------ 2.2 gradient of collision cost-------------------
				for (int k = 0; k < 3; k++) {  // 每一行对应xyz三个轴，一行的各列对应具体轴上对p,v,a的梯度
					g_colli.row(k) = g_colli.row(k) + (gd * grad(k) * T * Ldp) * dt;
				}

				// ---------- 3. Deprecated: get velocity and accleration cost --------
				if (0) {
					double cv = 0, ca = 0, gv = 0, ga = 0;
					Eigen::Vector3d acc;
					getAccelerationFromCoeff(acc, coe, s, t);

					for (int k = 0; k < 3; k++) {
						getVelocityPenalty(vel(k), cv);
						cost_vel += cv * dt;
						getAccelerationPenalty(acc(k), ca);
						cost_acc += ca * dt;
					}

					for (int k = 0; k < 3; k++) {
						getVelocityPenaltyGradient(vel(k), gv);
						g_vel.row(k) = g_vel.row(k) + (gv * T * V * Ldp) * dt;
						getAccelerationPenaltyGradient(acc(k), ga);
						g_acc.row(k) = g_acc.row(k) + (ga * T * V * V * Ldp) * dt;
					}
				}
			}
		}
	}

	// ---------------------------- 4. get goal cost ---------------------------------
	// 4.1 make the trajectry longer
	Eigen::Vector3d start_pos(df[0], df[num_dp], df[2 * num_dp]);
	Eigen::Vector3d end_pos(dp[0], dp[num_dp], dp[2 * num_dp]);
	Eigen::Vector3d delta_pos = end_pos - start_pos;
	double goal_r             = 100;  // param can be moved to config
	cost_long                 = exp(-(delta_pos(0) * delta_pos(0) + delta_pos(1) * delta_pos(1)) / goal_r);
	g_long(0, 0)              = -2 / goal_r * delta_pos(0) * cost_long;
	g_long(1, 0)              = -2 / goal_r * delta_pos(1) * cost_long;

	// 4.2 make the trajectry approach the goal
	cost_goal    = (end_pos - this->goal).norm() * (end_pos - this->goal).norm();
	g_goal(0, 0) = dp[0] - this->goal(0);
	g_goal(1, 0) = dp[num_dp] - this->goal(1);
	g_goal(2, 0) = dp[2 * num_dp] - this->goal(2);
	g_goal       = 2 * g_goal;

	//------------------------ sum up all cost and gradient -----------------------------
	double ws = this->w_smooth, wc = this->w_collision, wg = this->w_goal, wv = this->w_vel, wa = this->w_acc, wl = this->w_long;
	cost     = ws * cost_smooth + wc * cost_colli + wv * cost_vel + wa * cost_acc + wg * cost_goal + wl * cost_long + 1e-3;
	gradient = ws * g_smooth + wc * g_colli + wg * g_goal + wv * g_vel + wa * g_acc + wl * g_long;

	// gradient: 3x3 每一行对应xyz三个轴，一行的各列对应具体轴上对p,v,a的梯度
	_grad.resize(num_dp * 3);
	for (int i = 0; i < num_dp; ++i) {
		_grad[i]              = gradient(0, i);
		_grad[i + num_dp]     = gradient(1, i);
		_grad[i + 2 * num_dp] = gradient(2, i);
	}

	// cout << "smooth cost:" << ws * cost_smooth << " collision cost:" << wc * cost_colli << " goal cost:" << wg * cost_goal << endl;
	// cout << "smooth grad:" << ws * g_smooth(0) << " collision grad:" << wc * g_colli(0) << " goal grad:" << wg * g_goal(0) << endl;
}

// get position from coefficient
void GradTrajOptimizer::getPositionFromCoeff(Eigen::Vector3d &pos, const Eigen::MatrixXd &coeff, const int &index, const double &time) const {
	int s    = index;
	double t = time;
	float x  = coeff(s, 0) + coeff(s, 1) * t + coeff(s, 2) * pow(t, 2) + coeff(s, 3) * pow(t, 3) + coeff(s, 4) * pow(t, 4) + coeff(s, 5) * pow(t, 5);
	float y = coeff(s, 6) + coeff(s, 7) * t + coeff(s, 8) * pow(t, 2) + coeff(s, 9) * pow(t, 3) + coeff(s, 10) * pow(t, 4) + coeff(s, 11) * pow(t, 5);
	float z =
	    coeff(s, 12) + coeff(s, 13) * t + coeff(s, 14) * pow(t, 2) + coeff(s, 15) * pow(t, 3) + coeff(s, 16) * pow(t, 4) + coeff(s, 17) * pow(t, 5);

	pos(0) = x;
	pos(1) = y;
	pos(2) = z;
}

// get velocity from cofficient
void GradTrajOptimizer::getVelocityFromCoeff(Eigen::Vector3d &vel, const Eigen::MatrixXd &coeff, const int &index, const double &time) const {
	int s    = index;
	double t = time;
	float vx = coeff(s, 1) + 2 * coeff(s, 2) * pow(t, 1) + 3 * coeff(s, 3) * pow(t, 2) + 4 * coeff(s, 4) * pow(t, 3) + 5 * coeff(s, 5) * pow(t, 4);
	float vy = coeff(s, 7) + 2 * coeff(s, 8) * pow(t, 1) + 3 * coeff(s, 9) * pow(t, 2) + 4 * coeff(s, 10) * pow(t, 3) + 5 * coeff(s, 11) * pow(t, 4);
	float vz =
	    coeff(s, 13) + 2 * coeff(s, 14) * pow(t, 1) + 3 * coeff(s, 15) * pow(t, 2) + 4 * coeff(s, 16) * pow(t, 3) + 5 * coeff(s, 17) * pow(t, 4);

	vel(0) = vx;
	vel(1) = vy;
	vel(2) = vz;
}

// get acceleration from coefficient
void GradTrajOptimizer::getAccelerationFromCoeff(Eigen::Vector3d &acc, const Eigen::MatrixXd &coeff, const int &index, const double &time) const {
	int s    = index;
	double t = time;
	float ax = 2 * coeff(s, 2) + 6 * coeff(s, 3) * pow(t, 1) + 12 * coeff(s, 4) * pow(t, 2) + 20 * coeff(s, 5) * pow(t, 3);
	float ay = 2 * coeff(s, 8) + 6 * coeff(s, 9) * pow(t, 1) + 12 * coeff(s, 10) * pow(t, 2) + 20 * coeff(s, 11) * pow(t, 3);
	float az = 2 * coeff(s, 14) + 6 * coeff(s, 15) * pow(t, 1) + 12 * coeff(s, 16) * pow(t, 2) + 20 * coeff(s, 17) * pow(t, 3);

	acc(0) = ax;
	acc(1) = ay;
	acc(2) = az;
}

inline void GradTrajOptimizer::getDistancePenalty(const double &d, double &cost) const { cost = this->alpha * exp(-(d - this->d0) / this->r); }

inline void GradTrajOptimizer::getDistancePenaltyGradient(const double &d, double &grad) const {
	grad = -(this->alpha / this->r) * exp(-(d - this->d0) / this->r);
}

inline void GradTrajOptimizer::getVelocityPenalty(const double &v, double &cost) const { cost = alphav * exp((abs(v) - v0) / rv); }

inline void GradTrajOptimizer::getVelocityPenaltyGradient(const double &v, double &grad) const { grad = (alphav / rv) * exp((abs(v) - v0) / rv); }

inline void GradTrajOptimizer::getAccelerationPenalty(const double &a, double &cost) const { cost = alphaa * exp((abs(a) - a0) / ra); }

inline void GradTrajOptimizer::getAccelerationPenaltyGradient(const double &a, double &grad) const { grad = (alphaa / ra) * exp((abs(a) - a0) / ra); }

// get distance in signed distance field ,by position query
void GradTrajOptimizer::getDistanceAndGradient(Eigen::Vector3d &pos, double &dist, Eigen::Vector3d &grad) const {
	// get sdf directly from sdf_tools
	Eigen::Vector3d ori_pos = pos;
	// 1、限定在地图边界内 2、后面越界的惩罚回来
	constrains(pos(0), map_boundary_min(0), map_boundary_max(0));
	constrains(pos(1), map_boundary_min(1), map_boundary_max(1));
	constrains(pos(2), map_boundary_min(2), map_boundary_max(2));
	std::vector<double> location_gradient_query = this->sdf->GetGradient(pos(0), pos(1), pos(2), true);
	grad(0)                                     = location_gradient_query[0];
	grad(1)                                     = location_gradient_query[1];
	grad(2)                                     = location_gradient_query[2];
	std::pair<float, bool> location_sdf_query   = this->sdf->GetSafe(pos(0), pos(1), pos(2));
	dist                                        = location_sdf_query.first;

	// update distance and gradient using boundary
	double dtb = getDistanceToBoundary(ori_pos(0), ori_pos(1), ori_pos(2));
	// 1. 点在边界内时：把点推向内部; 2. 如果在Boundary外: (dtb<0)梯度是指向Boundary的,同样推向内部
	if (dtb < dist) {
		dist = dtb;
		recaluculateGradient(ori_pos(0), ori_pos(1), ori_pos(2), grad);
	}
}

double GradTrajOptimizer::getDistanceToBoundary(const double &x, const double &y, const double &z) const {
	double dist_x = std::min(x - boundary(0), boundary(1) - x);
	double dist_y = std::min(y - boundary(2), boundary(3) - y);
	double dist_z = std::min(z - boundary(4), boundary(5) - z);
	double dtb    = std::min(dist_x, dist_y);
	dtb           = std::min(dtb, dist_z);

	return dtb;
}

void GradTrajOptimizer::recaluculateGradient(const double &x, const double &y, const double &z, Eigen::Vector3d &grad) const {
	double r = this->resolution;

	grad(0) = (10 * (GDTB(x + r, y, z) - GDTB(x - r, y, z)) + 3 * (GDTB(x + r, y + r, z) - GDTB(x - r, y + r, z)) +
	           3 * (GDTB(x + r, y - r, z) - GDTB(x - r, y - r, z))) /
	          (32 * r);
	grad(1) = (10 * (GDTB(x, y + r, z) - GDTB(x, y - r, z)) + 3 * (GDTB(x + r, y + r, z) - GDTB(x + r, y - r, z)) +
	           3 * (GDTB(x - r, y + r, z) - GDTB(x - r, y - r, z))) /
	          (32 * r);
	grad(2) = (10 * (GDTB(x, y, z + r) - GDTB(x, y, z - r)) + 3 * (GDTB(x, y + r, z + r) - GDTB(x, y + r, z - r)) +
	           3 * (GDTB(x, y - r, z + r) - GDTB(x, y - r, z - r))) /
	          (32 * r);
}

void GradTrajOptimizer::getTimeMatrix(const double &t, Eigen::MatrixXd &T) const {
	T.resize(1, 6);
	T.setZero();

	for (int i = 0; i < 6; ++i) {
		T(0, i) = pow(t, i);
	}
}