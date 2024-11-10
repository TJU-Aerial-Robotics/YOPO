#include <nav_msgs/Odometry.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>

#include "flightlib/controller/PositionCommand.h"
#include "flightlib/controller/ctrl_ref.h"
#include "flightlib/grad_traj_optimization/opt_utile.h"
#include "flightlib/grad_traj_optimization/traj_optimization_bridge.h"

namespace yopo_net {

nav_msgs::Odometry odom_msg;
quad_pos_ctrl::ctrl_ref ctrl_ref_last;
quadrotor_msgs::PositionCommand pos_cmd_last;
bool odom_init     = false;
bool odom_ref_init = false;
bool yopo_init     = false;
bool arrive        = false;
float traj_time    = 2.0;
float sample_t     = 0.0;
float last_yaw_    = 0;  // NWU

TrajOptimizationBridge* traj_opt_bridge;
TrajOptimizationBridge* traj_opt_bridge_for_vis;
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lattice_nodes;

Eigen::Vector3d goal_(100, 0, 2);
Eigen::Quaterniond quat_(1, 0, 0, 0);
Eigen::Vector3d last_pos_(0, 0, 0), last_vel_(0, 0, 0), last_acc_(0, 0, 0);

ros::Publisher trajs_visual_pub, best_traj_visual_pub, state_ref_pub, our_ctrl_pub, so3_ctrl_pub, lattice_trajs_visual_pub;
ros::Subscriber odom_sub, odom_ref_sub, yopo_best_sub, yopo_all_sub, goal_sub;

void odom_cb(const nav_msgs::Odometry::Ptr msg) {
	odom_msg  = *msg;
	odom_init = true;
	quat_.w() = odom_msg.pose.pose.orientation.w;
	quat_.x() = odom_msg.pose.pose.orientation.x;
	quat_.y() = odom_msg.pose.pose.orientation.y;
	quat_.z() = odom_msg.pose.pose.orientation.z;
	if (!odom_ref_init) {
		last_pos_ << odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z;
		last_vel_ << 0, 0, 0;
		last_acc_ << 0, 0, 0;
	}

	// check if reach the goal
	Eigen::Vector3d dist(odom_msg.pose.pose.position.x - goal_(0), odom_msg.pose.pose.position.y - goal_(1),
	                     odom_msg.pose.pose.position.z - goal_(2));
	if (dist.norm() < 4 && !arrive) {
		printf("Arrive!\n");
		arrive = true;
	}
}

void goal_cb(const std_msgs::Float32MultiArray::Ptr msg) {
	Eigen::Vector3d last_goal = goal_;
	goal_(0)                  = msg->data[0];
	goal_(1)                  = msg->data[1];
	goal_(2)                  = msg->data[2];
	if (last_goal != goal_)
		arrive = false;
}

void traj_to_pcl(TrajOptimizationBridge* traj_opt_bridge_input, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, double cost = 0.0) {
	for (float dt = 0.0; dt < traj_time; dt = dt + 0.05) {
		Eigen::Vector3d next_pos, next_vel, next_acc;
		traj_opt_bridge_input->getNextState(next_pos, next_vel, next_acc, dt);
		pcl::PointXYZI clrP;
		clrP.x         = next_pos(0);
		clrP.y         = next_pos(1);
		clrP.z         = next_pos(2);
		clrP.intensity = cost;
		cloud->points.push_back(clrP);
	}
}

void yopo_cb(const std_msgs::Float32MultiArray::ConstPtr msg) {
	if (!odom_init)
		return;
	std::vector<double> endstate;
	for (int i = 0; i < msg->data.size(); i++) {
		endstate.push_back(static_cast<double>(msg->data[i]));
	}

	traj_opt_bridge->setState(last_pos_.cast<double>(), quat_.cast<double>(), last_vel_.cast<double>(), last_acc_.cast<double>());
	traj_opt_bridge->solveBVP(endstate);
	// int milliseconds_yopo_pub = msg->layout.data_offset;  // 预测时采用的状态和图像的时间戳(ms)
	// int milliseconds_now = uint64_t(ros::Time::now().toSec() * 1000) % 1000000;
	// double delta_t = double(milliseconds_now - milliseconds_yopo_pub) / 1000;
	// std::cout<<"yopo开始预测到当前时间: "<<delta_t<<std::endl;
	sample_t = 0.0;  // = delta_t

	pcl::PointCloud<pcl::PointXYZI>::Ptr best_traj_cld(new pcl::PointCloud<pcl::PointXYZI>);
	traj_to_pcl(traj_opt_bridge, best_traj_cld);
	pcl_conversions::toPCL(ros::Time::now(), best_traj_cld->header.stamp);  // for test
	best_traj_cld->header.frame_id = "world";
	best_traj_visual_pub.publish(best_traj_cld);
	yopo_init = true;
}

void trajs_vis_cb(const std_msgs::Float32MultiArray::ConstPtr msg) {
	if (!odom_init)
		return;

	// ---------------- visualization of all trajs --------------------
	std::vector<std::vector<double>> endstates_b;
	endstates_b.resize(msg->layout.dim[0].size);
	std::vector<double> scores;
	for (int i = 0; i < msg->layout.dim[0].size; i++) {
		for (int j = 0; j < msg->layout.dim[1].size - 1; j++) {
			endstates_b[i].push_back(static_cast<double>(msg->data[i * msg->layout.dim[1].size + j]));
		}
		scores.push_back(static_cast<double>(msg->data[(i + 1) * msg->layout.dim[1].size - 1]));
	}

	traj_opt_bridge_for_vis->setState(last_pos_.cast<double>(), quat_.cast<double>(), last_vel_.cast<double>(), last_acc_.cast<double>());

	pcl::PointCloud<pcl::PointXYZI>::Ptr trajs_cld(new pcl::PointCloud<pcl::PointXYZI>);
	for (size_t i = 0; i < endstates_b.size(); i++) {
		traj_opt_bridge_for_vis->solveBVP(endstates_b[i]);
		traj_to_pcl(traj_opt_bridge_for_vis, trajs_cld, scores[i]);
	}
	pcl_conversions::toPCL(ros::Time::now(), trajs_cld->header.stamp);
	trajs_cld->header.frame_id = "world";
	trajs_visual_pub.publish(trajs_cld);

	// ---------------- visualization of lattice ------------------------
	pcl::PointCloud<pcl::PointXYZI>::Ptr lattice_trajs_cld(new pcl::PointCloud<pcl::PointXYZI>);
	Eigen::Vector3d pos_1(0.0, 0.0, 0.0), vel_1(0.0, 0.0, 0.0), acc_1(0.0, 0.0, 0.0);
	for (size_t i = 0; i < lattice_nodes.size(); i++) {
		pos_1                          = lattice_nodes[i].first;
		vel_1                          = lattice_nodes[i].second;
		std::vector<double> endstate_lattice = {pos_1(0), vel_1(0), acc_1(0), pos_1(1), vel_1(1), acc_1(1), pos_1(2), vel_1(2), acc_1(2)};
		traj_opt_bridge_for_vis->solveBVP(endstate_lattice);
		traj_to_pcl(traj_opt_bridge_for_vis, lattice_trajs_cld);
	}
	pcl_conversions::toPCL(ros::Time::now(), lattice_trajs_cld->header.stamp);
	lattice_trajs_cld->header.frame_id = "world";
	lattice_trajs_visual_pub.publish(lattice_trajs_cld);
}

std::pair<float, float> calculate_yaw(float sample_t, float dt) {
	constexpr float PI                  = 3.1415926;
	constexpr float YAW_DOT_MAX_PER_SEC = 0.3 * PI;
	std::pair<float, float> yaw_yawdot(0, 0);
	float yaw    = 0;
	float yawdot = 0;

	// dir of vel
	Eigen::Vector3d nxt_p, nxt_v, nxt_a;
	traj_opt_bridge->getNextState(nxt_p, nxt_v, nxt_a, sample_t);
	Eigen::Vector3d dir_vel = nxt_v / nxt_v.norm();
	// dir of goal
	Eigen::Vector3d dir_goal(goal_(0) - nxt_p(0), goal_(1) - nxt_p(1), goal_(2) - nxt_p(2));
	float goal_dist = dir_goal.norm();
	dir_goal        = dir_goal / goal_dist;
	// or just dir_des = dir_vel
	Eigen::Vector3d dir_des = dir_vel + dir_goal;

	float yaw_temp       = goal_dist > 0.2 ? atan2(dir_des(1), dir_des(0)) : last_yaw_;
	float max_yaw_change = YAW_DOT_MAX_PER_SEC * dt;

	if (yaw_temp - last_yaw_ > PI) {
		if (yaw_temp - last_yaw_ - 2 * PI < -max_yaw_change) {
			yaw = last_yaw_ - max_yaw_change;
			if (yaw < -PI)
				yaw += 2 * PI;
			yawdot = -YAW_DOT_MAX_PER_SEC;
		} else {
			yaw = yaw_temp;
			if (yaw - last_yaw_ > PI)
				yawdot = -YAW_DOT_MAX_PER_SEC;
			else
				yawdot = (yaw_temp - last_yaw_) / dt;
		}
	} else if (yaw_temp - last_yaw_ < -PI) {
		if (yaw_temp - last_yaw_ + 2 * PI > max_yaw_change) {
			yaw = last_yaw_ + max_yaw_change;
			if (yaw > PI)
				yaw -= 2 * PI;
			yawdot = YAW_DOT_MAX_PER_SEC;
		} else {
			yaw = yaw_temp;
			if (yaw - last_yaw_ < -PI)
				yawdot = YAW_DOT_MAX_PER_SEC;
			else
				yawdot = (yaw_temp - last_yaw_) / dt;
		}
	} else {
		if (yaw_temp - last_yaw_ < -max_yaw_change) {
			yaw = last_yaw_ - max_yaw_change;
			if (yaw < -PI)
				yaw += 2 * PI;
			yawdot = -YAW_DOT_MAX_PER_SEC;
		} else if (yaw_temp - last_yaw_ > max_yaw_change) {
			yaw = last_yaw_ + max_yaw_change;
			if (yaw > PI)
				yaw -= 2 * PI;
			yawdot = YAW_DOT_MAX_PER_SEC;
		} else {
			yaw = yaw_temp;
			if (yaw - last_yaw_ > PI)
				yawdot = -YAW_DOT_MAX_PER_SEC;
			else if (yaw - last_yaw_ < -PI)
				yawdot = YAW_DOT_MAX_PER_SEC;
			else
				yawdot = (yaw_temp - last_yaw_) / dt;
		}
	}

	last_yaw_         = yaw;
	yaw_yawdot.first  = yaw;
	yaw_yawdot.second = yawdot;
	return yaw_yawdot;
}

void ref_pub_cb(const ros::TimerEvent&) {
	if (!yopo_init)
		return;

	if (arrive) {
		odom_ref_init = false;
		// single state control for smoother performance
		ctrl_ref_last.header.stamp = ros::Time::now();
		ctrl_ref_last.vel_ref      = {0, 0, 0};
		ctrl_ref_last.acc_ref      = {0, 0, 0};
		ctrl_ref_last.ref_mask     = 1;
		our_ctrl_pub.publish(ctrl_ref_last);
		// larger position error, just for simpler demonstration
		pos_cmd_last.header.stamp  = ros::Time::now();
		// pos_cmd_last.velocity.x     = 0.0;
		// pos_cmd_last.velocity.y     = 0.0;
		// pos_cmd_last.velocity.z     = 0.0;
		// pos_cmd_last.acceleration.x = 0.0;
		// pos_cmd_last.acceleration.y = 0.0;
		// pos_cmd_last.acceleration.z = 0.0;
		// pos_cmd_last.yaw_dot        = 0.0;
		so3_ctrl_pub.publish(pos_cmd_last);
		return;
	}

	sample_t += 0.02;
	Eigen::Vector3d desired_p, desired_v, desired_a;
	traj_opt_bridge->getNextState(desired_p, desired_v, desired_a, sample_t);
	std::pair<float, float> yaw_yawdot(0, 0);
	yaw_yawdot = calculate_yaw(sample_t, 0.02);

	// Realworld & our PID Controller
	quad_pos_ctrl::ctrl_ref ctrl_msg;
	ctrl_msg.header.stamp = ros::Time::now();
	ctrl_msg.pos_ref      = {desired_p(0), -desired_p(1), -desired_p(2)};
	ctrl_msg.vel_ref      = {desired_v(0), -desired_v(1), -desired_v(2)};
	ctrl_msg.acc_ref      = {desired_a(0), -desired_a(1), -desired_a(2)};
	ctrl_msg.yaw_ref      = -yaw_yawdot.first;
	ctrl_msg.ref_mask     = 7;
	ctrl_ref_last         = ctrl_msg;
	our_ctrl_pub.publish(ctrl_msg);

	// SO3 Simulator & SO3 Controller
	quadrotor_msgs::PositionCommand cmd;
	cmd.header.frame_id = "world";
	cmd.header.stamp    = ros::Time::now();
	cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
	cmd.position.x      = desired_p(0);
	cmd.position.y      = desired_p(1);
	cmd.position.z      = desired_p(2);
	cmd.velocity.x      = desired_v(0);
	cmd.velocity.y      = desired_v(1);
	cmd.velocity.z      = desired_v(2);
	cmd.acceleration.x  = desired_a(0);
	cmd.acceleration.y  = desired_a(1);
	cmd.acceleration.z  = desired_a(2);
	cmd.yaw             = yaw_yawdot.first;
	cmd.yaw_dot         = yaw_yawdot.second;
	pos_cmd_last        = cmd;
	so3_ctrl_pub.publish(cmd);

	// update the desire state for next planning
	last_pos_ = desired_p;
	last_vel_ = desired_v;
	last_acc_ = desired_a;

	// for reference of yopo network
	nav_msgs::Odometry odom_;
	odom_.header.stamp          = ros::Time::now();
	odom_.pose.pose.position.x  = desired_p(0);
	odom_.pose.pose.position.y  = desired_p(1);
	odom_.pose.pose.position.z  = desired_p(2);
	odom_.twist.twist.linear.x  = desired_v(0);
	odom_.twist.twist.linear.y  = desired_v(1);
	odom_.twist.twist.linear.z  = desired_v(2);
	odom_.twist.twist.angular.x = desired_a(0);
	odom_.twist.twist.angular.y = desired_a(1);
	odom_.twist.twist.angular.z = desired_a(2);
	state_ref_pub.publish(odom_);
	odom_ref_init = true;
}

}  // namespace yopo_net

using namespace yopo_net;
int main(int argc, char** argv) {
	ros::init(argc, argv, "yopo_test");
	ros::NodeHandle nh;

	string cfg_path     = getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/traj_opt.yaml");
	YAML::Node cfg_     = YAML::LoadFile(cfg_path);
	traj_time           = 2 * cfg_["radio_range"].as<double>() / cfg_["vel_max"].as<double>();
	int horizon_num     = cfg_["horizon_num"].as<int>();
	int vertical_num    = cfg_["vertical_num"].as<int>();
	int vel_num         = cfg_["vel_num"].as<int>();
	int radio_num       = cfg_["radio_num"].as<int>();
	double horizon_fov  = cfg_["horizon_camera_fov"].as<double>() * (horizon_num - 1) / horizon_num;
	double vertical_fov = cfg_["vertical_camera_fov"].as<double>() * (vertical_num - 1) / vertical_num;
	double vel_fov      = cfg_["vel_fov"].as<double>();
	double radio_range  = cfg_["radio_range"].as<double>();
	double vel_prefile  = cfg_["vel_prefile"].as<double>();

	getLatticeGuiding(lattice_nodes, horizon_num, vertical_num, radio_num, vel_num, horizon_fov, vertical_fov, radio_range, vel_fov, vel_prefile);
	traj_opt_bridge         = new TrajOptimizationBridge();
	traj_opt_bridge_for_vis = new TrajOptimizationBridge();

	lattice_trajs_visual_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("/yopo_net/lattice_trajs_visual", 1);
	trajs_visual_pub         = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("/yopo_net/trajs_visual", 1);
	best_traj_visual_pub     = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("/yopo_net/best_traj_visual", 1);
	state_ref_pub            = nh.advertise<nav_msgs::Odometry>("/juliett/state_ref/odom", 10);

	// our PID Controller (realworld) & SO3 Controller (simulation)
	our_ctrl_pub = nh.advertise<quad_pos_ctrl::ctrl_ref>("/juliett_pos_ctrl_node/controller/ctrl_ref", 10);
	so3_ctrl_pub = nh.advertise<quadrotor_msgs::PositionCommand>("/so3_control/pos_cmd", 10);

	odom_sub      = nh.subscribe("/juliett/ground_truth/odom", 1, yopo_net::odom_cb, ros::TransportHints().tcpNoDelay());
	yopo_best_sub = nh.subscribe("/yopo_net/pred_endstate", 1, yopo_net::yopo_cb, ros::TransportHints().tcpNoDelay());
	yopo_all_sub  = nh.subscribe("/yopo_net/pred_endstates", 1, trajs_vis_cb);
	goal_sub      = nh.subscribe("/yopo_net/goal", 1, goal_cb);

	ros::Timer ref_timer = nh.createTimer(ros::Duration(0.02), ref_pub_cb);
	std::cout << "YOPO Planner Node OK!" << std::endl;
	ros::spin();
}