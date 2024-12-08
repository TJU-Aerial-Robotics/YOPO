// A very rough evaluation: Input the map point cloud, receive odometry, and calculate: execution time,
// trajectory length, distance to the nearest obstacle, and dynamic cost (logging method to be optimized).
#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <ros/ros.h>
#include <time.h>
#include <Eigen/Core>
#include "flightlib/controller/ctrl_ref.h"

namespace traj_eval {

// Evaluate the performance of the trajectory from the origin [start_record, y, z] to destination [finish_record, y, z]
float start_record  = -39;
float finish_record = 18;
float ctrl_dt = 0.02;
// eval
std::ofstream dist_log, ctrl_log;
Eigen::Vector3f pose_last;
Eigen::Vector3f acc_last;
ros::Time start, end;
bool odom_init   = false;
bool odom_finish = false;
bool first_cmd   = true;
float length_    = 0;
float dist_      = 0;
float ctrl_cost_ = 0;
int num_         = 0;
// map
pcl::search::KdTree<pcl::PointXYZ> kdtree;
float resolution = 0.2;
Eigen::Vector3i m_size;
Eigen::Vector3i m_origin;

void map_input() {
	std::string ply_path_ = getenv("FLIGHTMARE_PATH") + std::string("/flightrender/RPG_Flightmare/pointcloud_data/pointcloud-0.ply");
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path_, *cloud);

	pcl::PointXYZ min, max;
	pcl::getMinMax3D(*cloud, min, max);
	m_size(0) = ceil((max.x - min.x) / resolution);
	m_size(1) = ceil((max.y - min.y) / resolution);
	m_size(2) = ceil((max.z - min.z + 0.01) / resolution);
	m_origin  = Eigen::Vector3i(min.x, min.y, min.z);
	kdtree.setInputCloud(cloud);
}

int to_id(int x, int y, int z) { return x * m_size(1) * m_size(2) + y * m_size(2) + z; }

float distance_at(Eigen::Vector3f pose) {
	pcl::PointXYZ drone_;
	drone_.x = pose(0);
	drone_.y = pose(1);
	drone_.z = pose(2);

	int K = 1;
	std::vector<int> indices(K);
	std::vector<float> distances(K);  // 存储近邻点对应距离的平方
	kdtree.nearestKSearch(drone_, K, indices, distances);
	return std::sqrt(distances[0]);
}

void odom_cb(const nav_msgs::Odometry::Ptr odom_msg) {
	if (!odom_init && odom_msg->pose.pose.position.x > start_record) {
		odom_init = true;
		pose_last = Eigen::Vector3f(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
		start     = odom_msg->header.stamp;
		std::cout << "start!" << std::endl;
		return;
	}
	if (!odom_init || odom_finish)
		return;

	Eigen::Vector3f pose_cur(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
	length_ += (pose_cur - pose_last).norm();
	dist_ += distance_at(pose_cur);

	dist_log << (odom_msg->header.stamp - start).toSec() << ',';
	dist_log << pose_cur(0) << ',';
	dist_log << distance_at(pose_cur) << std::endl;

	pose_last = pose_cur;
	num_++;
	if (odom_msg->pose.pose.position.x > finish_record) {
		odom_finish = true;
		end         = odom_msg->header.stamp;
		std::cout << "finish! \ntime:" << (end - start).toSec() << " s,\nlength:" << length_ << " m,\ndist:" << dist_ / num_
		          << " m,\nctrl cost:" << ctrl_cost_ << " m2/s5" << std::endl;
		dist_log.close();
		ctrl_log.close();
	}
}

void ctrl_cb(const quad_pos_ctrl::ctrl_ref& ctrl_msg) {
	if (!odom_init || odom_finish)
		return;
	if (first_cmd) {
		first_cmd = false;
		acc_last  = Eigen::Vector3f(ctrl_msg.acc_ref[0], -ctrl_msg.acc_ref[1], -ctrl_msg.acc_ref[2]);
		return;
	}

	Eigen::Vector3f cur_acc(ctrl_msg.acc_ref[0], -ctrl_msg.acc_ref[1], -ctrl_msg.acc_ref[2]);
	Eigen::Vector3f d_acc = (acc_last - cur_acc) / ctrl_dt;
	float acc_norm2       = d_acc.dot(d_acc);
	ctrl_cost_ += ctrl_dt * acc_norm2;
	acc_last = cur_acc;

	ctrl_log << ctrl_msg.pos_ref[0] << ',';
	ctrl_log << ctrl_msg.pos_ref[1] << ',';
	ctrl_log << ctrl_msg.pos_ref[2] << ',';

	ctrl_log << ctrl_msg.vel_ref[0] << ',';
	ctrl_log << ctrl_msg.vel_ref[1] << ',';
	ctrl_log << ctrl_msg.vel_ref[2] << ',';

	ctrl_log << ctrl_msg.acc_ref[0] << ',';
	ctrl_log << ctrl_msg.acc_ref[1] << ',';
	ctrl_log << ctrl_msg.acc_ref[2] << std::endl;
}

}  // namespace traj_eval

using namespace traj_eval;
int main(int argc, char** argv) {
	map_input();

	std::string log_file = getenv("FLIGHTMARE_PATH") + std::string("/run/utils/dist_log.csv");
	std::cout << "log path:" << log_file << std::endl;
	dist_log.open(log_file.c_str(), std::ios::out);

	std::string log_file2 = getenv("FLIGHTMARE_PATH") + std::string("/run/utils/ctrl_log.csv");
	ctrl_log.open(log_file2.c_str(), std::ios::out);

	ros::init(argc, argv, "traj_eval");
	ros::NodeHandle nh;
	ros::Subscriber odom_sub = nh.subscribe("/juliett/ground_truth/odom", 1, odom_cb);
	ros::Subscriber ctrl_sub = nh.subscribe("/juliett_pos_ctrl_node/controller/ctrl_ref", 1, ctrl_cb);
	ros::spin();
}