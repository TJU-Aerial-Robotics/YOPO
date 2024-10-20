// Publish the surrounding point cloud map based on the drone's position for visualization.
#include <nav_msgs/Odometry.h>
#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <time.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include "visualization_msgs/Marker.h"

namespace map_visual {

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

void pcl_input() {
	std::string cfg_path  = getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/quadrotor_ros.yaml");
	YAML::Node cfg_       = YAML::LoadFile(cfg_path);
	std::string ply_path_ = getenv("FLIGHTMARE_PATH") + cfg_["ply_path"].as<std::string>() + "pointcloud-0.ply";
	pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path_, *cloud);
	std::cout << "size of pointcloud: " << cloud->points.size() << std::endl;
}

void odom_cb(const nav_msgs::Odometry::ConstPtr odom_msg, ros::Publisher* local_map_pub, ros::Publisher* mesh_pub,
             tf::TransformBroadcaster* uav_tf_br) {
	// 1. publish tf
	tf::Transform transform;
	transform.setOrigin(tf::Vector3(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z));
	tf::Quaternion q(0, 0, 0, 1);
	// tf::Quaternion q(odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y,
	//                  odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w);
	transform.setRotation(q);
	uav_tf_br->sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "uav"));

	// 2. publish map
	Eigen::Vector3f pose_cur(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
	Eigen::Vector3f local_map_half_size(8, 8, 1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZ>);
	for (auto& x : cloud->points) {
		if (x.z > pose_cur(2) - 6 && x.z < pose_cur(2) + local_map_half_size(2) && x.x > pose_cur(0) - local_map_half_size(0) &&
		    x.x < pose_cur(0) + local_map_half_size(0) && x.y > pose_cur(1) - local_map_half_size(1) && x.y < pose_cur(1) + local_map_half_size(1)) {
			local_map->points.push_back(x);
		}
	}
	pcl_conversions::toPCL(ros::Time::now(), local_map->header.stamp);
	local_map->header.frame_id = "world";
	local_map_pub->publish(local_map);

	// 3. publish UAV model
	std::string mesh_resource = std::string("file://") + getenv("FLIGHTMARE_PATH") + std::string("/flightlib/src/ros_nodes/model/uav.dae");

	visualization_msgs::Marker meshROS;
	meshROS.header.frame_id = "world";
	meshROS.header.stamp    = ros::Time::now();
	meshROS.ns              = "mesh";
	meshROS.id              = 0;
	meshROS.type            = visualization_msgs::Marker::MESH_RESOURCE;
	meshROS.action          = visualization_msgs::Marker::ADD;

	meshROS.pose.position.x = odom_msg->pose.pose.position.x - 0.2;
	meshROS.pose.position.y = odom_msg->pose.pose.position.y;
	meshROS.pose.position.z = odom_msg->pose.pose.position.z;

	meshROS.pose.orientation.w = odom_msg->pose.pose.orientation.w;
	meshROS.pose.orientation.x = odom_msg->pose.pose.orientation.x;
	meshROS.pose.orientation.y = odom_msg->pose.pose.orientation.y;
	meshROS.pose.orientation.z = odom_msg->pose.pose.orientation.z;

	float scale     = 2;
	meshROS.scale.x = scale;
	meshROS.scale.y = scale;
	meshROS.scale.z = scale;
	meshROS.color.a = 1;
	meshROS.color.r = 1;
	meshROS.color.g = 1;
	meshROS.color.b = 1;

	meshROS.mesh_resource               = mesh_resource;
	meshROS.mesh_use_embedded_materials = true;
	mesh_pub->publish(meshROS);
}

}  // namespace map_visual

using namespace map_visual;
int main(int argc, char** argv) {
	pcl_input();
	ros::init(argc, argv, "map_visual");
	ros::NodeHandle nh;
	tf::TransformBroadcaster uav_tf_br;
	ros::Publisher local_map_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("/local_map", 1);
	ros::Publisher mesh_pub      = nh.advertise<visualization_msgs::Marker>("/uav_mesh", 1);
	ros::Subscriber odom_sub =
	    nh.subscribe<nav_msgs::Odometry>("/juliett/ground_truth/odom", 1, boost::bind(&odom_cb, _1, &local_map_pub, &mesh_pub, &uav_tf_br));
	std::cout << "Map visual node OK!" << std::endl;
	ros::spin();
}