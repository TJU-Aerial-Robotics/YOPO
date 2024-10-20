#include <ros/ros.h>

#include "flightlib/ros_nodes/flight_pilot.hpp"

int main(int argc, char** argv) {
	ros::init(argc, argv, "flight_pilot");
	flightros::FlightPilot pilot(ros::NodeHandle(), ros::NodeHandle("~"));

	// spin the ros
	ros::spin();

	return 0;
}