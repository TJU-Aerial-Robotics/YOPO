#include "flightlib/ros_nodes/flight_pilot.hpp"

namespace flightros {

FlightPilot::FlightPilot(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), scene_id_(4), unity_ready_(false), unity_render_(false), receive_id_(0), frameID(0), main_loop_freq_(50.0) {
	// quad initialization
	quad_ptr_ = std::make_shared<Quadrotor>();
	// load parameters
	std::string cfg_path = getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/quadrotor_ros.yaml");
	YAML::Node cfg_      = YAML::LoadFile(cfg_path);
	loadParams(cfg_);
	configCamera(cfg_);
	quad_ptr_->setSize(quad_size_);

	// initialization
	quad_state_.setZero();
	quad_ptr_->reset(quad_state_);

	sgm_ = std::make_shared<sgm_gpu::SgmGpu>(width_, height_);

	// initialize subscriber and publisher
	left_img_pub = nh_.advertise<sensor_msgs::Image>("RGB_image", 1);
	stereo_pub   = nh_.advertise<sensor_msgs::Image>("stereo_image", 1);
	depth_pub    = nh_.advertise<sensor_msgs::Image>("depth_image", 1);
	cam_info_pub = nh_.advertise<sensor_msgs::CameraInfo>("camera_info", 1);

	state_est_sub_  = nh_.subscribe(odom_topic_, 1, &FlightPilot::poseCallback, this);
	spawn_tree_sub_ = nh_.subscribe("/spawn_tree", 1, &FlightPilot::spawnTreeCallback, this);
	clear_tree_sub_ = nh_.subscribe("/clear_tree", 1, &FlightPilot::clearTreeCallback, this);
	save_pc_sub_    = nh_.subscribe("/save_pc", 1, &FlightPilot::savePointcloudCallback, this);
	timestamp       = ros::Time::now();

	// connect unity and setup unity
	setUnity();
	connectUnity();
	if (!unity_ready_){
		ROS_ERROR("[FlightRos] Connection Faild! Start the Flightmare Unity First!");
		exit(1);
	}
	spawnTreesAndSavePointCloud();

	timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_), &FlightPilot::mainLoopCallback, this);
	std::cout << "[FlightRos] Ros Node is Ready!" << std::endl;
}

FlightPilot::~FlightPilot() { disconnectUnity(); }

void FlightPilot::spawnTreeCallback(const std_msgs::Empty::ConstPtr& msg) {
	if (!unity_ready_ || unity_bridge_ptr_ == nullptr)
		return;
	unity_bridge_ptr_->spawnTrees(bounding_box_, bounding_box_origin_, avg_tree_spacing_);
}

void FlightPilot::clearTreeCallback(const std_msgs::Empty::ConstPtr& msg) { unity_bridge_ptr_->rmTrees(); }

// If the point cloud is not saved when init, it also can be saved by this callback.
void FlightPilot::savePointcloudCallback(const std_msgs::Empty::ConstPtr& msg) {
	Vector<3> min_corner = bounding_box_origin_ - 0.5 * bounding_box_;
	Vector<3> max_corner = bounding_box_origin_ + 0.5 * bounding_box_;
	unity_bridge_ptr_->generatePointcloud(min_corner, max_corner, ply_id_, ply_path_, scene_id_, pointcloud_resolution_);
}

void FlightPilot::poseCallback(const nav_msgs::Odometry::ConstPtr& msg) {
	quad_state_.x[QS::POSX] = (Scalar)msg->pose.pose.position.x;
	quad_state_.x[QS::POSY] = (Scalar)msg->pose.pose.position.y;
	quad_state_.x[QS::POSZ] = (Scalar)msg->pose.pose.position.z;
	quad_state_.x[QS::ATTW] = (Scalar)msg->pose.pose.orientation.w;
	quad_state_.x[QS::ATTX] = (Scalar)msg->pose.pose.orientation.x;
	quad_state_.x[QS::ATTY] = (Scalar)msg->pose.pose.orientation.y;
	quad_state_.x[QS::ATTZ] = (Scalar)msg->pose.pose.orientation.z;

	quad_ptr_->setState(quad_state_);
	timestamp = msg->header.stamp;
}

void FlightPilot::mainLoopCallback(const ros::TimerEvent& event) {
	if (!unity_render_ || !unity_ready_)
		return;

	frameID++;
	ros::Time timestamp_ = timestamp;
	unity_bridge_ptr_->getRender(frameID);  // 1ms

	FrameID frameID_rt;
	unity_bridge_ptr_->handleOutput(frameID_rt);  // 30ms
	while (frameID != frameID_rt)
		unity_bridge_ptr_->handleOutput(frameID_rt);

	cv::Mat left_img, right_img, depth_img;
	// publish RGB image
	rgb_camera_left->getRGBImage(left_img);
	sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", left_img).toImageMsg();
	img_msg->header.stamp         = timestamp_;
	left_img_pub.publish(img_msg);

	// publish camera info
	int width                 = rgb_camera_left->getWidth();
	int hight                 = rgb_camera_left->getHeight();
	float fov                 = rgb_camera_left->getFOV();
	float cx                  = width / 2.0;
	float cy                  = hight / 2.0;
	float fx                  = cy / tan(0.5 * fov * M_PI / 180.0);  // 他这个特殊，fov是竖直方向的
	float fy                  = fx;
	boost::array<double, 9> K = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
	sensor_msgs::CameraInfo cam_msg;
	cam_msg.height = hight;
	cam_msg.width  = width;
	cam_msg.K      = K;
	cam_info_pub.publish(cam_msg);

	// publish depth image
	if (use_depth) {
		rgb_camera_left->getDepthMap(depth_img);
		depth_img                       = depth_img * 1000.0;
		depth_img                       = cv::min(depth_img, 20.0);
		sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth_img).toImageMsg();
		depth_msg->header.stamp         = timestamp_;
		depth_pub.publish(depth_msg);
	}

	// publish stereo image
	if (use_stereo) {
		rgb_camera_right->getRGBImage(right_img);
		cv::Mat stereo_(height_, width_, CV_32FC1);
		computeDepthImage(left_img, right_img, &stereo_);
		if (use_depth) {
			// Complete the NaN values in the depth map, as the RealSense performs better than SGM.
			cv::Mat mask, mask1, mask2;
			cv::compare(stereo_, 0, mask1, cv::CMP_EQ);   // 将 A 中为 0 的位置置为 255，其余位置置为 0
			cv::compare(stereo_, 20, mask2, cv::CMP_GT);  // 将 A 中大于 20 的位置置为 255，其余位置置为 0
			mask = mask1 | mask2;                         // 将两个掩码进行逻辑或操作
			depth_img.copyTo(stereo_, mask);              // 将 B 中 mask 为 255 的位置的值复制到 A 中
		}

		sensor_msgs::ImagePtr stereo_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", stereo_).toImageMsg();
		stereo_msg->header.stamp         = timestamp_;
		stereo_pub.publish(stereo_msg);
	}
}

bool FlightPilot::setUnity() {
	if (unity_render_ && unity_bridge_ptr_ == nullptr) {
		// create unity bridge
		unity_bridge_ptr_ = UnityBridge::getInstance();
		unity_bridge_ptr_->addQuadrotor(quad_ptr_);
		ROS_INFO("[%s] Unity Bridge is created.", pnh_.getNamespace().c_str());
	}
	return true;
}

bool FlightPilot::spawnTreesAndSavePointCloud() {
	if (!unity_ready_ || unity_bridge_ptr_ == nullptr || !spawn_tree_)
		return false;

	unity_bridge_ptr_->spawnTrees(bounding_box_, bounding_box_origin_, avg_tree_spacing_);

	if (!save_pointcloud_)
		return true;
	// Saving point cloud during the testing is much time-consuming, but can be used for evaluation.
	// The following can be commented if evaluation is not needed.
	Vector<3> min_corner = bounding_box_origin_ - 0.5 * bounding_box_;
	Vector<3> max_corner = bounding_box_origin_ + 0.5 * bounding_box_;
	unity_bridge_ptr_->generatePointcloud(min_corner, max_corner, ply_id_, ply_path_, scene_id_, pointcloud_resolution_);
	return true;
}

bool FlightPilot::connectUnity() {
	if (!unity_render_ || unity_bridge_ptr_ == nullptr)
		return false;
	unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
	return unity_ready_;
}

void FlightPilot::disconnectUnity() {
	if (unity_render_ && unity_bridge_ptr_ != nullptr)
		return;
	unity_bridge_ptr_->disconnectUnity();
	unity_ready_ = false;
}

bool FlightPilot::loadParams(const YAML::Node& cfg) {
	// ros
	main_loop_freq_ = cfg["main_loop_freq"].as<int>();
	odom_topic_     = cfg["odom_topic"].as<std::string>();
	// camera
	width_     = cfg["rgb_camera_left"]["width"].as<int>();
	height_    = cfg["rgb_camera_left"]["height"].as<int>();
	fov_       = cfg["rgb_camera_left"]["fov"].as<Scalar>();
	use_depth  = cfg["rgb_camera_left"]["enable_depth"].as<bool>();
	use_stereo = cfg["rgb_camera_right"]["on"].as<bool>();
	// scence
	scene_id_          = cfg["scene_id"].as<int>();
	unity_render_      = cfg["unity_render"].as<bool>();
	Scalar quad_size_i = cfg["quad_size"].as<Scalar>();
	quad_size_         = Vector<3>(quad_size_i, quad_size_i, quad_size_i);
	spawn_tree_        = cfg["unity"]["spawn_trees"].as<bool>();
	save_pointcloud_   = cfg["unity"]["save_pointcloud"].as<bool>();
	avg_tree_spacing_  = cfg["unity"]["avg_tree_spacing"].as<Scalar>();
	for (int i = 0; i < 3; ++i) {
		bounding_box_(i)        = cfg["unity"]["bounding_box"][i].as<Scalar>();
		bounding_box_origin_(i) = cfg["unity"]["bounding_box_origin"][i].as<Scalar>();
	}
	pointcloud_resolution_ = cfg["unity"]["pointcloud_resolution"].as<Scalar>();
	ply_path_              = getenv("FLIGHTMARE_PATH") + cfg["ply_path"].as<std::string>();
	if (!boost::filesystem::exists(ply_path_)) {
		boost::filesystem::create_directories(ply_path_);
		std::cout << "Directory created: " << ply_path_ << std::endl;
	}
	return true;
}

void FlightPilot::computeDepthImage(const cv::Mat& left_frame, const cv::Mat& right_frame, cv::Mat* const depth) {
	cv::Mat disparity(height_, width_, CV_8UC1);
	sgm_->computeDisparity(left_frame, right_frame, disparity);
	disparity.convertTo(disparity, CV_32FC1);

	// compute depth from disparity
	float f = (width_ / 2.0) / std::tan((M_PI * (fov_ / 180.0)) / 2.0);
	//  depth = stereo_baseline_ * f / disparity
	for (int r = 0; r < height_; ++r) {
		for (int c = 0; c < width_; ++c) {
			if (disparity.at<float>(r, c) == 0.0f) {
				depth->at<float>(r, c) = 0.0f;
			} else if (disparity.at<float>(r, c) == 255.0f) {
				depth->at<float>(r, c) = 0.0f;
			} else {
				depth->at<float>(r, c) = static_cast<float>(stereo_baseline_) * f / disparity.at<float>(r, c);
			}
		}
	}
}

bool FlightPilot::configCamera(const YAML::Node& cfg) {
	if (!cfg["rgb_camera_left"] || !cfg["rgb_camera_right"]) {
		ROS_ERROR("Cannot config stereo Camera");
		return false;
	}
	// create left camera --------------------------------------------
	rgb_camera_left = std::make_shared<RGBCamera>();

	// load camera settings
	std::vector<Scalar> t_BC_vec = cfg["rgb_camera_left"]["t_BC"].as<std::vector<Scalar>>();
	std::vector<Scalar> r_BC_vec = cfg["rgb_camera_left"]["r_BC"].as<std::vector<Scalar>>();
	Vector<3> t_BC(t_BC_vec.data());
	Matrix<3, 3> r_BC = (AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()) * AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
	                     AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ())).toRotationMatrix();  // the rotation order has been verified
	// Convert the horizontal FOV (usually used) to vertical FOV (flightmare).
	Scalar rgb_fov_deg_    = cfg["rgb_camera_left"]["fov"].as<Scalar>();
	double hor_fov_radians = (M_PI * (rgb_fov_deg_ / 180.0));
	Scalar img_rows_       = cfg["rgb_camera_left"]["height"].as<Scalar>();
	Scalar img_cols_       = cfg["rgb_camera_left"]["width"].as<Scalar>();
	double flightmare_fov  = 2. * std::atan(std::tan(hor_fov_radians / 2) * img_rows_ / img_cols_);
	flightmare_fov         = (flightmare_fov / M_PI) * 180.0;
	rgb_camera_left->setFOV(flightmare_fov);
	rgb_camera_left->setWidth(cfg["rgb_camera_left"]["width"].as<int>());
	rgb_camera_left->setHeight(cfg["rgb_camera_left"]["height"].as<int>());
	rgb_camera_left->setRelPose(t_BC, r_BC);
	rgb_camera_left->enableOpticalFlow(cfg["rgb_camera_left"]["enable_opticalflow"].as<bool>());
	rgb_camera_left->enableSegmentation(cfg["rgb_camera_left"]["enable_segmentation"].as<bool>());
	rgb_camera_left->enableDepth(cfg["rgb_camera_left"]["enable_depth"].as<bool>());
	// add camera to the quadrotor
	quad_ptr_->addRGBCamera(rgb_camera_left);

	// create right camera --------------------------------------------
	if (use_stereo) {
		rgb_camera_right = std::make_shared<RGBCamera>();

		// load camera settings
		std::vector<Scalar> t_BC_vec_r = cfg["rgb_camera_right"]["t_BC"].as<std::vector<Scalar>>();
		std::vector<Scalar> r_BC_vec_r = cfg["rgb_camera_right"]["r_BC"].as<std::vector<Scalar>>();

		Vector<3> t_BC_r(t_BC_vec_r.data());
		Matrix<3, 3> r_BC_r = (AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()) * AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
		                       AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ())).toRotationMatrix();

		rgb_camera_right->setFOV(flightmare_fov);
		rgb_camera_right->setWidth(cfg["rgb_camera_left"]["width"].as<int>());
		rgb_camera_right->setHeight(cfg["rgb_camera_left"]["height"].as<int>());
		rgb_camera_right->setRelPose(t_BC_r, r_BC_r);
		rgb_camera_right->enableOpticalFlow(false);
		rgb_camera_right->enableSegmentation(false);
		rgb_camera_right->enableDepth(false);
		// add camera to the quadrotor
		quad_ptr_->addRGBCamera(rgb_camera_right);

		stereo_baseline_ = fabs(t_BC(1) - t_BC_r(1));
	}
	return true;
}

}  // namespace flightros