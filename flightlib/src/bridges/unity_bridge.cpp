#include "flightlib/bridges/unity_bridge.hpp"

namespace flightlib {

// constructor
UnityBridge::UnityBridge()
    : client_address_("tcp://*"),
      pub_port_("10253"),
      sub_port_("10254"),
      num_frames_(0),
      last_downloaded_utime_(0),
      last_download_debug_utime_(0),
      u_packet_latency_(0),
      unity_ready_(false) {
	// initialize connections upon creating unity bridge
	initializeConnections();
	generator_ = std::default_random_engine(random_device_());
}

bool UnityBridge::initializeConnections() {
	logger_.info("Initializing ZMQ connection...");

	// create and bind an upload socket
	pub_.set(zmqpp::socket_option::send_high_water_mark, 6);
	pub_.bind(client_address_ + ":" + pub_port_);
	std::cout << "pub_port_" << pub_port_ << std::endl;
	// create and bind a download_socket
	sub_.set(zmqpp::socket_option::receive_high_water_mark, 6);
	sub_.bind(client_address_ + ":" + sub_port_);
	std::cout << "sub_port_" << sub_port_ << std::endl;
	// subscribe all messages from ZMQ
	sub_.subscribe("");

	logger_.info("Initializing ZMQ connections done!");
	return true;
}

bool UnityBridge::connectUnity(const SceneID scene_id) {
	Scalar time_out_count = 0;
	Scalar sleep_useconds = 0.2 * 1e5;
	setScene(scene_id);
	// try to connect unity
	logger_.info("Trying to Connect Unity.");
	std::cout << "[";
	while (!unity_ready_) {
		// if time out
		if (time_out_count / 1e6 > unity_connection_time_out_) {
			std::cout << "]" << std::endl;
			logger_.warn(
			    "Unity Connection time out! Make sure that Unity Standalone "
			    "or Unity Editor is running the Flightmare.");
			return false;
		}
		// initialize Scene settings
		sendInitialSettings();
		// check if setting is done
		unity_ready_ = handleSettings();
		// sleep
		usleep(sleep_useconds);
		// increase time out counter
		time_out_count += sleep_useconds;
		// print something
		std::cout << ".";
		std::cout.flush();
	}
	logger_.info("Flightmare Unity is connected.");
	// wait 1 seconds. until to environment is fully loaded.
	usleep(1 * 1e6);
	// wait until point cloud is created.
	// Unity is frozen as long as point cloud is created
	// check if it's possible to send and receive a frame again and then continue
	FrameID send_id = 1;
	getRender(send_id);
	FrameID receive_id = 0;
	while (send_id != receive_id) {
		handleOutput(receive_id);
	}
	logger_.info("Flightmare Unity is ready.");
	return unity_ready_;
}

bool UnityBridge::disconnectUnity() {
	unity_ready_ = false;
	// create new message object
	std::string client_address_dis_{"tcp://*"};
	std::string pub_port_dis_{"10255"};
	zmqpp::context context_dis_;
	zmqpp::socket pub_dis_{context_dis_, zmqpp::socket_type::publish};
	pub_dis_.set(zmqpp::socket_option::send_high_water_mark, 6);
	pub_dis_.bind(client_address_dis_ + ":" + pub_port_dis_);

	// wait until publisher is properly connected
	usleep(1000000);
	zmqpp::message msg_dis_;
	printf("Disconnect from Unity!\n");
	msg_dis_ << "DISCONNECT";
	pub_dis_.send(msg_dis_, true);
	FrameID send_id = 1;
	getRender(send_id);

	pub_.close();
	sub_.close();
	pub_dis_.close();
	printf("Disconnected!\n");
	return true;
}

bool UnityBridge::sendInitialSettings(void) {
	// create new message object
	zmqpp::message msg;
	// add topic header
	msg << "Pose";
	// create JSON object for initial settings
	json json_mesg = settings_;
	msg << json_mesg.dump();
	// send message without blocking
	pub_.send(msg, true);
	return true;
};

bool UnityBridge::handleSettings(void) {
	// create new message object
	zmqpp::message msg;

	bool done = false;
	// Unpack message metadata
	if (sub_.receive(msg, true)) {
		std::string metadata_string = msg.get(0);
		// Parse metadata
		if (json::parse(metadata_string).size() > 1) {
			return false;  // hack
		}
		done = json::parse(metadata_string).at("ready").get<bool>();
	}
	return done;
};

bool UnityBridge::getRender(const FrameID frame_id) {
	pub_msg_.ntime = frame_id;
	QuadState quad_state;
	for (size_t idx = 0; idx < pub_msg_.vehicles.size(); idx++) {
		unity_quadrotors_[idx]->getState(&quad_state);
		// 传给unity的飞机位置 = 实际飞机 - 相机和飞机的位姿差, 使得图像渲染位置=飞机位置,使视野无飞机机身遮挡。请确保第0个camera是左目
		Matrix<4, 4> cam_pose = rgb_cameras_[0]->getRelPose();
		Vector<3> delta_pose  = cam_pose.block<3, 1>(0, 3);
		// printf("camera pose to body: %f, %f, %f \n",delta_pose(0),delta_pose(1),delta_pose(2));
		pub_msg_.vehicles[idx].position = convertToDoubleVector(positionRos2Unity(quad_state.p - delta_pose));
		pub_msg_.vehicles[idx].rotation = convertToDoubleVector(quaternionRos2Unity(quad_state.q()));
	}

	for (size_t idx = 0; idx < pub_msg_.objects.size(); idx++) {
		std::shared_ptr<StaticObject> gate = static_objects_[idx];
		pub_msg_.objects[idx].position     = convertToDoubleVector(positionRos2Unity(gate->getPosition()));
		pub_msg_.objects[idx].rotation     = convertToDoubleVector(quaternionRos2Unity(gate->getQuaternion()));
	}

	// create new message object
	zmqpp::message msg;
	// add topic header
	msg << "Pose";
	// create JSON object for pose update and append
	json json_msg = pub_msg_;
	msg << json_msg.dump();
	// send message without blocking
	pub_.send(msg, true);
	return true;
}

bool UnityBridge::setScene(const SceneID& scene_id) {
	if (scene_id >= UnityScene::SceneNum) {
		logger_.warn("Scene ID is not defined, cannot set scene.");
		return false;
	}
	// logger_.info("Scene ID is set to %d.", scene_id);
	settings_.scene_id = scene_id;
	return true;
}

bool UnityBridge::addQuadrotor(std::shared_ptr<Quadrotor> quad) {
	Vehicle_t vehicle_t;
	// get quadrotor state
	QuadState quad_state;
	if (!quad->getState(&quad_state)) {
		logger_.error("Cannot get Quadrotor state.");
		return false;
	}

	vehicle_t.ID       = "quadrotor" + std::to_string(settings_.vehicles.size());
	vehicle_t.position = convertToDoubleVector(positionRos2Unity(quad_state.p));
	vehicle_t.rotation = convertToDoubleVector(quaternionRos2Unity(quad_state.q()));
	vehicle_t.size     = convertToDoubleVector(scalarRos2Unity(quad->getSize()));

	// get camera
	std::vector<std::shared_ptr<RGBCamera>> rgb_cameras = quad->getCameras();
	for (size_t cam_idx = 0; cam_idx < rgb_cameras.size(); cam_idx++) {
		std::shared_ptr<RGBCamera> cam = rgb_cameras[cam_idx];
		Camera_t camera_t;
		camera_t.ID                     = vehicle_t.ID + "_" + std::to_string(cam_idx);
		std::vector<Scalar> T_BC_Scalar = transformationRos2Unity(rgb_cameras[cam_idx]->getRelPose());
		std::vector<double> T_BC_double(T_BC_Scalar.begin(), T_BC_Scalar.end());
		camera_t.T_BC            = T_BC_double;
		camera_t.channels        = rgb_cameras[cam_idx]->getChannels();
		camera_t.width           = rgb_cameras[cam_idx]->getWidth();
		camera_t.height          = rgb_cameras[cam_idx]->getHeight();
		camera_t.fov             = rgb_cameras[cam_idx]->getFOV();
		camera_t.depth_scale     = rgb_cameras[cam_idx]->getDepthScale();
		camera_t.post_processing = rgb_cameras[cam_idx]->GetPostProcessing();
		camera_t.is_depth        = false;
		camera_t.output_index    = cam_idx;
		vehicle_t.cameras.push_back(camera_t);

		// add rgb_cameras
		rgb_cameras_.push_back(rgb_cameras[cam_idx]);
	}
	unity_quadrotors_.push_back(quad);

	settings_.vehicles.push_back(vehicle_t);
	pub_msg_.vehicles.push_back(vehicle_t);
	return true;
}

bool UnityBridge::addStaticObject(std::shared_ptr<StaticObject> static_object) {
	Object_t object_t;
	object_t.ID        = static_object->getID();
	object_t.prefab_ID = static_object->getPrefabID();
	object_t.position  = convertToDoubleVector(positionRos2Unity(static_object->getPosition()));
	object_t.rotation  = convertToDoubleVector(quaternionRos2Unity(static_object->getQuaternion()));
	object_t.size      = convertToDoubleVector(scalarRos2Unity(static_object->getSize()));

	static_objects_.push_back(static_object);
	settings_.objects.push_back(object_t);
	pub_msg_.objects.push_back(object_t);

	return true;
}

bool UnityBridge::handleOutput(FrameID& frameID) {
	// create new message object
	zmqpp::message msg;
	sub_.receive(msg);
	// unpack message metadata
	std::string json_sub_msg = msg.get(0);
	// parse metadata
	SubMessage_t sub_msg = json::parse(json_sub_msg);
	frameID              = sub_msg.ntime;

	size_t image_i = 1;
	// ensureBufferIsAllocated(sub_msg);
	for (size_t idx = 0; idx < settings_.vehicles.size(); idx++) {
		// update vehicle collision flag
		unity_quadrotors_[idx]->setCollision(sub_msg.sub_vehicles[idx].collision);

		// feed image data to RGB camera
		for (const auto& cam : settings_.vehicles[idx].cameras) {
			// 1、RGB图-----------------------------------------
			{
				uint32_t image_len = cam.width * cam.height * cam.channels;
				// Get raw image bytes from ZMQ message.
				// WARNING: This is a zero-copy operation that also casts the input to an array of unit8_t. when the message is deleted, this pointer
				// is also dereferenced.
				const uint8_t* image_data;
				msg.get(image_data, image_i);
				image_i = image_i + 1;
				// Pack image into cv::Mat
				cv::Mat new_image = cv::Mat(cam.height, cam.width, CV_MAKETYPE(CV_8U, cam.channels));
				memcpy(new_image.data, image_data, image_len);
				// Flip image since OpenCV origin is upper left, but Unity's is lower left.
				cv::flip(new_image, new_image, 0);
				// Tell OpenCv that the input is RGB.
				if (cam.channels == 3) {
					cv::cvtColor(new_image, new_image, CV_RGB2BGR);
				}
				unity_quadrotors_[idx]->getCameras()[cam.output_index]->feedImageQueue(0, new_image);
			}

			// 之前Flightmare的layer_idx： 0 是RGB, 1是Depth, 2是Seg, 3是光流
			// 现在的post_processing： 0 是RGB, 后面按设置打开的Denpth、Seg等排列

			// 2、附加开启的图-------------------------------------------
			for (size_t layer_idx = 0; layer_idx < cam.post_processing.size(); layer_idx++) {
				if (cam.post_processing[layer_idx] == RGBCameraTypes::Depth) {
					// depth (float32存在4个uint8)
					uint32_t image_len = cam.width * cam.height * 4;
					// Get raw image bytes from ZMQ message.
					// WARNING: This is a zero-copy operation that also casts the input to an array of unit8_t. when the message is deleted, this
					// pointer is also dereferenced.
					const uint8_t* image_data;
					msg.get(image_data, image_i);
					image_i = image_i + 1;
					// Pack image into cv::Mat
					cv::Mat new_image = cv::Mat(cam.height, cam.width, CV_32FC1);
					memcpy(new_image.data, image_data, image_len);
					// Flip image since OpenCV origin is upper left, but Unity's is lower left.
					new_image = new_image * (1.f);			// 默认单位km
					cv::flip(new_image, new_image, 0);
					new_image = cv::max(new_image, 0.0f);	// 有时候返回的深度值有负数
					unity_quadrotors_[idx]->getCameras()[cam.output_index]->feedImageQueue(CameraLayer::DepthMap, new_image);
				}
			}
		}
	}
	return true;
}

bool UnityBridge::spawnTrees(Ref<Vector<3>> bounding_box_, Ref<Vector<3>> bounding_box_origin_, Scalar avg_tree_spacing_) {
	printf("Start Spawn Trees... \n");
	rmTrees();
	// 循环多次避免偶尔一次没render上，后面树再也无法生成
	for (size_t i = 0; i < 3; i++)
		refreshUnity(10086 + i);

	TreeMessage_t tree_msg;
	// compute the requested tree density for Poisson
	Scalar density = 1.0 / (avg_tree_spacing_ * avg_tree_spacing_);
	int num_trees  = static_cast<int>(bounding_box_[0] * bounding_box_[1] * density);
	// draw sample from poisson distribution
	std::poisson_distribution<int> poisson_dist(num_trees);
	tree_msg.density = static_cast<double>(poisson_dist(generator_));
	printf("Tree Spacing is %f. \n", avg_tree_spacing_);
	// generate random seed
	std::uniform_int_distribution<int> distribution(1, 1 << 20);
	tree_msg.seed = distribution(generator_);

	tree_msg.bounding_origin[0] = bounding_box_origin_[0];
	tree_msg.bounding_origin[1] = bounding_box_origin_[1];
	tree_msg.bounding_area[0]   = bounding_box_[0];
	tree_msg.bounding_area[1]   = bounding_box_[1];
	bool spawned                = placeTrees(tree_msg);
	std::cout << "Tree Spawned" << std::endl;
	return spawned;
}

void UnityBridge::generatePointcloud(const Ref<Vector<3>>& min_corner, const Ref<Vector<3>>& max_corner, int ply_idx, std::string path,
                                     SceneID scene_id, Scalar pc_resolution_) {
	printf("Start creating pointcloud... \n");
	PointCloudMessage_t pcd_msg;
	pcd_msg.scene_id = scene_id;
	pcd_msg.bounding_box_scale =
	    std::vector<double>{(max_corner.x() - min_corner.x()), (max_corner.y() - min_corner.y()), (max_corner.z() - min_corner.z())};
	printf("Scale pointcloud: [%.2f, %.2f, %.2f]\n", pcd_msg.bounding_box_scale.at(0), pcd_msg.bounding_box_scale.at(1),
	       pcd_msg.bounding_box_scale.at(2));

	pcd_msg.bounding_box_origin = std::vector<double>{(max_corner.x() + min_corner.x()) / 2.0, (max_corner.y() + min_corner.y()) / 2.0,
	                                                  (max_corner.z() + min_corner.z()) / 2.0};
	printf("Origin pointcloud: [%.2f, %.2f, %.2f]\n", pcd_msg.bounding_box_origin.at(0), pcd_msg.bounding_box_origin.at(1),
	       pcd_msg.bounding_box_origin.at(2));

	pcd_msg.path                    = path;
	pcd_msg.file_name               = "pointcloud-" + std::to_string(ply_idx);
	pcd_msg.unity_ground_offset     = 0.0;
	pcd_msg.resolution_above_ground = pc_resolution_;
	pcd_msg.resolution_below_ground = pc_resolution_;

	std::cout << "Save file name: " << pcd_msg.path + pcd_msg.file_name + ".ply" << std::endl;

	while (std::experimental::filesystem::exists(pcd_msg.path + pcd_msg.file_name + ".ply")) {
		std::cout << "file already exists, delete! " << std::endl;
		std::experimental::filesystem::remove(pcd_msg.path + pcd_msg.file_name + ".ply");
		usleep(1 * 1e6);
	}
	std::cout << "Pointcloud Saving...";

	pointCloudGenerator(pcd_msg);

	// render Unity until point cloud exists
	FrameID frameID = 10086;
	while (!std::experimental::filesystem::exists(pcd_msg.path + pcd_msg.file_name + ".ply")) {
		std::cout << ".";
		std::cout.flush();
		refreshUnity(frameID);  // render, not usleep (BUG: Flightmare must render continuously to refresh the sense and generate pointcloud.)
		frameID++;
	}
	usleep(5 * 1e6);
	printf("Pointcloud saved!\n");
}

template<typename T>
std::vector<double> UnityBridge::convertToDoubleVector(const std::vector<T>& input) {
	std::vector<double> output(input.size());
	std::transform(input.begin(), input.end(), output.begin(), [](T value) { return static_cast<double>(value); });
	return output;
}

void UnityBridge::refreshUnity(FrameID id = 10086) {
	FrameID frameID_test = id;
	getRender(frameID_test);
	FrameID frameID_rt;
	handleOutput(frameID_rt);
	while (frameID_test != frameID_rt)
		handleOutput(frameID_rt);
}

bool UnityBridge::placeTrees(TreeMessage_t& tree_msg) {
	std::string client_address_{"tcp://*"};
	std::string pub_tree_port_{"10255"};
	zmqpp::context context_;
	zmqpp::socket pub_tree_{context_, zmqpp::socket_type::publish};
	pub_tree_.set(zmqpp::socket_option::send_high_water_mark, 6);
	pub_tree_.bind(client_address_ + ":" + pub_tree_port_);

	std::string sub_tree_port_{"10256"};
	zmqpp::socket sub_tree_{context_, zmqpp::socket_type::subscribe};
	sub_tree_.set(zmqpp::socket_option::receive_high_water_mark, 6);
	sub_tree_.bind(client_address_ + ":" + sub_tree_port_);
	sub_tree_.subscribe("PLACETREE");

	// wait until publisher is properly connected
	usleep(1000000);
	zmqpp::message msg;
	msg << "PLACETREE";

	// check if seed is not initialized
	if (tree_msg.seed == -1)
		tree_msg.seed = rand();

	json json_msg = tree_msg;
	msg << json_msg.dump();
	pub_tree_.send(msg, true);
	pub_tree_.close();

	double sleep_useconds = 0.2 * 1e5;
	FrameID frameID       = 10086;
	//  Wait until response received
	while (true) {
		if (sub_tree_.receive(msg, true)) {
			break;
		}
		// render, not usleep (BUG: Flightmare must render continuously to refresh the sense and tree.)
		refreshUnity(frameID);
		frameID++;
	}
	sub_tree_.close();
	return true;
}

void UnityBridge::rmTrees() {
	std::string client_address_{"tcp://*"};
	std::string pub_tree_port_{"10255"};
	zmqpp::context context_;
	zmqpp::socket pub_tree_{context_, zmqpp::socket_type::publish};
	pub_tree_.set(zmqpp::socket_option::send_high_water_mark, 6);
	pub_tree_.bind(client_address_ + ":" + pub_tree_port_);

	// wait until publisher is properly connected
	usleep(1000000);
	zmqpp::message msg;
	msg << "RMTREE";
	pub_tree_.send(msg, true);
	pub_tree_.close();
}

void UnityBridge::pointCloudGenerator(PointCloudMessage_t& pcd_msg) {
	std::string client_address_{"tcp://*"};
	std::string pub_pc_port_{"10255"};
	zmqpp::context context_;
	zmqpp::socket pub_pc_{context_, zmqpp::socket_type::publish};
	pub_pc_.set(zmqpp::socket_option::send_high_water_mark, 6);
	pub_pc_.bind(client_address_ + ":" + pub_pc_port_);

	// wait until publisher is properly connected
	usleep(1000000);
	zmqpp::message msg;
	msg << "PCD";
	json json_msg = pcd_msg;
	msg << json_msg.dump();
	pub_pc_.send(msg, true);

	pub_pc_.close();
}

}  // namespace flightlib
