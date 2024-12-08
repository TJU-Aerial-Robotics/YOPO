"""
算法具有有一定的Sim2Real的泛化能力, 如果有条件可用雷达+深度相机收集数据, 合并至仿真数据集中一同训练, 以进一步保证实飞的可靠性
# (1) 运行雷达里程计以记录无人机状态和地图真值. 注意保证地图和里程计处于同一坐标系，请在一次运行中同时记录图像与里程计的rosbag + 保存地图
# (可选) 由于里程计漂移，可用utils/pointcloud_clip对地图进行微调和降噪，本文件需对无人机位置translation_no和姿态R_no(yaw, pitch, roll)做相同的变换
# (2) 播包rosbag, 运行本文件: 记录位置、姿态、图像，保存至save_dir
"""
import cv2
import numpy as np
import time, os, sys
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation

# IMPORTANT PARAM
save_dir = os.environ["FLIGHTMARE_PATH"] + "/run/yopo_realworld"
label_path = save_dir + "/label.npz"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# Due to odometry drift, the map is adjusted, and the drone's position is also adjusted accordingly.
R_no = Rotation.from_euler('ZYX', [0.0, 0.0, 0.0], degrees=True)  # yaw, pitch, roll
translation_no = np.array([0.0, 0.0, 0.0])
img_height = 270
img_width = 480

# VARIABLES
depth_img = None
pos = None
quat = None
positions = []
quaternions = []
frame_id = 0
new_depth = False
new_odom = False
first_frame = True
last_time = time.time()


def callback_odometry(data):
    # NWU
    global pos, quat, new_odom, R_no, translation_no
    p_ob = np.array([[data.pose.pose.position.x],
                     [data.pose.pose.position.y],
                     [data.pose.pose.position.z]])
    q_ob = np.array([data.pose.pose.orientation.x,
                     data.pose.pose.orientation.y,
                     data.pose.pose.orientation.z,
                     data.pose.pose.orientation.w])
    R_ob = Rotation.from_quat(q_ob)      # old->body (xyzw)
    quat_xyzw = (R_no * R_ob).as_quat()  # new->body (xyzw)
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    pos = np.squeeze(np.dot(R_no.as_matrix(), p_ob)) + translation_no
    new_odom = True


def callback_depth(data):
    global depth_img, new_depth, img_height, img_width
    max_dis = 20.0
    min_dis = 0.03
    height = img_height
    width = img_width
    scale = 0.001
    bridge = CvBridge()
    try:
        depth_ = bridge.imgmsg_to_cv2(data, "32FC1")
    except:
        print("CV_bridge ERROR: Possible solutions may be found at https://github.com/TJU-Aerial-Robotics/YOPO/issues/2")

    if depth_.shape[0] != height or depth_.shape[1] != width:
        depth_ = cv2.resize(depth_, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_ = np.minimum(depth_ * scale, max_dis) / max_dis

    try:
        nan_mask = np.isnan(depth_) | (depth_ < min_dis)
        depth_ = cv2.inpaint(np.uint8(depth_ * 255), np.uint8(nan_mask), 3, cv2.INPAINT_NS)
        depth_ = depth_.astype(np.float32) / 255.0
    except:
        print("Interpolation failed")

    # Not necessary, but encountered some inexplicable errors previously, so temporarily kept.
    if np.sum(np.isnan(depth_)) > 0:
        depth_[np.isnan(depth_)] = 0
        print("WARN: Have NAN values in depth image")

    depth_img = depth_.copy()
    new_depth = True


def save_data(_timer):
    global pos, quat, new_odom, depth_img, new_depth, last_time, first_frame
    global save_dir, label_path, frame_id, positions, quaternions
    if not (new_odom and new_depth):
        if not first_frame and time.time() - last_time > 1:
            np.savez(
                label_path,
                positions=np.asarray(positions),
                quaternions=np.asarray(quaternions),
            )
            print("Record Done!")
            sys.exit()
        return
    new_odom, new_depth = False, False

    image_path = save_dir + "/img_" + str(frame_id) + ".tif"
    cv2.imwrite(image_path, depth_img)
    positions.append(pos)
    quaternions.append(quat)

    last_time = time.time()
    first_frame = False
    frame_id = frame_id + 1


if __name__ == "__main__":
    rospy.init_node('data_collect', anonymous=False)
    odom_ref_sub = rospy.Subscriber("/odometry/imu", Odometry, callback_odometry, queue_size=1)
    depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback_depth, queue_size=1)
    timer = rospy.Timer(rospy.Duration(0.033), save_data)
    print("Data Collection Node Ready!")
    rospy.spin()
