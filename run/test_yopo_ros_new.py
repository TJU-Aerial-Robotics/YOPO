"""
    YOPO ROS NODE:
    Subscribe odometry and depth messages, perform network inference, solve trajectory, and publish control commands.
    Used to replace test_yopo_ros.py and yopo_planner_node.cpp.
    If you encounter issues (such as unsmooth) with this script, try using the following instead:
        $ cd ~/YOPO/run
        $ conda activate yopo
        $ python test_yopo_ros.py --trial=1 --epoch=0 --iter=0
        $ cd ~/YOPO/flightlib/build
        $ ./yopo_planner_node
"""
import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
from threading import Lock
import numpy as np
import cv2
import os
import torch
import argparse
import time
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
from flightpolicy.control_msg import PositionCommand
from flightpolicy.yopo.yopo_policy import YopoPolicy
from flightpolicy.yopo.primitive_utils import LatticeParam, LatticePrimitive, Poly5Solver, Polys5Solver, calculate_yaw

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        self.height = self.config['img_height']
        self.width = self.config['img_width']
        self.goal = np.array(self.config['goal'])
        self.env = self.config['env']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/traj_opt.yaml", 'r'))
        self.lattice_space = LatticeParam(cfg)
        self.lattice_primitive = LatticePrimitive(self.lattice_space)

        # variables
        self.bridge = CvBridge()
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            saved_variables = torch.load(weight, map_location=self.device)
            saved_variables["data"]["lattice_space"] = self.lattice_space
            saved_variables["data"]["lattice_primitive"] = self.lattice_primitive
            self.policy = YopoPolicy(device=self.device, **saved_variables["data"])
            self.policy.load_state_dict(saved_variables["state_dict"], strict=False)
            self.policy.to(self.device)
            self.policy.set_training_mode(False)
        torch.set_grad_enabled(False)
        self.warm_up()

        # ros publisher
        odom_topic = self.config['odom_topic']
        depth_topic = self.config['depth_topic']
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher("/so3_control/pos_cmd", PositionCommand, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.callback_odometry, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.callback_depth, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        self.arrive = False
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if np.linalg.norm(pos - self.goal) < 4 and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # pose and goal_dir
        goal_w = (self.goal - self.desire_pos) / np.linalg.norm(self.goal - self.desire_pos)
        goal_c = np.dot(Rotation_cw, goal_w)

        vel_acc = np.concatenate((vel_c, acc_c), axis=0)
        vel_acc_norm = self.normalize_obs(vel_acc[np.newaxis, :])
        obs_norm = np.hstack((vel_acc_norm, goal_c[np.newaxis, :]))
        return obs_norm

    def callback_depth(self, data):
        if not self.odom_init:
            return

        # 1. Depth Image Process
        min_dis, max_dis = 0.03, 20.0
        scale = {'435': 0.001, 'flightmare': 1.0}.get(self.env, 1.0)

        try:
            depth = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except:
            print("CV_bridge ERROR: Possible solutions may be found at https://github.com/TJU-Aerial-Robotics/YOPO/issues/2")

        time0 = time.time()
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth * scale, max_dis) / max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < min_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs = self.process_odom()
        obs_input = self.prepare_input_observation(obs)
        obs_input = obs_input.to(self.device, non_blocking=True)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        with torch.no_grad():
            network_output = self.policy(depth_input, obs_input)
        network_output = network_output.cpu().numpy()   # torch.cuda.synchronize() is not needed here
        time3 = time.time()
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate_pred, score_pred = self.process_output(network_output, return_all_preds=self.visualize)
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate_pred.T.reshape(-1, 3, 3)
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)
        # endstate_w = endstate_w.reshape(-1, 9).T

        action_id = np.argmin(score_pred) if self.visualize else 0
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            self.optimal_poly_x = Poly5Solver(self.desire_pos[0], self.desire_vel[0], self.desire_acc[0],
                                              endstate_w[action_id, 0, 0] + self.desire_pos[0], endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.lattice_space.segment_time)
            self.optimal_poly_y = Poly5Solver(self.desire_pos[1], self.desire_vel[1], self.desire_acc[1],
                                              endstate_w[action_id, 1, 0] + self.desire_pos[1], endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.lattice_space.segment_time)
            self.optimal_poly_z = Poly5Solver(self.desire_pos[2], self.desire_vel[2], self.desire_acc[2],
                                              endstate_w[action_id, 2, 0] + self.desire_pos[2], endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.lattice_space.segment_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        if self.verbose:
            self.time_interpolation = self.time_interpolation + (time1 - time0)
            self.time_prepare = self.time_prepare + (time2 - time1)
            self.time_forward = self.time_forward + (time3 - time2)
            self.time_process = self.time_process + (time4 - time3)
            self.time_visualize = self.time_visualize + (time5 - time4)
            self.count = self.count + 1
            print(f"Time Consuming:"
                  f"depth-interpolation: {1000 * self.time_interpolation / self.count:.2f}ms;"
                  f"data-prepare: {1000 * self.time_prepare / self.count:.2f}ms; "
                  f"network-inference: {1000 * self.time_forward / self.count:.2f}ms; "
                  f"post-process: {1000 * self.time_process / self.count:.2f}ms;"
                  f"visualize-trajectory: {1000 * self.time_visualize / self.count:.2f}ms")

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.lattice_space.segment_time:
            return
        if self.arrive:
            self.desire_init = False  # ready for next rollout
            return
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.ctrl_pub.publish(control_msg)

    def process_output(self, network_output, return_all_preds=False):
        if network_output.shape[0] != 1:
            raise ValueError("batch of output values must be 1 in test!")
        network_output = network_output.reshape(10, self.lattice_space.horizon_num * self.lattice_space.vertical_num)
        endstate_pred = network_output[0:9, :]
        score_pred = network_output[9, :]

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - action_id
            endstate_prediction = self.pred_to_endstate(endstate_pred[:, action_id], lattice_id)
            endstate_prediction = endstate_prediction[:, np.newaxis]
            score_prediction = score_pred[action_id]
        else:
            endstate_prediction = np.zeros_like(endstate_pred)
            score_prediction = score_pred
            for i in range(self.lattice_space.horizon_num * self.lattice_space.vertical_num):
                lattice_id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - i
                endstate_prediction[:, i] = self.pred_to_endstate(endstate_pred[:, i], lattice_id)

        return endstate_prediction, score_prediction

    def prepare_input_observation(self, obs):
        """
            convert the observation from body frame to primitive frame,
            and then concatenate it with the depth features (to ensure the translational invariance)
        """
        if obs.shape[0] != 1:
            raise ValueError("batch of input observations must be 1 in test!")

        obs_return = np.ones((obs.shape[0], obs.shape[1], self.lattice_space.vertical_num, self.lattice_space.horizon_num), dtype=np.float32)
        id = 0
        obs_reshaped = obs.reshape(3, 3)
        for i in range(self.lattice_space.vertical_num - 1, -1, -1):
            for j in range(self.lattice_space.horizon_num - 1, -1, -1):
                Rbp = self.lattice_primitive.getRotation(id)
                obs_return_reshaped = np.dot(obs_reshaped, Rbp)
                obs_return[:, :, i, j] = obs_return_reshaped.reshape(9)
                id = id + 1
        return torch.from_numpy(obs_return)

    def pred_to_endstate(self, endstate_pred: np.ndarray, id: int):
        """
            Transform the predicted state to the body frame.
        """
        delta_yaw = endstate_pred[0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[1] * self.lattice_primitive.pitch_diff
        radio = endstate_pred[2] * self.lattice_space.radio_range + self.lattice_space.radio_range
        yaw, pitch = self.lattice_primitive.getAngleLattice(id)
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.array((endstate_x, endstate_y, endstate_z))

        endstate_vp = endstate_pred[3:6] * self.lattice_space.vel_max
        endstate_ap = endstate_pred[6:9] * self.lattice_space.acc_max
        Rpb = self.lattice_primitive.getRotation(id).T
        endstate_vb = np.matmul(endstate_vp, Rpb)
        endstate_ab = np.matmul(endstate_ap, Rpb)
        endstate = np.concatenate((endstate_p, endstate_vb, endstate_ab))
        endstate[[0, 1, 2, 3, 4, 5, 6, 7, 8]] = endstate[[0, 3, 6, 1, 4, 7, 2, 5, 8]]
        return endstate

    def normalize_obs(self, vel_acc):
        vel_norm = vel_acc[:, 0:3] / self.lattice_space.vel_max
        acc_norm = vel_acc[:, 3:6] / self.lattice_space.acc_max
        return np.hstack((vel_norm, acc_norm))

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.lattice_space.segment_time / 20.0
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.lattice_space.segment_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(self.desire_pos[0], self.desire_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + self.desire_pos[0], zero_state[:, 0], zero_state[:, 0], self.lattice_space.segment_time)
            lattice_poly_y = Polys5Solver(self.desire_pos[1], self.desire_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + self.desire_pos[1], zero_state[:, 1], zero_state[:, 1], self.lattice_space.segment_time)
            lattice_poly_z = Polys5Solver(self.desire_pos[2], self.desire_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + self.desire_pos[2], zero_state[:, 2], zero_state[:, 2], self.lattice_space.segment_time)
            t_values = np.arange(0, self.lattice_space.segment_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(self.desire_pos[0], self.desire_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + self.desire_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.lattice_space.segment_time)
            all_poly_y = Polys5Solver(self.desire_pos[1], self.desire_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + self.desire_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.lattice_space.segment_time)
            all_poly_z = Polys5Solver(self.desire_pos[2], self.desire_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + self.desire_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.lattice_space.segment_time)
            t_values = np.arange(0, self.lattice_space.segment_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def warm_up(self):
        depth = np.zeros(shape=[1, 1, self.height, self.width], dtype=np.float32)
        obs = np.zeros(shape=[1, 9], dtype=np.float32)
        obs_input = self.prepare_input_observation(obs)
        network_output = self.policy(torch.from_numpy(depth).to(self.device), obs_input.to(self.device))
        self.process_output(network_output.cpu().numpy(), return_all_preds=True)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=0, help="epoch number")
    parser.add_argument("--iter", type=int, default=0, help="iter number")
    parser.add_argument("--trt_file", type=str, default='yopo_trt.pth', help="tensorrt filename")
    return parser


# In realworld flight: visualize=False; use_tensorrt=True, and ensure the pitch_angle consistent with your platform
# When modifying the pitch_angle, there's no need to re-collect and re-train, as all predictions are in the camera coordinate system
# Change the flight speed at traj_opt.yaml and there's no need to re-collect and re-train
def main():
    args = parser().parse_args()
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    weight = args.trt_file if args.use_tensorrt else f"{rsg_root}/saved/YOPO_{args.trial}/Policy/epoch{args.epoch}_iter{args.iter}.pth"
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'img_height': 96,
                'img_width': 160,
                'goal': [20, 20, 2],           # the goal
                'env': 'flightmare',           # use Realsense D435 or Flightmare Simulator ('435' or 'flightmare')
                'pitch_angle_deg': -5,         # pitch of camera, ensure consistent with the simulator or your platform (no need to re-collect and re-train when modifying)
                'odom_topic': '/juliett/ground_truth/odom',
                'depth_topic': '/depth_image',
                'verbose': False,              # print the latency?
                'visualize': True              # visualize all predictions? set False in real flight
                }
    YopoNet(settings, weight)


if __name__ == "__main__":
    main()
