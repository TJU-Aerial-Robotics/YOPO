import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import numpy as np
import cv2
import os
import torch
import argparse
import time
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
from flightpolicy.yopo.yopo_policy import YopoPolicy
from flightpolicy.yopo.primitive_utils import LatticeParam, LatticePrimitive

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        self.bridge = CvBridge()
        self.odom = Odometry()
        self.odom_ref = Odometry()
        self.height = self.config['img_height']
        self.width = self.config['img_width']
        self.depth = np.zeros((1, 1, self.config['img_height'], self.config['img_width']))
        self.goal = np.array(self.config['goal'])
        self.env = self.config['env']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.new_odom = False
        self.new_depth = False
        self.odom_ref_init = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/traj_opt.yaml", 'r'))
        self.lattice_space = LatticeParam(cfg)
        self.lattice_primitive = LatticePrimitive(self.lattice_space)

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.count = 0
        self.count_interpolation = 0

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
        self.endstate_pub = rospy.Publisher("/yopo_net/pred_endstate", Float32MultiArray, queue_size=1)
        self.all_endstate_pub = rospy.Publisher("/yopo_net/pred_endstates", Float32MultiArray, queue_size=1)
        self.goal_pub = rospy.Publisher("/yopo_net/goal", Float32MultiArray, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.odom_ref_sub = rospy.Subscriber("/juliett/state_ref/odom", Odometry, self.callback_odometry_ref, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        self.timer_net = rospy.Timer(rospy.Duration(1. / self.config['network_frequency']), self.test_policy)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        print("New goal:", self.goal)

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.odom_ref_init:
            self.new_odom = True

    # the following frame (The planner is planning from the desired state, instead of the actual state)
    def callback_odometry_ref(self, data):
        if not self.odom_ref_init:
            self.odom_ref_init = True
        self.odom_ref = data
        self.new_odom = True

    def process_odom(self):
        # Rwb
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)

        if self.odom_ref_init:
            odom_data = self.odom_ref
            # vel_b
            vel_w = np.array([odom_data.twist.twist.linear.x, odom_data.twist.twist.linear.y, odom_data.twist.twist.linear.z])
            vel_b = np.dot(np.linalg.inv(self.Rotation_wc), vel_w)
            # acc_b (acc stored in angular in our ref_state topic)
            acc_w = np.array([odom_data.twist.twist.angular.x, odom_data.twist.twist.angular.y, odom_data.twist.twist.angular.z])
            acc_b = np.dot(np.linalg.inv(self.Rotation_wc), acc_w)
        else:
            odom_data = self.odom
            vel_b = np.array([0.0, 0.0, 0.0])
            acc_b = np.array([0.0, 0.0, 0.0])

        # pose and goal_dir
        pos = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, odom_data.pose.pose.position.z])
        goal_w = (self.goal - pos) / np.linalg.norm(self.goal - pos)
        goal_b = np.dot(np.linalg.inv(self.Rotation_wc), goal_w)

        vel_acc = np.concatenate((vel_b, acc_b), axis=0)
        vel_acc_norm = self.normalize_obs(vel_acc[np.newaxis, :])
        obs_norm = np.hstack((vel_acc_norm, goal_b[np.newaxis, :]))
        return obs_norm

    def callback_depth(self, data):
        max_dis = 20.0
        min_dis = 0.03
        scale = {'435': 0.001, 'flightmare': 1.0}.get(self.env, 1.0)

        try:
            depth_ = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except:
            print("CV_bridge ERROR: Possible solutions may be found at https://github.com/TJU-Aerial-Robotics/YOPO/issues/2")

        if depth_.shape[0] != self.height or depth_.shape[1] != self.width:
            depth_ = cv2.resize(depth_, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_ = np.minimum(depth_ * scale, max_dis) / max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        start = time.time()
        nan_mask = np.isnan(depth_) | (depth_ < min_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth_ * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth_ = interpolated_image.reshape([1, 1, self.height, self.width])
        if self.verbose:
            self.time_interpolation = self.time_interpolation + (time.time() - start)
            self.count_interpolation = self.count_interpolation + 1
            print("Time Consuming: interpolation:", self.time_interpolation / self.count_interpolation)

        # cv2.imshow("1", depth_[0][0])
        # cv2.waitKey(1)
        self.depth = depth_.astype(np.float32)
        self.new_depth = True

    # TODO: Move the test_policy to callback_depth directly?
    def test_policy(self, _timer):
        if self.new_depth and self.new_odom:
            self.new_odom = False
            self.new_depth = False
            obs = self.process_odom()
            odom_sec = self.odom.header.stamp.to_sec()

            # input prepare
            time0 = time.time()
            depth = torch.from_numpy(self.depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
            obs_norm_input = self.prepare_input_observation(obs)
            obs_norm_input = obs_norm_input.to(self.device, non_blocking=True)
            # torch.cuda.synchronize()

            # forward
            if self.use_trt:  # TensorRT (inference speed increased by 10x)
                time1 = time.time()
                trt_output = self.policy(depth, obs_norm_input)
                time2 = time.time()
                endstate_pred, score_pred = self.trt_process(trt_output, return_all_preds=self.visualize)
                endstate_pred = endstate_pred.squeeze()
                time3 = time.time()
            else:
                time1 = time.time()
                endstate_pred, score_pred = self.policy.predict(depth, obs_norm_input, return_all_preds=self.visualize)
                endstate_pred = endstate_pred.cpu().numpy().squeeze()
                score_pred = score_pred.cpu().numpy()
                time2 = time3 = time.time()

            # Transform the prediction(body frame) to the world frame with the attitude in inference
            # Replacing PyTorch calculations on CUDA with NumPy calculations on the CPU (speed increased by 10x)
            endstate_b = endstate_pred
            endstate_w = np.zeros_like(endstate_b)
            traj_num = self.lattice_space.horizon_num * self.lattice_space.vertical_num if self.visualize else 1
            Pb, Vb, Ab = [np.zeros((3, traj_num)) for _ in range(3)]
            for i in range(3):
                Pb[i] = endstate_b[3 * i]
                Vb[i] = endstate_b[3 * i + 1]
                Ab[i] = endstate_b[3 * i + 2]
            # pos_actual = np.array([self.odom.pose.pose.position.x,
            #                        self.odom.pose.pose.position.y,
            #                        self.odom.pose.pose.position.z])
            Pw = np.dot(self.Rotation_wc, Pb)  # + np.tile(pos_actual, (15, 1)).T
            Vw = np.dot(self.Rotation_wc, Vb)
            Aw = np.dot(self.Rotation_wc, Ab)
            for i in range(3):
                endstate_w[3 * i] = Pw[i]
                endstate_w[3 * i + 1] = Vw[i]
                endstate_w[3 * i + 2] = Aw[i]

            if self.verbose:
                self.time_prepare = self.time_prepare + (time1 - time0)
                self.time_forward = self.time_forward + (time2 - time1)
                self.time_process = self.time_process + (time3 - time2)
                self.count = self.count + 1
                print("Time Consuming: prepare:", self.time_prepare / self.count, "; forward:", self.time_forward / self.count,
                      "; process:", self.time_process / self.count)

            # publish
            if not self.visualize:
                endstate_pred_to_pub = Float32MultiArray(data=endstate_w.reshape(-1))
                endstate_pred_to_pub.layout.data_offset = int(1000 * odom_sec) % 1000000  # 预测时用的里程计时间戳(ms)
                self.endstate_pub.publish(endstate_pred_to_pub)
            else:
                action_id = np.argmin(score_pred)
                best_endstate_pred = endstate_w[:, action_id].reshape(-1)
                endstate_pred_to_pub = Float32MultiArray(data=best_endstate_pred)
                endstate_pred_to_pub.layout.data_offset = int(1000 * odom_sec) % 1000000  # 预测时用的里程计时间戳(ms)
                self.endstate_pub.publish(endstate_pred_to_pub)
                # visualization
                endstate_score_preds = np.concatenate((endstate_w, score_pred), axis=0)
                all_endstate_pred = Float32MultiArray(data=endstate_score_preds.T.reshape(-1))
                all_endstate_pred.layout.dim.append(MultiArrayDimension())
                all_endstate_pred.layout.dim[0].size = endstate_score_preds.shape[1]
                all_endstate_pred.layout.dim[0].label = "primitive_num"
                all_endstate_pred.layout.dim.append(MultiArrayDimension())
                all_endstate_pred.layout.dim[1].size = endstate_score_preds.shape[0]
                all_endstate_pred.layout.dim[1].label = "endstate_and_score_num"
                self.all_endstate_pub.publish(all_endstate_pred)
            self.goal_pub.publish(Float32MultiArray(data=self.goal))
        # start a new round
        elif not self.new_odom:
            self.odom_ref_init = False

    def trt_process(self, input_tensor: torch.Tensor, return_all_preds=False) -> torch.Tensor:
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.cpu().numpy()
        input_tensor = input_tensor.reshape(batch_size, 10, self.lattice_space.horizon_num * self.lattice_space.vertical_num)
        endstate_pred = input_tensor[:, 0:9, :]
        score_pred = input_tensor[:, 9, :]

        if not return_all_preds:
            endstate_prediction = np.zeros((batch_size, 9))
            score_prediction = np.zeros((batch_size, 1))
            for i in range(0, batch_size):
                action_id = np.argmin(score_pred[i])
                lattice_id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - action_id
                endstate_prediction[i] = self.pred_to_endstate(np.expand_dims(endstate_pred[i, :, action_id], axis=0), lattice_id)
                score_prediction[i] = score_pred[i, action_id]
        else:
            endstate_prediction = np.zeros_like(endstate_pred)
            score_prediction = score_pred
            for i in range(0, self.lattice_space.horizon_num * self.lattice_space.vertical_num):
                lattice_id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - i
                endstate = self.pred_to_endstate(endstate_pred[:, :, i], lattice_id)
                endstate_prediction[:, :, i] = endstate

        return endstate_prediction, score_prediction

    def prepare_input_observation(self, obs):
        """
            convert the observation from body frame to primitive frame,
            and then concatenate it with the depth features (to ensure the translational invariance)
        """
        obs_return = np.ones((obs.shape[0], self.lattice_space.vertical_num, self.lattice_space.horizon_num, obs.shape[1]), dtype=np.float32)
        id = 0
        v_b = obs[:, 0:3]
        a_b = obs[:, 3:6]
        g_b = obs[:, 6:9]
        for i in range(self.lattice_space.vertical_num - 1, -1, -1):
            for j in range(self.lattice_space.horizon_num - 1, -1, -1):
                Rbp = self.lattice_primitive.getRotation(id)
                v_p = np.dot(Rbp.T, v_b.T).T
                a_p = np.dot(Rbp.T, a_b.T).T
                g_p = np.dot(Rbp.T, g_b.T).T
                obs_return[:, i, j, 0:3] = v_p
                obs_return[:, i, j, 3:6] = a_p
                obs_return[:, i, j, 6:9] = g_p
                # obs_return[:, i, j, 0:6] = self.normalize_obs(obs_return[:, i, j, 0:6])
                id = id + 1
        obs_return = np.transpose(obs_return, [0, 3, 1, 2])
        return torch.from_numpy(obs_return)

    def pred_to_endstate(self, endstate_pred: np.ndarray, id: int):
        """
            Transform the predicted state to the body frame.
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = endstate_pred[:, 2] * self.lattice_space.radio_range + self.lattice_space.radio_range
        yaw, pitch = self.lattice_primitive.getAngleLattice(id)
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_space.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_space.acc_max
        Rbp = self.lattice_primitive.getRotation(id)
        endstate_vb = np.matmul(Rbp, endstate_vp.T).T
        endstate_ab = np.matmul(Rbp, endstate_ap.T).T
        endstate = np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)
        endstate[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]] = endstate[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        return endstate

    def normalize_obs(self, vel_acc):
        vel_norm = vel_acc[:, 0:3] / self.lattice_space.vel_max
        acc_norm = vel_acc[:, 3:6] / self.lattice_space.acc_max
        return np.hstack((vel_norm, acc_norm))

    def warm_up(self):
        depth = np.zeros(shape=[1, 1, self.height, self.width], dtype=np.float32)
        obs = np.zeros(shape=[1, 9], dtype=np.float32)
        obs_input = self.prepare_input_observation(obs)
        if self.use_trt:
            trt_output = self.policy(torch.from_numpy(depth).to(self.device), obs_input.to(self.device))
            self.trt_process(trt_output, return_all_preds=True)
        else:
            self.policy.predict(torch.from_numpy(depth).to(self.device), obs_input.to(self.device), return_all_preds=True)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=0, help="epoch number")
    parser.add_argument("--iter", type=int, default=0, help="iter number")
    return parser


# In realworld flight: visualize=False; use_tensorrt=True, and ensure the pitch_angle consistent with your platform
# When modifying the pitch_angle, there's no need to re-collect and re-train, as all predictions are in the camera coordinate system
def main():
    args = parser().parse_args()
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    if args.use_tensorrt:
        weight = "yopo_trt.pth"
    else:
        weight = rsg_root + "/saved/YOPO_{}/Policy/epoch{}_iter{}.pth".format(args.trial, args.epoch, args.iter)

    settings = {'use_tensorrt': args.use_tensorrt,
                'network_frequency': 30,
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
