import os
# MINCO 的 12x12 求逆用单线程 BLAS 更快且无抖动；多线程 BLAS 在小矩阵上会偶发卡顿几十 ms
# （post-process 偶发大延迟的根因）。必须在 numpy/BLAS 导入前设置。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from threading import Lock
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import cv2
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.poly_solver import MincoTraj, calculate_yaw
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.array(self.config['goal'])
        self.goal_length = float(cfg['goal_length'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.topk = self.config.get('topk', 1)  # pick the traj closest to last inner among the top-K best
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
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
        self.optimal_traj = None
        self.best_inner_w = None       # world-frame inner waypoint of best traj (for marker viz)
        self.best_tail_pos_w = None    # world-frame tail position of best traj (for marker viz)
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time   # total = 2 × piece_duration
        self.piece_duration = self.traj_time / cfg["piece_num"]
        # predicted safety corridor of the selected traj (map-free): radii = unwarp(μ), alpha = confidence(b)
        self.radius_num = int(cfg["radius_num"])
        self.radius_lambda = float(cfg["radius_warp_lambda"])
        self.radius_b_min = float(cfg["radius_b_min"])
        self.radius_b_max = float(cfg["radius_b_max"])
        self.safe_radius = 0.05     # corridor radius (m) a traj must clear to be eligible; none clears it -> brake
        self.safe_mu = 1.0 - np.exp(-self.safe_radius / self.radius_lambda)   # warped, compared against μ directly
        self.brake = False

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30  # used only as processing time tolerance for printing logs

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", MarkerArray, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", MarkerArray, queue_size=1)
        self.corridor_pub = rospy.Publisher("/yopo_net/corridor_visual", MarkerArray, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", MarkerArray, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        # ros timer
        rospy.sleep(1.0)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def _odom_pos(self):
        p = self.odom.pose.pose.position
        return np.array((p.x, p.y, p.z))

    def _odom_vel(self):
        v = self.odom.twist.twist.linear
        return np.array((v.x, v.y, v.z))

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        self.arrive = False
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    def callback_odometry(self, data):
        """Odom subscriber: cache the latest odom, seed the desire state on the first frame, flag arrival."""
        self.odom = data
        if not self.desire_init:
            self.desire_pos = self._odom_pos()
            self.desire_vel = self._odom_vel()
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = self._odom_pos()
        if np.linalg.norm(pos - self.goal) < 5 and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def process_odom(self):
        """Build the normalized network obs (vel/acc/goal in the camera frame) from the current odom + goal."""
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else self._odom_vel()
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir: z-height is prioritized
        goal_w = self.goal - self.desire_pos
        if np.linalg.norm(goal_w) > self.goal_length:
            gz = np.clip(goal_w[2], -self.goal_length, self.goal_length)
            hxy = goal_w[:2] / (np.linalg.norm(goal_w[:2]) + 1e-9) * np.sqrt(self.goal_length ** 2 - gz ** 2)
            goal_w = np.array([hxy[0], hxy[1], gz])
        self.goal_local_w = goal_w        # world-frame local goal offset, for visualization
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        return self.state_transform.normalize_obs_cpu(obs[None, :])

    @torch.inference_mode()
    def callback_depth(self, data):
        """Depth subscriber: the plan-once-per-frame pipeline (depth → inference → decode → solve → viz)."""
        if not self.odom_init: return

        # 1. depth image → normalized, inpainted (1,1,H,W)
        time0 = time.time()
        depth = self._process_depth(data)

        # 2. network inference (depth + current state → raw traj params + scores)
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input_cpu(self.process_odom())
        obs_input = torch.from_numpy(obs_input).to(self.device, non_blocking=True)
        time2 = time.time()
        endstate_pred, score_pred, radius_pred = self.policy(depth_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        radius_pred = radius_pred.cpu().numpy()
        time3 = time.time()

        # 3. decode + body→world (rotation only; translation added when solving)
        inner_pos_b, tail_pva_b, durations_b, score = self.process_output(endstate_pred, score_pred)
        inner_pos_w = inner_pos_b @ self.Rotation_wc.T                    # (N, 3)
        tail_pva_w = np.einsum('ij,...kj->...ki', self.Rotation_wc, tail_pva_b)  # rotate each pva row

        # 4. select the best traj (corridor admission + score) + solve its MINCO (under the control lock)
        mu_lo = self._corridor_mu_lo(radius_pred)
        start_pos, start_vel, action_id, best_durations, brake = self._solve_best_traj(
            score, inner_pos_w, tail_pva_w, durations_b, mu_lo)
        time4 = time.time()

        if brake and not self.brake:   # log once, on entering brake
            rospy.logwarn(f"BRAKE: no trajectory clears a {self.safe_radius} m corridor")
        self.brake = brake

        # predicted safety corridor (lazy: only when subscribed) + trajectory viz + timing
        if self.corridor_pub.get_num_connections() > 0:
            self._publish_corridor(radius_pred, action_id, best_durations)
        self.visualize_trajectory(score, inner_pos_w, tail_pva_w, durations_b, start_pos, start_vel)
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def _process_depth(self, data):
        """Decode the depth msg → clip to max_dis, normalize to [0,1], inpaint NaN/near holes.
        Returns (1, 1, H, W) float32."""
        if data.encoding == "32FC1":
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis

        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        inpainted = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS).astype(np.float32) / 255.0
        return inpainted.reshape([1, 1, self.height, self.width])

    def _solve_best_traj(self, score, inner_pos_w, tail_pva_w, durations_b, mu_lo):
        """Pick the best traj (safety admission + score + top-k inner continuity) and solve its MINCO,
        storing the result for control_pub. Runs under the control lock."""
        with self.lock:
            start_pos = self.desire_pos if self.plan_from_reference else self._odom_pos()
            start_vel = self.desire_vel if self.plan_from_reference else self._odom_vel()
            head_pva_w = np.stack([start_pos, start_vel, self.desire_acc], axis=0)  # (3, 3)

            action_id, brake = self.select_action(score, inner_pos_w + start_pos, mu_lo)
            # inner/tail world positions are body-frame offsets rotated, plus drone position
            best_inner_w = inner_pos_w[action_id] + start_pos
            best_tail_pva_w = tail_pva_w[action_id].copy()
            best_tail_pva_w[0] += start_pos                # only position needs translation
            best_durations = durations_b[action_id]        # (2,) per-piece durations

            self.optimal_traj = MincoTraj().solve(head_pva_w, best_tail_pva_w, best_inner_w, durations=best_durations)
            self.best_inner_w = best_inner_w
            self.best_tail_pos_w = best_tail_pva_w[0]
            self.best_total_time = float(best_durations.sum())
            self.ctrl_time = 0.0
        return start_pos, start_vel, action_id, best_durations, brake

    def _corridor_mu_lo(self, radius_pred):
        """Tightest conservative corridor per candidate, warped: min over the nr balls of (μ − b).
        Subtracting b discounts the model's uncertainty. Returns (N,), image order."""
        m = radius_pred[0].reshape(2 * self.radius_num, -1)                 # (2nr, N)
        return (m[:self.radius_num] - m[self.radius_num:]).min(axis=0)

    def _publish_corridor(self, radius_pred, sel_img, durations):
        """Selected traj's predicted corridor (map-free): μ/b at radius_num time-uniform instants;
        centers = traj at those instants, radii = unwarp(μ) = -λ·log(1-μ), alpha = confidence(b)."""
        nr = self.radius_num
        row = radius_pred[0].reshape(2 * nr, -1)[:, sel_img]     # (2nr,) channels of the selected cell
        mu, b = row[:nr], row[nr:]
        radii = -self.radius_lambda * np.log1p(-np.clip(mu, 0.0, 1.0 - 1e-4))
        centers = self.optimal_traj.position(np.arange(1, nr + 1) / nr * float(durations.sum()))   # (nr, 3)
        b_ref = 0.1   # viz-only upper bound (learned b ≈ 0.06-0.09; the true b_max=0.5 washes alpha out)
        conf = 1.0 - (b - self.radius_b_min) / max(b_ref - self.radius_b_min, 1e-6)
        self.corridor_pub.publish(self._build_corridor_markers(centers, radii, 0.06 + 0.4 * np.clip(conf, 0.0, 1.0)))

    def _build_corridor_markers(self, centers, radii, alpha):
        """One SPHERE per corridor ball (diameter 2·radius), blue, alpha = confidence."""
        now = rospy.Time.now()
        msgs = MarkerArray()
        for i, (c, r, a) in enumerate(zip(centers, radii, alpha)):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = now
            m.ns = 'pred_corridor'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, c)
            m.scale.x = m.scale.y = m.scale.z = max(2.0 * float(r), 0.05)
            m.color = ColorRGBA(0.1, 0.6, 1.0, float(a))
            msgs.markers.append(m)
        return msgs

    def control_pub(self, _timer):
        """Control timer: sample the current MINCO trajectory at ctrl_time and publish the pos command + yaw."""
        if self.ctrl_time is None or self.ctrl_time > getattr(self, "best_total_time", self.traj_time):
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return
        if self.brake and self.last_control_msg is not None:
            goal_dir = self.goal - self._odom_pos()
            yaw, yawdot = calculate_yaw(goal_dir, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.last_control_msg.yaw, self.last_control_msg.yaw_dot = yaw, yawdot
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            pos = self.optimal_traj.position(self.ctrl_time)
            vel = self.optimal_traj.velocity(self.ctrl_time)
            acc = self.optimal_traj.acceleration(self.ctrl_time)

            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x, control_msg.position.y, control_msg.position.z = pos
            control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z = vel
            control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z = acc

            self.desire_pos = np.array(pos)
            self.desire_vel = np.array(vel)
            self.desire_acc = np.array(acc)
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def select_action(self, score, inner_w, mu_lo):
        """Highest score among the trajs whose conservative corridor clears safe_mu (with topk > 1:
        among their top-K, the inner waypoint closest to the last executed one).
        Returns (action_id, brake); brake means none cleared it, so the best-scoring traj is used."""
        cand = np.flatnonzero(mu_lo >= self.safe_mu)
        if cand.size == 0:
            return int(np.argmax(score)), True
        if self.topk <= 1 or self.best_inner_w is None:
            return int(cand[np.argmax(score[cand])]), False
        k = min(self.topk, cand.size)
        topk_ids = cand[np.argpartition(score[cand], -k)[-k:]]
        dists = np.linalg.norm(inner_w[topk_ids] - self.best_inner_w, axis=1)
        return int(topk_ids[np.argmin(dists)]), False

    def process_output(self, endstate_pred, score_pred):
        """
        endstate_pred: (1, 14, V, H)  — raw tanh-space traj params
        score_pred:    (1, V, H)      — softplus scores per cell
        Returns (all N trajectories, rows in image order):
            inner_pos_b: (N, 3) body frame
            tail_pva_b:  (N, 3, 3) body frame (rows: pos, vel, acc)
            durations:   (N, 2)
            score:       (N,)
        """
        lp = self.lattice_primitive
        V, H, N = lp.vertical_num, lp.horizon_num, lp.traj_num

        # (1, 14, V, H) → (V, H, 14) → (N, 14) — image order
        endstate_pred = endstate_pred.reshape(14, V, H).transpose(1, 2, 0).reshape(N, 14)
        lattice_ids = lp.convert_ImageGrid_LatticeID(np.arange(N))
        inner_pos_b, tail_pva_b, durations = self.state_transform.pred_to_traj_params_cpu(
            endstate_pred, lattice_ids)
        return inner_pos_b, tail_pva_b, durations, score_pred.reshape(N)

    def visualize_trajectory(self, scores, inner_pos_w_all, tail_pva_w_all, durations_all, start_pos, start_vel):
        """Publish RViz markers (only for topics with subscribers): best traj, lattice primitives, all scored trajs."""
        # use the best-traj total time for the time grid; clamp inside MincoTraj handles others
        t_max = getattr(self, "best_total_time", self.traj_time)
        dt = t_max / 20.0
        t_values = np.arange(0, t_max, dt)

        # best predicted trajectory — line strip + spheres at (inner, tail)
        if self.best_traj_pub.get_num_connections() > 0 and self.optimal_traj is not None:
            points_array = self.optimal_traj.position(t_values)  # (K, 3)
            self.best_traj_pub.publish(self._build_best_traj_markers(
                points_array, self.best_inner_w, self.best_tail_pos_w))

        # lattice primitive visualization (grid center as inner waypoint, local goal as tail; default duration)
        if self.lattice_traj_pub.get_num_connections() > 0:
            lattice_b = self.state_transform.lattice_pos_np  # (N, 3) body-frame grid centers
            lattice_w = lattice_b @ self.Rotation_wc.T                          # rotated to world
            N = lattice_w.shape[0]
            head_b = np.tile(np.stack([start_pos, start_vel, np.zeros(3)], axis=0)[None], (N, 1, 1))  # (N, 3, 3)
            inner_w = lattice_w + start_pos
            tail_pva = np.zeros((N, 3, 3))
            tail_pva[:, 0] = self.goal_local_w + start_pos
            traj_batch = MincoTraj().solve(head_b, tail_pva, inner_w, durations=np.full(2, self.piece_duration))
            pts = traj_batch.position(np.linspace(0, 2 * self.piece_duration, 20))  # (N, 20, 3)
            self.lattice_traj_pub.publish(
                self._build_all_traj_markers(pts, np.zeros(N), inner_w, ns='yopo_lattice'))

        # all predicted trajectories — as MarkerArray lines, colored by score
        if self.all_trajs_pub.get_num_connections() > 0:
            N = inner_pos_w_all.shape[0]
            head_pva_batch = np.tile(np.stack([start_pos, start_vel, self.desire_acc], axis=0)[None], (N, 1, 1))
            inner_w_batch = inner_pos_w_all + start_pos
            tail_pva_batch = tail_pva_w_all.copy()
            tail_pva_batch[:, 0] += start_pos
            traj_batch = MincoTraj().solve(head_pva_batch, tail_pva_batch, inner_w_batch, durations=durations_all)
            t_max_all = float(durations_all.sum(axis=-1).max())
            t_values_all = np.arange(0, t_max_all, t_max_all / 20.0)
            pts = traj_batch.position(t_values_all)  # (N, K, 3)

            self.all_trajs_pub.publish(self._build_all_traj_markers(pts, scores, inner_w_batch))

    def _build_best_traj_markers(self, line_pts, inner_pt, tail_pt):
        """
        Build a MarkerArray for the best trajectory:
          - id=0 LINE_STRIP: the sampled trajectory curve
          - id=1 SPHERE_LIST: two small spheres at the inner waypoint and the tail
        """
        now = rospy.Time.now()
        msgs = MarkerArray()

        line = Marker()
        line.header.frame_id = 'world'
        line.header.stamp = now
        line.ns = 'yopo_best'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08            # line width
        line.color = ColorRGBA(0.0, 1.0, 0.2, 1.0)
        line.points = [Point(float(p[0]), float(p[1]), float(p[2])) for p in line_pts]

        balls = Marker()
        balls.header.frame_id = 'world'
        balls.header.stamp = now
        balls.ns = 'yopo_best'
        balls.id = 1
        balls.type = Marker.SPHERE_LIST
        balls.action = Marker.ADD
        balls.pose.orientation.w = 1.0
        balls.scale.x = balls.scale.y = balls.scale.z = 0.3   # sphere diameter
        balls.points = [
            Point(float(inner_pt[0]), float(inner_pt[1]), float(inner_pt[2])),
            Point(float(tail_pt[0]),  float(tail_pt[1]),  float(tail_pt[2])),
        ]
        balls.colors = [
            ColorRGBA(1.0, 0.6, 0.0, 1.0),   # inner = orange
            ColorRGBA(1.0, 0.0, 0.0, 1.0),   # tail  = red
        ]

        msgs.markers = [line, balls]
        return msgs

    def _build_all_traj_markers(self, line_pts_batch, scores, inner_w_batch, ns='yopo_all'):
        """
        Build a MarkerArray for all predicted trajectories:
          - each trajectory gets a LINE_STRIP (curve) and a SPHERE (inner waypoint),
            both colored by its score (equal scores → one uniform colour).
        Args:
            line_pts_batch: (N, K, 3) — sampled trajectory points for each trajectory
            scores: (N,) — score per trajectory
            inner_w_batch: (N, 3) — inner waypoint positions in world frame
            ns: marker namespace
        """
        now = rospy.Time.now()
        msgs = MarkerArray()

        # normalize scores to [0, 1] for color mapping (higher score = better)
        score_min, score_max = scores.min(), scores.max()
        if score_max - score_min > 1e-6:
            scores_norm = (scores - score_min) / (score_max - score_min)
        else:
            scores_norm = np.ones_like(scores) * 0.5

        for i in range(line_pts_batch.shape[0]):
            s = scores_norm[i]
            color = ColorRGBA(1.0 - s, s, 0.0, 0.5)  # green (high) to red (low), semi-transparent

            # trajectory line
            line = Marker()
            line.header.frame_id = 'world'
            line.header.stamp = now
            line.ns = ns
            line.id = i * 2
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.06  # line width
            line.color = color
            line.points = [Point(float(p[0]), float(p[1]), float(p[2])) for p in line_pts_batch[i]]
            msgs.markers.append(line)

            # inner waypoint sphere
            ball = Marker()
            ball.header.frame_id = 'world'
            ball.header.stamp = now
            ball.ns = ns
            ball.id = i * 2 + 1
            ball.type = Marker.SPHERE
            ball.action = Marker.ADD
            ball.pose.position.x = float(inner_w_batch[i, 0])
            ball.pose.position.y = float(inner_w_batch[i, 1])
            ball.pose.position.z = float(inner_w_batch[i, 2])
            ball.pose.orientation.w = 1.0
            ball.scale.x = ball.scale.y = ball.scale.z = 0.25  # sphere diameter
            ball.color = color
            msgs.markers.append(ball)

        return msgs

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """Accumulate + log per-stage timing (interp/prepare/inference/post/viz); warn when a frame exceeds the fps budget."""
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        """Run one dummy forward + MINCO solve to pay the first-call CUDA / BLAS init cost up front."""
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred, _radius = self.policy(depth, obs)
        _ = self.state_transform.pred_to_traj_params(endstate_pred)
        _ = MincoTraj().solve(np.zeros((3, 3)), np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([0.5, 0, 0]), durations=np.array([self.piece_duration, self.piece_duration]))



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,      # run the TensorRT engine instead of PyTorch
                'goal': [50, 0, 2],                     # initial goal (world xyz); RViz 2D Nav Goal overrides it
                'topk': 1,                              # >=1: among the top-K scores, keep the traj closest to the last one
                'pitch_angle_deg': -0,                  # camera pitch w.r.t. the body (upward is negative)
                'odom_topic': '/sim/odom',              # odometry topic (FLU)
                'depth_topic': '/depth_image',          # depth image topic
                'ctrl_topic': '/so3_control/pos_cmd',   # control command topic (FLU)
                'plan_from_reference': False,           # set True when flying with a position controller
                'verbose': False                        # print the per-stage timing every frame
                }
    YopoNet(settings, weight)
