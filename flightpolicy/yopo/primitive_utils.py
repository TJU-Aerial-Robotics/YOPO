import numpy as np
from scipy.spatial.transform import Rotation as R


class LatticeParam():
    def __init__(self, cfg):
        self.vel_max = cfg["vel_max"]
        self.segment_time = 2 * cfg["radio_range"] / self.vel_max
        self.horizon_num = cfg["horizon_num"]
        self.vertical_num = cfg["vertical_num"]
        self.radio_num = cfg["radio_num"]
        self.vel_num = cfg["vel_num"]
        self.horizon_fov = cfg["horizon_camera_fov"] * (self.horizon_num - 1) / self.horizon_num
        self.vertical_fov = cfg["vertical_camera_fov"] * (self.vertical_num - 1) / self.vertical_num
        self.horizon_anchor_fov = cfg["horizon_anchor_fov"]
        self.vertical_anchor_fov = cfg["vertical_anchor_fov"]
        self.radio_range = cfg["radio_range"]
        self.vel_fov = cfg["vel_fov"]
        self.vel_prefile = cfg["vel_prefile"]
        self.acc_max = self.vel_max / self.segment_time
        print("---------------------")
        print("| max speed = ", round(self.vel_max, 1), " |")
        print("| traj time = ", round(self.segment_time, 1), " |")
        print("| max radio = ", round(2 * self.radio_range, 1), " |")
        print("---------------------")


# ID in images:
#   [8, 7, 6,
#    5, 4, 3,
#    2, 1, 0]
class LatticePrimitive():
    def __init__(self, LatticeParam):
        self.lattice_param = LatticeParam

        if self.lattice_param.horizon_num == 1:
            direction_diff = 0
        else:
            direction_diff = (self.lattice_param.horizon_fov / 180.0 * np.pi) / (self.lattice_param.horizon_num - 1)
        if self.lattice_param.vertical_num == 1:
            altitude_diff = 0
        else:
            altitude_diff = (self.lattice_param.vertical_fov / 180.0 * np.pi) / (self.lattice_param.vertical_num - 1)
        radio_diff = self.lattice_param.radio_range / self.lattice_param.radio_num
        if self.lattice_param.vel_num == 1:
            vel_dir_diff = 0
        else:
            vel_dir_diff = (self.lattice_param.vel_fov / 180.0 * np.pi) / (self.lattice_param.vel_num - 1)

        lattice_pos_list = []
        lattice_vel_list = []
        lattice_angle_list = []
        self.lattice_Rbp_list = []

        # Primitives: Bottom to Top, Right to Left
        # We retain the code of sampling primitives with different velocity directions and length,
        # hope to predict multiple outputs in each grid like YOLO, but it does not work well.
        for h in range(0, self.lattice_param.radio_num):
            for i in range(0, self.lattice_param.vertical_num):
                for j in range(0, self.lattice_param.horizon_num):
                    for k in range(0, self.lattice_param.vel_num):
                        search_radio = (h + 1) * radio_diff
                        alpha = -direction_diff * (self.lattice_param.horizon_num - 1) / 2 + j * direction_diff
                        beta = -altitude_diff * (self.lattice_param.vertical_num - 1) / 2 + i * altitude_diff
                        gamma = -vel_dir_diff * (self.lattice_param.vel_num - 1) / 2 + k * vel_dir_diff

                        pos_node = [np.cos(beta) * np.cos(alpha) * search_radio,
                                    np.cos(beta) * np.sin(alpha) * search_radio,
                                    np.sin(beta) * search_radio]
                        vel_node = [np.cos(alpha + gamma) * self.lattice_param.vel_prefile,
                                    np.sin(alpha + gamma) * self.lattice_param.vel_prefile,
                                    0.0]
                        lattice_pos_list.append(pos_node)
                        lattice_vel_list.append(vel_node)
                        lattice_angle_list.append([alpha, beta])
                        # inner rotation: yaw-pitch-roll
                        Rotation = R.from_euler('ZYX', [alpha, -beta, 0.0], degrees=False)
                        self.lattice_Rbp_list.append(Rotation.as_matrix().astype(np.float32))

        self.lattice_pos_node = np.array(lattice_pos_list)
        self.lattice_vel_node = np.array(lattice_vel_list)
        self.lattice_angle_node = np.array(lattice_angle_list)

        self.yaw_diff = 0.5 * self.lattice_param.horizon_anchor_fov / 180.0 * np.pi
        self.pitch_diff = 0.5 * self.lattice_param.vertical_anchor_fov / 180.0 * np.pi

    def getStateLattice(self, id):
        return self.lattice_pos_node[id, :], self.lattice_vel_node[id, :]

    # yaw, pitch
    def getAngleLattice(self, id):
        return self.lattice_angle_node[id, 0], self.lattice_angle_node[id, 1]

    def getRotation(self, id):
        return self.lattice_Rbp_list[id]


class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ 5-th order polynomial at each Axis """
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_snap(self, t):
        """Return the scalar jerk at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t):
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t):
        """Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
            5 * self.A[5] * t * t * t * t

    def get_position(self, t):
        """Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
            self.A[5] * t * t * t * t * t


class Polys5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ multiple 5-th order polynomials at each Axis (only used for visualization of multiple trajectories) """
        N = len(pos1)
        State_Mat = np.array([[pos0] * N, [vel0] * N, [acc0] * N, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_position(self, t):
        """Return the position array at time t."""
        t = np.atleast_1d(t)
        result = (self.A[0][:, np.newaxis] + self.A[1][:, np.newaxis] * t + self.A[2][:, np.newaxis] * t ** 2 +
                  self.A[3][:, np.newaxis] * t ** 3 + self.A[4][:, np.newaxis] * t ** 4 + self.A[5][:, np.newaxis] * t ** 5 )
        return result.flatten()

"""
From body to world
p_w = Rwb * p_b + t_w
"""

def rotate(q_wb, pos_b):  # quat: wxzy
    pos_w = np.zeros_like(pos_b)
    if q_wb.ndim == 1:
        Rotation_wb = R.from_quat([q_wb[1], q_wb[2], q_wb[3], q_wb[0]])  # xyzw
        pos_w[:] = np.dot(Rotation_wb.as_matrix(), pos_b[:])
    else:
        for i in range(0, q_wb.shape[0]):
            Rotation_wb = R.from_quat([q_wb[i, 1], q_wb[i, 2], q_wb[i, 3], q_wb[i, 0]])  # xyzw
            pos_w[i, :] = np.dot(Rotation_wb.as_matrix(), pos_b[i, :])
    return pos_w

def transform(q_wb, tw, pos_b):
    pos_w = rotate(q_wb, pos_b)
    return pos_w + tw


"""
From world to body
p_b = Rbw * (p_w - t_w)
"""

def rotate_inv(q_wb, pos_w):  # quat: wxzy
    pos_b = np.zeros_like(pos_w)
    if q_wb.ndim == 1:
        Rotation_bw = R.from_quat([-q_wb[1], -q_wb[2], -q_wb[3], q_wb[0]])  # xyzw
        pos_b[:] = np.dot(Rotation_bw.as_matrix(), pos_w[:])
    else:
        for i in range(0, q_wb.shape[0]):
            Rotation_bw = R.from_quat([-q_wb[i, 1], -q_wb[i, 2], -q_wb[i, 3], q_wb[i, 0]])  # xyzw
            pos_b[i, :] = np.dot(Rotation_bw.as_matrix(), pos_w[i, :])
    return pos_b

def transform_inv(q_wb, tw, pos_w):
    pos_b = rotate_inv(q_wb, pos_w - tw)
    return pos_b


def calculate_yaw(vel_dir, goal_dir, last_yaw_, dt, max_yaw_rate=0.3):
    YAW_DOT_MAX_PER_SEC = max_yaw_rate * np.pi
    # Normalize direction of velocity
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)

    # Direction of goal
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)  # Prevent division by zero

    # Desired direction
    dir_des = vel_dir + goal_dir

    # Temporary yaw calculation
    yaw_temp = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.2 else last_yaw_
    max_yaw_change = YAW_DOT_MAX_PER_SEC * dt

    # Initialize yaw and yawdot
    yaw = last_yaw_
    yawdot = 0

    # Logic for yaw adjustment
    if yaw_temp - last_yaw_ > np.pi:
        if yaw_temp - last_yaw_ - 2 * np.pi < -max_yaw_change:
            yaw = last_yaw_ - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw_ > np.pi:
                yawdot = -YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw_) / dt
    elif yaw_temp - last_yaw_ < -np.pi:
        if yaw_temp - last_yaw_ + 2 * np.pi > max_yaw_change:
            yaw = last_yaw_ + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw_ < -np.pi:
                yawdot = YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw_) / dt
    else:
        if yaw_temp - last_yaw_ < -max_yaw_change:
            yaw = last_yaw_ - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -YAW_DOT_MAX_PER_SEC
        elif yaw_temp - last_yaw_ > max_yaw_change:
            yaw = last_yaw_ + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw_ > np.pi:
                yawdot = -YAW_DOT_MAX_PER_SEC
            elif yaw - last_yaw_ < -np.pi:
                yawdot = YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw_) / dt

    return yaw, yawdot