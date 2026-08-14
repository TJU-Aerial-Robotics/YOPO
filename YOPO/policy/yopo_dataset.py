import os, sys
import cv2
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


class YOPODataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(YOPODataset, self).__init__()
        # image params
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        # random state sampling: x = lognormal (forward bias), y/z = normal
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = cfg["acc_max_train"]
        self.v_mean = np.array(cfg["v_mean_unit"], dtype=float)   # [x, y, z]
        self.v_std = np.array(cfg["v_std_unit"], dtype=float)
        self.a_mean = np.array(cfg["a_mean_unit"], dtype=float)
        self.a_std = np.array(cfg["a_std_unit"], dtype=float)
        self.vx_lognorm_mean = np.log(1 - self.v_mean[0])         # vx lognormal from the x-axis params
        self.vx_logmorm_sigma = np.log(self.v_std[0])
        self.goal_length = cfg['goal_length']
        self.goal_pitch_std = cfg["goal_pitch_std"]
        self.goal_yaw_std = cfg["goal_yaw_std"]
        self.goal_height_min, self.goal_height_max = cfg["goal_height_range"]   # goal world-frame altitude band (m)

        # dataset
        if mode not in ('train', 'valid'):
            raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")
        split = 0 if mode == 'train' else 1   # column offset into train_test_split's (train, val) pairs
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])

        datafolders = sorted((f.path for f in os.scandir(data_dir) if f.is_dir()),
                             key=lambda x: int(os.path.basename(x)))
        if mode == 'train': self.print_data(datafolders)

        print("Loading", mode, "dataset")
        self.img_list, self.map_idx, positions, quaternions = [], [], [], []
        for data_idx, datafolder in enumerate(datafolders):
            image_file_names = [datafolder + "/" + f for f in os.listdir(datafolder)
                                if os.path.splitext(f)[1] == '.png']
            image_file_names.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split("_")[1]))  # align with label
            states = np.loadtxt(data_dir + f"/pose-{data_idx}.csv", delimiter=',', skiprows=1).astype(np.float32)

            parts = train_test_split(image_file_names, states[:, 0:3], states[:, 3:7],
                                     test_size=val_ratio, random_state=0)   # [img, pos, quat] × (train, val)
            imgs, pos, quat = parts[split], parts[2 + split], parts[4 + split]
            self.img_list.extend(imgs)
            self.map_idx.extend([data_idx] * len(imgs))
            positions.append(pos.astype(np.float32))
            quaternions.append(quat.astype(np.float32))

        self.positions = np.vstack(positions) if positions else np.empty((0, 3), np.float32)
        self.quaternions = np.vstack(quaternions) if quaternions else np.empty((0, 4), np.float32)

        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Images'      :<12} | Count: {len(self.img_list):<3} |  Shape: {self.width},{self.height}")
        print(f"{'Positions'   :<12} | Count: {self.positions.shape[0]:<3} |  Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions' :<12} | Count: {self.quaternions.shape[0]:<3} |  Shape: {self.quaternions.shape[1]}")
        print("==================================================")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        # 1. depth image + body frames (W: world, b: level-yaw body)
        image = self._read_depth(item)
        R_WB, R_Bw, _ = self._body_frames(item)

        # 2. random current state (rotated into the body frame)
        vel_w, acc_w = self._get_random_state()
        vel_b, acc_b = R_Bw.apply(vel_w), R_Bw.apply(acc_w)

        # 3. random goal
        goal_w = self._get_random_goal(self.positions[item, 2])
        goal_b = R_Bw.apply(goal_w)

        # obs = vel|acc|goal, body frame, NWU, unnormalized
        obs = np.hstack((vel_b, acc_b, goal_b)).astype(np.float32)
        rot_wb = R_WB.as_matrix().astype(np.float32)
        return image, self.positions[item], rot_wb, obs, self.map_idx[item]

    def _read_depth(self, item):
        """Read + resize the depth png to (1, H, W) float in [0, 1] (stored as int16, 0–20 m → 0–1)."""
        image = cv2.imread(self.img_list[item], -1).astype(np.float32)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST) / 65535.0
        return np.expand_dims(image, axis=0)

    def _body_frames(self, item):
        """From the stored quaternion: R_WB (body→world), R_Bw (world→level-yaw body, roll/pitch
        removed), and euler ZYX = [yaw, pitch, roll]."""
        q = self.quaternions[item, :]  # wxyz
        R_WB = R.from_quat([q[1], q[2], q[3], q[0]])
        euler = R_WB.as_euler('ZYX', degrees=False)
        R_Bw = R.from_euler('ZYX', [0, euler[1], euler[2]], degrees=False).inv()
        return R_WB, R_Bw, euler

    def _get_random_state(self):
        """Sample a random (vel, acc) in the level frame: x is right-skewed lognormal (cruise-forward
        bias), y/z Gaussian; reject norms above 1.2× the max as outliers."""
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max  # * 1.2 to ensure v_max can be sampled
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:  # avoid outliers
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:  # avoid outliers
                break
        return vel, acc

    def _get_random_goal(self, pos_z=1.0):
        """Random goal displacement in the level 'w' frame: sample yaw direction + distance (90% far
        at goal_length, 10% near), then override the height so the goal's world altitude lands in the
        configured band while keeping the total distance unchanged. pos_z is the drone's world height."""
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)
        goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(goal_yaw_angle)
        goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                               np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle), np.sin(goal_pitch_angle)])
        # 10% probability to generate a nearby goal (× goal_length is actual length)
        random_near = np.random.rand()
        if random_near < 0.1:
            goal_w_dir = random_near * 10 * goal_w_dir
        goal_w = self.goal_length * goal_w_dir
        # z from the world-height band (world height = pos_z + gz, clipped to ±gdist); horizontal
        # rescaled (Pythagoras) so the total distance stays gdist.
        gdist = np.linalg.norm(goal_w)
        gz = np.clip(np.random.uniform(self.goal_height_min, self.goal_height_max) - pos_z, -gdist, gdist)
        hxy = goal_w[:2]
        hxy = hxy / (np.linalg.norm(hxy) + 1e-9) * np.sqrt(max(gdist ** 2 - gz ** 2, 0.0))
        return np.array([hxy[0], hxy[1], gz])

    def print_data(self, datafolders):
        """Print the sampling-range table: 5–95% velocity/acceleration bands and goal yaw/pitch spans."""
        print("Datafolders:")
        for folder in datafolders:
            print("    ", folder)
        import scipy.stats as stats
        # 计算Vx 5% ~ 95% 区间
        p5 = self.vel_max * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_max * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_max * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_max * (self.v_mean + 2 * self.v_std)
        v_lower[0] = max(-p95 + 1.2 * self.vel_max, 0)
        v_upper[0] = -p5 + 1.2 * self.vel_max

        a_lower = self.acc_max * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_max * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 95% Range(m/s)  | Acc 95% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")
        print(f"| Goal Pitch 90% (deg)        | {-self.goal_pitch_std * 2:^9.1f}~{self.goal_pitch_std * 2:^9.1f} |")
        print(f"| Goal Yaw   90% (deg)        | {-self.goal_yaw_std * 2:^9.1f}~{self.goal_yaw_std * 2:^9.1f} |")
        print("-----------------------------------------------------")

    def plot_sample_distribution(self):
        """Debug: histogram the goal-direction / velocity / acceleration sample distributions."""
        import matplotlib.pyplot as plt
        # ===== 采样 =====
        N = 10000
        goals = np.array([self._get_random_goal() for _ in range(N)])
        states = np.array([self._get_random_state() for _ in range(N)])
        vels = np.stack([s[0] for s in states])
        accs = np.stack([s[1] for s in states])

        x, y, z = goals[:, 0], goals[:, 1], goals[:, 2]
        yaw = np.degrees(np.arctan2(y, x))  # 水平角 [-180, 180]
        pitch = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))  # 垂直角 [-90, 90]

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))

        # Goal方向角分布
        axs[0, 0].hist(yaw, bins=180)
        axs[0, 0].set_title("Goal Yaw Distribution")
        axs[0, 0].set_xlabel("Yaw (deg)")
        axs[0, 0].set_xlim([-60, 60])
        axs[0, 0].grid(True)

        axs[0, 1].hist(pitch, bins=90)
        axs[0, 1].set_title("Goal Pitch Distribution")
        axs[0, 1].set_xlabel("Pitch (deg)")
        axs[0, 1].set_xlim([-60, 60])
        axs[0, 1].grid(True)

        # Goal往图像投影分布(未考虑机体旋转)
        axs[0, 2].scatter(yaw, pitch, s=2, alpha=0.3)
        axs[0, 2].set_title("Goal Distribution in Image")
        axs[0, 2].set_xlabel("Yaw (deg)")
        axs[0, 2].set_ylabel("Pitch (deg)")
        axs[0, 2].set_xlim([-45, 45])
        axs[0, 2].set_ylim([-30, 30])
        axs[0, 2].grid(True)

        # Velocity分布
        for i, name in enumerate(['Vx', 'Vy', 'Vz']):
            axs[1, i].hist(vels[:, i], bins=100)
            axs[1, i].set_title(f"Velocity {name}")
            axs[1, i].grid(True)

        # Acceleration分布
        for i, name in enumerate(['Ax', 'Ay', 'Az']):
            axs[2, i].hist(accs[:, i], bins=100)
            axs[2, i].set_title(f"Acceleration {name}")
            axs[2, i].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # plot the random sample
    dataset = YOPODataset()
    dataset.plot_sample_distribution()
