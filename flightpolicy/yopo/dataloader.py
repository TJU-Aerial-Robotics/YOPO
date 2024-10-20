import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ruamel.yaml import YAML
import time
from scipy.spatial.transform import Rotation as R


class YopoDataset(Dataset):
    def __init__(self):
        super(YopoDataset, self).__init__()
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/traj_opt.yaml", 'r'))
        scale = 32  # 神经网络下采样倍数
        self.height = scale * cfg["vertical_num"]
        self.width = scale * cfg["horizon_num"]
        multiple_ = 0.5 * cfg["vel_max"]
        # The x-direction follows a log-normal distribution,
        # while the yz-direction follows a normal distribution with a mean of 0.
        self.v_max = cfg["vel_max"]
        v_des = multiple_ * cfg["vx_mean_unit"]
        self.vx_lognorm_mean = np.log(self.v_max - v_des)
        self.vx_logmorm_sigma = np.log(np.sqrt(v_des))
        self.v_mean = multiple_ * np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_var = multiple_ * multiple_ * np.array([cfg["vx_var_unit"], cfg["vy_var_unit"], cfg["vz_var_unit"]])
        self.a_mean = multiple_ * multiple_ * np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_var = multiple_ * multiple_ * multiple_ * multiple_ * np.array([cfg["ax_var_unit"], cfg["ay_var_unit"], cfg["az_var_unit"]])

        print("Loading dataset, it may take a while...")
        data_cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml", 'r'))
        data_dir = os.environ["FLIGHTMARE_PATH"] + data_cfg["env"]["dataset_path"]

        self.img_list = []
        self.map_idx = []
        self.positions = np.empty((0, 3))
        self.quaternions = np.empty((0, 4))
        subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        subfolders.sort(key=lambda x: os.path.basename(x).lower())
        for i in range(len(subfolders)):
            img_dir = subfolders[i]
            file_names = [filename
                          for filename in os.listdir(img_dir)
                          if os.path.splitext(filename)[1] == '.tif']
            file_names.sort(key=lambda x: int(x.split('.')[0].split("_")[1]))  # sort by filename
            images = [cv2.imread(img_dir + "/" + filename, -1).astype(np.float32) for filename in file_names]
            self.img_list.extend(images)
            self.map_idx.extend([i] * len(images))

            label_path = img_dir + "/label.npz"
            labels = np.load(label_path)
            self.positions = np.vstack((self.positions, labels["positions"]))
            self.quaternions = np.vstack((self.quaternions, labels["quaternions"]))

        print("Dataset loaded!")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        if self.img_list[item].shape[-2] != self.height or self.img_list[item].shape[-1] != self.width:
            self.img_list[item] = cv2.resize(self.img_list[item], (self.width, self.height))  # OpenCV and NumPy is Dif

        if len(self.img_list[item].shape) == 2:
            self.img_list[item] = np.expand_dims(self.img_list[item], axis=0)

        vel, acc = self._get_random_state()

        # generate random goal in front of the quadrotor.
        q_wxyz = self.quaternions[item, :]  # q: wxyz
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler_angles = R_WB.as_euler('ZYX', degrees=False)  # [yaw(z) pitch(y) roll(x)]
        R_wB = R.from_euler('ZYX', [0, euler_angles[1], euler_angles[2]], degrees=False)
        goal_w = np.random.randn(3) + np.array([2, 0, 0])
        goal_b = R_wB.inv().apply(goal_w)

        goal_dist = np.linalg.norm(goal_b)
        goal_dir = goal_b / goal_dist
        random_obs = np.hstack((vel, acc, goal_dir))

        return (self.img_list[item], self.positions[item, :], self.quaternions[item, :], random_obs,
                self.map_idx[item])  # in body frame, vel_acc no-normalization

    def _get_random_state(self):
        vel = self.v_mean + np.sqrt(self.v_var) * np.random.randn(3)
        acc = self.a_mean + np.sqrt(self.a_var) * np.random.randn(3)

        right_skewed_vx = -1
        while right_skewed_vx < 0:
            right_skewed_vx = np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
            right_skewed_vx = -right_skewed_vx + self.v_max + 0.2  # +0.2 to ensure v_max can be sampled
        vel[0] = right_skewed_vx
        # distribution of vx is visualized in docs/distribution_of_sampled_velocity.png (v_max=6)
        return vel, acc


if __name__ == '__main__':
    data_loader = DataLoader(YopoDataset(), batch_size=32, shuffle=True, num_workers=4)

    start = time.time()
    for epoch in range(1):
        last = time.time()
        for i, (depth, pos, quat, obs, id) in enumerate(data_loader):
            pass
    end = time.time()

    print("总耗时：", end - start)
