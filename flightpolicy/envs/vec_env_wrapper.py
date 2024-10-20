import os
import gym
import torch
import numpy as np
import cv2
from ruamel.yaml import YAML
from typing import Any, List, Type
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices


class FlightEnvVec(VecEnv):

    def __init__(self, impl):
        self.wrapper = impl
        # params
        self.action_dim = self.wrapper.getActDim()
        self.observation_dim = self.wrapper.getObsDim()
        self.reward_dim = self.wrapper.getRewDim()
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/traj_opt.yaml", 'r'))
        scale = 32  # The downsampling factor of backbone
        self.network_height = scale * cfg["vertical_num"]
        self.network_width = scale * cfg["horizon_num"]
        self.world_box = np.zeros([6], dtype=np.float32)
        self.wrapper.getWorldBox(self.world_box)  # xyz_min, xyz_max
        self.reward_names = self.wrapper.getRewardNames()
        self.pretrained = False

        # observations
        self._traj_cost = np.zeros([self.num_envs, 1], dtype=np.float32)  # cost of current pred
        self._traj_grad = np.zeros([self.num_envs, 9], dtype=np.float32)  # gard of current pred x_pva y_pav z_pva
        self._observation = np.zeros([self.num_envs, self.observation_dim], dtype=np.float32)
        self._rgb_img_obs = np.zeros([self.num_envs, self.img_width * self.img_height * 3], dtype=np.uint8)
        self._gray_img_obs = np.zeros([self.num_envs, self.img_width * self.img_height], dtype=np.uint8)
        self._depth_img_obs = np.zeros([self.num_envs, self.img_width * self.img_height], dtype=np.float32)
        self._reward = np.zeros([self.num_envs, self.reward_dim], dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # observation: [p_wb, v_b, a_b, q_wb] (in Body Frame); action: dp_pred; reward: cost
    def step(self, action):
        if action.ndim <= 1:
            action = action.reshape((self.num_envs, -1))
        if action.dtype == np.dtype('int'):
            action = action.astype(np.float32)
        self.wrapper.step(
            action,
            self._observation,
            self._reward,
            self._done,
        )

        return (
            self._observation.copy(),
            self._reward.copy(),
            self._done.copy(),
        )

    # observation: [p_wb, v_b, a_b, q_wb] (in Body Frame)
    def reset(self, random=True):
        self._reward = np.zeros([self.num_envs, self.reward_dim], dtype=np.float32)
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    # (in World Frame) goal_w
    def setGoal(self, goal):
        if goal.ndim <= 1:
            goal = goal.reshape((self.num_envs, -1))
        self.wrapper.setGoal(goal)

    # (in World Frame) pos_wb, vel_w, acc_w, quat_wb
    def setState(self, pos, vel, acc, quad):
        if pos.ndim <= 1:
            pos = pos.reshape((self.num_envs, -1))
            quad = quad.reshape((self.num_envs, -1))  # wxyz
            vel = vel.reshape((self.num_envs, -1))
            acc = acc.reshape((self.num_envs, -1))
        state = np.hstack((pos, vel, acc, quad))
        self.wrapper.setState(state)

    # map_id: The ID of the map used in the current training;
    # during data collection or DAgger, map_id=-1 indicates that the latest map is used.
    def setMapID(self, map_id):
        if map_id.ndim <= 1:
            map_id = map_id.reshape((self.num_envs, -1))
        self.wrapper.setMapID(map_id)

    def getObs(self):
        self.wrapper.getObs(self._observation)
        return self._observation.copy()

    # pred_dp: x_pva, y_pva, z_pva (in Body Frame);  _traj_grad: x_pva, y_pva, z_pva (in Body Frame)
    def getCostAndGradient(self, pred_dp_in, traj_id):
        """
        Args:
            pred_dp_in: the prediction of dp (x_pva, y_pva, z_pva)
            traj_id: the id of the trajectory in lattice

        Returns: the cost and gradient of the prediction dp (x_pva, y_pva, z_pva)

        """
        if not isinstance(pred_dp_in, np.ndarray):
            pred_dp = pred_dp_in.detach().cpu().numpy()
        else:
            pred_dp = pred_dp_in

        if pred_dp.ndim <= 1:
            pred_dp = pred_dp.reshape((self.num_envs, -1))
        if traj_id.ndim <= 1:
            traj_id = traj_id.reshape((self.num_envs, -1))
        self.wrapper.getCostAndGradient(pred_dp, traj_id, self._traj_cost, self._traj_grad)
        return self._traj_cost.copy(), self._traj_grad.copy()

    def getRGBImage(self, rgb=False):
        if rgb:
            self.wrapper.getRGBImage(self._rgb_img_obs, True)
            return self._rgb_img_obs.copy()
        else:
            self.wrapper.getRGBImage(self._gray_img_obs, False)
            gray_img = self._gray_img_obs
            gray_img = np.reshape(gray_img, (gray_img.shape[0], self.img_height, self.img_width))
            return gray_img.copy()

    def getDepthImage(self, resize=True):
        self.wrapper.getDepthImage(self._depth_img_obs)
        # normalize the depth values from 0-20m to 0-1
        depth = 1000 * self._depth_img_obs
        depth = np.minimum(depth, 20)
        depth = depth / 20.0
        depth[np.isnan(depth)] = 1.0
        depth = np.reshape(depth, (depth.shape[0], self.img_height, self.img_width))
        if resize:
            depth_ = np.zeros((depth.shape[0], self.network_height, self.network_width), dtype=np.float32())
            for i in range(depth.shape[0]):
                depth_[i] = cv2.resize(depth[i], (self.network_width, self.network_height))
            depth = np.expand_dims(depth_, axis=1)
        else:
            depth = np.expand_dims(depth, axis=1)
        return depth.copy()

    def getStereoImage(self):
        # [n_envs, HxW]
        self.wrapper.getStereoImage(self._depth_img_obs)
        depth = self._depth_img_obs
        depth = np.minimum(depth, 20) / 20

        depth_ = np.zeros((depth.shape[0], self.network_height, self.network_width), dtype=np.float32())
        for i in range(depth.shape[0]):
            nan_mask = np.isnan(depth[i])
            interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
            interpolated_image = interpolated_image.astype(np.float32) / 255.0
            interpolated_image = np.reshape(interpolated_image, (self.img_height, self.img_width))
            depth_[i] = cv2.resize(interpolated_image, (self.network_width, self.network_height))
        depth_ = np.expand_dims(depth_, axis=1)

        return depth_.copy()

    def getQuadState(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate

    def spawnTrees(self):
        self.wrapper.spawnTrees()  # avg_tree_spacing is defined in .cfg

    def savePointcloud(self, ply_idx):
        self.wrapper.savePointcloud(ply_idx)

    def spawnTreesAndSavePointcloud(self, ply_idx=-1, spacing=-1):
        self.wrapper.spawnTreesAndSavePointcloud(ply_idx, spacing)

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def render(self):
        return self.wrapper.render()

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.wrapper.connectUnity()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices: VecEnvIndices = None,
            **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
            self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    def step_async(self):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        raise RuntimeError("This method is not implemented")
