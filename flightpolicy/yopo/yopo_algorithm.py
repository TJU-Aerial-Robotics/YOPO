"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import time
from copy import deepcopy
import os
import random
import cv2
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps, get_schedule_fn, configure_logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import get_device

# -----------
from flightpolicy.yopo.yopo_policy import YopoPolicy
from flightpolicy.yopo.dataloader import YopoDataset
from torch.utils.data import DataLoader
from flightpolicy.yopo.primitive_utils import transform, rotate, transform_inv, rotate_inv
from flightpolicy.yopo.primitive_utils import LatticeParam, LatticePrimitive
from flightpolicy.yopo.buffers import ReplayBuffer
from ruamel.yaml import YAML


class YopoAlgorithm:
    def __init__(
            self,
            env=None,
            learning_rate=0.001,
            is_imitation=False,
            buffer_size=1_000_000,
            learning_starts=100,
            batch_size=256,
            unselect=0.0,
            loss_weight=[],
            train_freq=(1, "step"),
            change_env_freq=-1,
            gradient_steps=1,
            policy_kwargs=None,
            tensorboard_log=None,
            verbose=0,
            max_grad_norm=10,
    ):
        # env
        self.observation_dim = env.observation_dim
        self.action_dim = env.action_dim
        self.n_envs = env.num_envs
        self.env = env
        # training
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.unselect = unselect
        self.loss_weight = loss_weight
        self.device = get_device('auto')
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        # imitation learning
        self.is_imitation = is_imitation
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.change_env_freq = change_env_freq
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.freq_reset = False
        self.replay_buffer = None
        # logger
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.logger = configure_logger(self.verbose, self.tensorboard_log, "YOPO")
        # trajectory
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/traj_opt.yaml", 'r'))
        self.lattice_space = LatticeParam(cfg)
        self.lattice_primitive = LatticePrimitive(self.lattice_space)

        self._setup_model()

    def _setup_model(self):
        self.lr_schedule = get_schedule_fn(self.learning_rate)

        # buffer: pos, quat, vel, acc, depth
        if self.replay_buffer is None and self.is_imitation:
            self.replay_buffer = ReplayBuffer(
                self.buffer_size,
                self.observation_dim,
                (self.env.network_width, self.env.network_height),
                device=self.device,
                n_envs=self.n_envs,
            )

        print("Loading Network...")

        self.policy = YopoPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            lattice_space=self.lattice_space,
            lattice_primitive=self.lattice_primitive,
            lr_schedule=self.lr_schedule,
            train_env=self.env,
            device=self.device,
            **self.policy_kwargs
        )

        self.policy = self.policy.to(self.device)
        print("Network Loaded!")

        if self.is_imitation:
            self._convert_train_freq()

    def supervised_learning(self, epoch, log_interval):
        self.policy.set_training_mode(True)
        data_loader = DataLoader(YopoDataset(), batch_size=self.batch_size, shuffle=True, num_workers=0)

        n_updates = 0
        start_time = time.time()
        for epoch_ in range(epoch):
            cost_losses = []   # Performance (score) of prediction
            score_losses = []  # Accuracy of the predicted score
            for step, (depth, pos, quat, obs_b, map_id) in enumerate(data_loader):  # obs: body frame
                if depth.shape[0] != self.batch_size:   # batch size == num of env
                    continue
                n_updates = n_updates + 1
                depth = depth.to(self.device)
                obs_b = obs_b.numpy()

                goal_dir = obs_b[:, 6:9]
                goal_w = transform(quat.numpy(), pos.numpy(), 10 * goal_dir)  # Rwb * g_b + t_wb
                vel_w = rotate(quat.numpy(), obs_b[:, 0:3])
                acc_w = rotate(quat.numpy(), obs_b[:, 3:6])
                self.env.setState(pos.numpy(), vel_w, acc_w, quat.numpy())
                self.env.setGoal(goal_w)
                self.env.setMapID(map_id.numpy())

                obs_b[:, 0:6] = self.normalize_obs(obs_b[:, 0:6])
                obs_norm_input = self.prapare_input_observation(obs_b)
                obs_norm_input = obs_norm_input.to(self.device)
                endstate_score_predictions, cost_labels = self.policy.inference(depth, obs_norm_input)
                score_labels = cost_labels.clone().detach()
                cost_labels_record = th.mean(cost_labels)
                cost_labels_filtered = self.cost_filter(cost_labels)

                cost_loss = th.mean(cost_labels_filtered)
                score_loss = F.smooth_l1_loss(endstate_score_predictions[:, 9, :], score_labels)
                loss = self.loss_weight[0] * cost_loss + self.loss_weight[1] * score_loss
                cost_losses.append(self.loss_weight[0] * cost_labels_record.item())
                score_losses.append(self.loss_weight[1] * score_loss.item())

                # Optimize the policy
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                if log_interval is not None and n_updates % log_interval[0] == 0:
                    self.logger.record("time/epoch", epoch_, exclude="tensorboard")
                    self.logger.record("time/steps", n_updates, exclude="tensorboard")
                    self.logger.record("time/batch_fps", log_interval[0] / (time.time() - start_time),
                                       exclude="tensorboard")
                    self.logger.record("train/trajectory_cost", np.mean(cost_losses))
                    self.logger.record("train/score_loss", np.mean(score_losses))
                    self.logger.dump(step=n_updates)
                    cost_losses = []
                    score_losses = []
                    start_time = time.time()

                if log_interval is not None and n_updates % log_interval[1] == 0:
                    policy_path = self.logger.get_dir() + "/Policy"
                    os.makedirs(policy_path, exist_ok=True)
                    path = policy_path + "/epoch{}_iter{}.pth".format(epoch_, step)
                    th.save({"state_dict": self.policy.state_dict(), "data": self.policy.get_constructor_parameters()}, path)

    # 模仿学习: 已弃用(暂未删除以备后续使用)
    # 0、reset_state、get_depth、reset_goal
    # 1、执行若干步（env_num * 200）
    # 2、训练若干步（batch_size = env_num, 训200次=1eposide）
    # 3、reset_state、get_depth、reset_goal
    def imitation_learning(
            self,
            total_timesteps,
            callback=None,
            log_interval=4,
            eval_env=None,
            eval_freq=-1,
            n_eval_episodes=5,
            tb_log_name="YOPO",
            eval_log_path=None,
            reset_num_timesteps=True,
    ):

        # 0. 初始化第一次观测
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        self.pretrained = self.env.pretrained
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # 1. 数据收集
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            # 2. 训练模型
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    self.reset_state()

            iteration = int(self.num_timesteps / (self.train_freq.frequency * self.env.num_envs))

            # 3. 重置环境
            if self.change_env_freq > 0 and iteration % self.change_env_freq == 0:
                self.env.spawnTreesAndSavePointcloud()
                self._map_id = self._map_id + 1
                self.reset_state()

            # 4. 终端打印log
            if log_interval is not None and iteration % log_interval[0] == 0:
                self._dump_logs()

            if log_interval is not None and iteration % log_interval[1] == 0:
                policy_path = self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                path = policy_path + "/epoch0_iter{}.pth".format(iteration)
                th.save({"state_dict": self.policy.state_dict(), "data": self.policy.get_constructor_parameters()}, path)

        callback.on_training_end()

    def test_policy(self, num_rollouts: int = 10):
        max_ep_length = 400
        self.policy.set_training_mode(False)

        for n_roll in range(num_rollouts):
            obs, done, ep_len = self.env.reset(), False, 0
            costs = []
            # Randomly initialize the position and goal on the map.
            random_y_goal = 20 * random.uniform(-1, 1) + 20
            random_y = 20 * random.uniform(-1, 1) + 20
            goal_w = np.array([[20, random_y_goal, 2]])
            obs = np.array([[-20, random_y, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
            self.env.setGoal(goal_w)
            self.env.setState(np.array([[-20, random_y, 2]]), np.array([[0, 0, 0]]),
                              np.array([[0, 0, 0]]), np.array([[1, 0, 0, 0]]))
            self.env.render()

            while not (done or (ep_len >= max_ep_length)):
                depth = self.env.getDepthImage()
                depth_vis = cv2.resize(depth[0][0], (320, 180))
                cv2.imshow("depth", depth_vis)
                cv2.waitKey(10)
                depth = th.from_numpy(depth).to(self.device)

                # transform observation to body frame
                quat_bw = -obs[:, 9:13]  # inv of quat: [w, -x, -y, -z]
                quat_bw[:, 0] = -quat_bw[:, 0]
                goal_dir_w = (goal_w - obs[:, 0:3]) / np.linalg.norm(goal_w - obs[:, 0:3])
                goal_dir_b = rotate(quat_bw, goal_dir_w)
                vel_acc_norm_b = self.normalize_obs(obs[:, 3:9])
                obs_norm_b = np.hstack((vel_acc_norm_b, goal_dir_b))

                obs_norm_input = self.prapare_input_observation(obs_norm_b)
                obs_norm_input = obs_norm_input.to(self.device)

                endstate_pred, score_pred = self.policy.predict(depth, obs_norm_input)
                endstate_pred = endstate_pred.cpu().numpy()
                # obs: p_wb, v_b, a_b, q_wb; endstate_pred: pva in body frame
                obs, rew, done = self.env.step(endstate_pred)

                costs.append(rew)
                ep_len += 1
            print("round ", n_roll, ", total steps:", len(costs), ", avg cost:", sum(costs) / len(costs))

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
            Sample the replay buffer and do the updates
            (gradient descent and update target networks)
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule (TODO in supervised learning)
        self._update_learning_rate(self.policy.optimizer)

        cost_losses = []
        score_losses = []  # dy, dz, r, p, vx, vy, vz
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            depth = th.from_numpy(replay_data.depths).to(self.device)
            pos = replay_data.observations[:, 0:3]
            vel_acc_b = replay_data.observations[:, 3:9]
            quat_wb = replay_data.observations[:, 9:13]
            goal_w = replay_data.goals
            map_id = replay_data.map_id

            goal_dir_w = (goal_w - pos) / np.linalg.norm(goal_w - pos, axis=1)[:, np.newaxis]
            goal_dir_b = rotate_inv(quat_wb, goal_dir_w)
            vel_w = rotate(quat_wb, vel_acc_b[:, 0:3])
            acc_w = rotate(quat_wb, vel_acc_b[:, 3:6])
            self.env.setState(pos, vel_w, acc_w, quat_wb)
            self.env.setGoal(goal_w)
            self.env.setMapID(map_id)

            vel_acc_norm_b = self.normalize_obs(vel_acc_b)
            obs_norm_b = np.hstack((vel_acc_norm_b, goal_dir_b))
            obs_norm_input = self.prapare_input_observation(obs_norm_b)
            obs_norm_input = obs_norm_input.to(self.device)
            endstate_score_predictions, cost_labels = self.policy.inference(depth, obs_norm_input)
            score_labels = cost_labels.clone().detach()

            cost_labels_record = th.mean(cost_labels)
            cost_labels_filtered = self.cost_filter(cost_labels)

            cost_loss = th.mean(cost_labels_filtered)
            score_loss = F.smooth_l1_loss(endstate_score_predictions[:, 9, :], score_labels)
            loss = self.loss_weight[0] * cost_loss + self.loss_weight[1] * score_loss
            cost_losses.append(self.loss_weight[0] * cost_labels_record.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/trajectory_cost", np.mean(cost_losses))
        self.logger.record("train/score_loss", np.mean(score_losses))

    def collect_rollouts(
            self,
            env,
            callback,
            train_freq,
            replay_buffer,
            action_noise=None,
            log_interval=None,
    ) -> RolloutReturn:

        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        callback.on_rollout_start()
        continue_training = True

        """
        1、pred endstate
        2、get obs: self._last_obs = env.step(endstate)
        3、get depth: self._last_depth = env.getDepthImage()
        4、record to buffer and back to 1
        """
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

            # 1. pred endstate used latest policy or pre-trained policy
            sampled_endstate = self._sample_action(action_noise, env.num_envs)

            # 2. perform action
            new_obs, rewards, dones = env.step(sampled_endstate)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # 3. store the last obs, depth, and goal
            # self._update_info_buffer(infos, dones)
            self._store_transition(replay_buffer)
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # 4. update the obs, depth, goal, and reset the goal for the done-env
            self._last_obs = new_obs
            self._last_depth = env.getDepthImage()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1
                    # reset goal for the 'done' env
                    self._last_goal[idx] = self.get_random_goal(self._last_obs[idx])

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def prapare_input_observation(self, obs):
        """
            convert the observation from body frame to primitive frame,
            and then concatenate it with the depth features (to ensure the translational invariance)
        """
        obs_return = np.ones(
            (obs.shape[0], self.lattice_space.vertical_num, self.lattice_space.horizon_num, obs.shape[1]),
            dtype=np.float32)
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
        return th.from_numpy(obs_return)

    def unnormalize_obs(self, vel_acc_norm):
        vel = vel_acc_norm[:, 0:3] * self.lattice_space.vel_max
        acc = vel_acc_norm[:, 3:6] * self.lattice_space.acc_max
        return np.hstack((vel, acc))

    def normalize_obs(self, vel_acc):
        vel_norm = vel_acc[:, 0:3] / self.lattice_space.vel_max
        acc_norm = vel_acc[:, 3:6] / self.lattice_space.acc_max
        return np.hstack((vel_norm, acc_norm))

    def cost_filter(self, costs_):
        # costs_ = costs.clone()  # NOTE: numpy.ndarray is reference invocation!
        if self.unselect <= 0 or self.unselect >= 1:
            return costs_
        # filter the negative samples
        rows, cols = costs_.size()
        unselect = int(cols * self.unselect)
        for i in range(rows):
            row = costs_[i]
            _, indices = th.topk(row, unselect)
            costs_[i][indices] = 0.0
        return costs_

    def _setup_learn(
            self,
            total_timesteps,
            eval_env=None,
            callback=None,
            eval_freq=10000,
            n_eval_episodes=5,
            log_path=None,
            reset_num_timesteps=True,
            tb_log_name="run",
    ):
        # ----------------- Init the First Observation  -----------------
        # super()._setup_learn() 中： self._last_obs = self.env.reset()
        total_timesteps_, callback_ = super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        self._last_depth = self.env.getDepthImage()
        self._last_goal = np.zeros([self.env.num_envs, 3], dtype=np.float32)
        for i in range(0, self.env.num_envs):
            self._last_goal[i] = self.get_random_goal(self._last_obs[i])
        self._map_id = np.zeros((self.env.num_envs, 1), dtype=np.float32)

        return total_timesteps_, callback_


    def _sample_action(self) -> np.ndarray:
        """
        use pretrained model or current model to sample the actions (endstate)
        self._last_obs: last state obs [p, v, a, q]
        self._last_depth: last depth image
        """
        obs = self._last_obs.copy()
        goal_w = self._last_goal.copy()
        depth = th.from_numpy(self._last_depth).to(self.device)
        # wxyz 四元数的逆[w, -x, -y, -z]
        quat_bw = -obs[:, 9:13]
        quat_bw[:, 0] = -quat_bw[:, 0]
        vel_acc_norm_b = self.normalize_obs(obs[:, 3:9])
        goal_dir_w = (goal_w - obs[:, 0:3]) / np.linalg.norm(goal_w - obs[:, 0:3], axis=1)[:, np.newaxis]
        goal_dir_b = rotate(quat_bw, goal_dir_w)
        obs_norm_b = np.hstack((vel_acc_norm_b, goal_dir_b))

        obs_norm_input = self.prapare_input_observation(obs_norm_b)
        obs_norm_input = obs_norm_input.to(self.device)

        endstate_pred, score_pred = self.policy.predict(depth, obs_norm_input)
        endstate_pred = endstate_pred.cpu().numpy()
        return endstate_pred

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/fps", fps, exclude="tensorboard")
        self.logger.record("time/minute_elapsed", int(time_elapsed / 60), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.record("train/map_id", self._map_id[0][0], exclude="tensorboard")

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _store_transition(self, replay_buffer):

        # Avoid modification by reference
        obs = deepcopy(self._last_obs)
        goal = deepcopy(self._last_goal)
        depth = deepcopy(self._last_depth)
        map_id = deepcopy(self._map_id)

        replay_buffer.add(
            obs,
            goal,
            depth,
            map_id
        )

    def get_random_goal(self, uav_state=None):
        world = self.env.world_box
        # 1. Use random goal in map
        if uav_state is None:
            world_center = np.array([world[3] + world[0], world[4] + world[1], world[5] + world[2]]) / 2
            world_scale = np.array([world[3] - world[0], world[4] - world[1], 1.0])
            # The goal can be out of the world, if strictly in world: np.random.uniform(-0.5, 0.5, 3)
            random_numbers = np.random.uniform(-1, 1, 3)
            random_goal = random_numbers * world_scale + world_center
        # 2. Use goal in front of the UAV (for better imitation learning)
        else:
            q_wb = uav_state[9:]
            p_wb = uav_state[0:3]
            goal = np.random.randn(3) + np.array([2, 0, 0])
            goal_dir = goal / np.linalg.norm(goal)
            random_goal_b = 50 * goal_dir
            random_goal_w = transform(q_wb, p_wb, random_goal_b)
            random_goal_w[2] = np.random.uniform(-1, 1) * 1 + (world[5] + world[2]) / 2
            random_goal = random_goal_w

        return random_goal

    def reset_state(self):
        """
            Reset the state and map_id after every train step, because the state and map_id are manually set in training,
            which will affect the cost, controller, image render, and other parts for next rollout
        """
        self.env.setMapID(-np.ones((self.env.num_envs, 1)))
        self._last_obs = self.env.reset()
        self._last_depth = self.env.getDepthImage()
        for i in range(0, self.env.num_envs):
            self._last_goal[i] = self.get_random_goal(self._last_obs[i])

    def _convert_train_freq(self) -> None:
        """
            Convert `train_freq` parameter (int or tuple)
            to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)
