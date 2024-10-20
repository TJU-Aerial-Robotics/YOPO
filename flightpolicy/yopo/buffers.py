"""
    The code is from stable_baseline3.
"""
from abc import ABC, abstractmethod
from gym import spaces
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple
from stable_baselines3.common.vec_env import VecNormalize
import torch as th
import numpy as np
import warnings
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_dim: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_dim: int,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim

        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
            self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    goals: th.Tensor
    depths: th.Tensor
    map_id: th.Tensor


class ReplayBuffer(BaseBuffer):
    """
    self.observations
    self.goals
    self.depths
    self.map_ids
    """

    def __init__(
            self,
            buffer_size: int,
            observation_dim: spaces.Space,
            image_WxH: tuple,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_dim, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, observation_dim), dtype=np.float32)
        self.goals = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)
        self.depths = np.zeros((self.buffer_size, self.n_envs, 1, image_WxH[1], image_WxH[0]), dtype=np.float32)
        self.map_ids = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.int16)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.goals.nbytes + self.depths.nbytes + self.map_ids.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self,
            obs: np.ndarray,
            goal: np.ndarray,
            depth: np.ndarray,
            map_id: int) -> None:

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.goals[self.pos] = np.array(goal).copy()
        self.depths[self.pos] = np.array(depth).copy()
        self.map_ids[self.pos] = np.array(map_id).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        data = (
            self.observations[batch_inds, env_indices, :],
            self.goals[batch_inds, env_indices, :],
            self.depths[batch_inds, env_indices, :],
            self.map_ids[batch_inds, env_indices, :],
        )
        return ReplayBufferSamples(*data)
