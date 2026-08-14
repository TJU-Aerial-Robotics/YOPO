import os
from ruamel.yaml import YAML


# Global Configuration Management
class Config:
    def __init__(self):
        """Load traj_opt.yaml and derive dependent params (goal_length, sgm_time, traj_num, image size)."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._data = YAML().load(open(os.path.join(base_dir, "traj_opt.yaml"), 'r'))
        self._data["train"] = True
        self._data["piece_num"] = 2
        self._data["goal_length"] = self._data["tail_radio_max"]
        self._data["sgm_time"] = self._data["tail_radio_max"] / self._data["vel_max_train"]
        self._data["traj_num"] = self._data['horizon_num'] * self._data['vertical_num']
        self._data["piece_duration"] = self._data["sgm_time"] / self._data["piece_num"]

        ds = self._data["downsample_factor"]
        self._data["image_height"] = self._data["vertical_num"] * ds
        self._data["image_width"] = self._data["horizon_num"] * ds

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


cfg = Config()
