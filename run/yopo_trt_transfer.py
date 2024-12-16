"""
    将yopo模型转换为Tensorrt
    prepare:
        1 pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
        2 git clone https://github.com/NVIDIA-AI-IOT/torch2trt
          cd torch2trt
          python setup.py install
"""

import argparse
import os
import numpy as np
import torch
import time
from torch2trt import torch2trt
from flightgym import QuadrotorEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from flightpolicy.envs import vec_env_wrapper as wrapper
from flightpolicy.yopo.yopo_algorithm import YopoAlgorithm


def prapare_input_observation(obs, lattice_space, lattice_primitive):
    obs_return = np.ones(
        (obs.shape[0], lattice_space.vertical_num, lattice_space.horizon_num, obs.shape[1]),
        dtype=np.float32)
    id = 0
    v_b = obs[:, 0:3]
    a_b = obs[:, 3:6]
    g_b = obs[:, 6:9]
    for i in range(lattice_space.vertical_num - 1, -1, -1):
        for j in range(lattice_space.horizon_num - 1, -1, -1):
            Rbp = lattice_primitive.getRotation(id)
            v_p = np.dot(Rbp.T, v_b.T).T
            a_p = np.dot(Rbp.T, a_b.T).T
            g_p = np.dot(Rbp.T, g_b.T).T
            obs_return[:, i, j, 0:3] = v_p
            obs_return[:, i, j, 3:6] = a_p
            obs_return[:, i, j, 6:9] = g_p
            id = id + 1
    obs_return = np.transpose(obs_return, [0, 3, 1, 2])
    return obs_return


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=0, help="epoch number")
    parser.add_argument("--iter", type=int, default=0, help="iter number")
    parser.add_argument("--filename", type=str, default='yopo_trt.pth', help="output file name")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    # load configurations
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml", 'r'))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["supervised"] = False
    cfg["env"]["imitation"] = False
    cfg["env"]["render"] = False

    # create environment
    train_env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)
    model = YopoAlgorithm(env=train_env,
                          policy_kwargs=dict(
                              activation_fn=torch.nn.ReLU,
                              net_arch=[256, 256],
                              hidden_state=64
                          ))

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    weight = rsg_root + "/saved/YOPO_{}/Policy/epoch{}_iter{}.pth".format(args.trial, args.epoch, args.iter)
    device = torch.device("cuda")
    saved_variables = torch.load(weight, map_location=device)
    model.policy.load_state_dict(saved_variables["state_dict"], strict=False)
    model.policy.set_training_mode(False)

    lattice_space = saved_variables["data"]["lattice_space"]
    lattice_primitive = saved_variables["data"]["lattice_primitive"]

    # The inputs should be consistent with training
    print("TensorRT Transfer...")
    depth = np.zeros(shape=[1, 1, 96, 160], dtype=np.float32)
    obs = np.zeros(shape=[1, 9], dtype=np.float32)
    obs_input = prapare_input_observation(obs, lattice_space, lattice_primitive)
    depth_in = torch.from_numpy(depth).cuda()
    obs_in = torch.from_numpy(obs_input).cuda()
    model_trt = torch2trt(model.policy, [depth_in, obs_in])
    torch.save(model_trt.state_dict(), args.filename)
    print("TensorRT Transfer Finish!")

    # from torch2trt import TRTModule
    # model_trt = TRTModule()
    # model_trt.load_state_dict(torch.load('yopo_trt.pth'))

    print("Evaluation...")
    # warm up...
    y_trt = model_trt(depth_in, obs_in)
    y = model.policy(depth_in, obs_in)

    torch_start = time.time()
    y = model.policy(depth_in, obs_in)
    torch_end = time.time()
    y_trt = model_trt(depth_in, obs_in)
    trt_end = time.time()

    error = torch.mean(torch.abs(y - y_trt))
    print("Torch Latency: ", 1000 * (torch_end - torch_start),
          "ms, TensorRT Latency: ", 1000 * (trt_end - torch_end),
          "ms, Transfer Error: ", error.item())

