"""
    将yopo模型转换为Tensorrt
    prepare:
        0. make sure you install already install TensorRT
        1. pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
        2. git clone https://github.com/NVIDIA-AI-IOT/torch2trt
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


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=0, help="epoch number")
    parser.add_argument("--iter", type=int, default=0, help="iter number")
    parser.add_argument("--fp16_mode", type=int, default=1, help="fp16 or fp32")
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
    torch.set_grad_enabled(False)

    lattice_space = saved_variables["data"]["lattice_space"]
    lattice_primitive = saved_variables["data"]["lattice_primitive"]

    # The inputs should be consistent with training
    print("TensorRT Transfer...")
    depth = np.zeros(shape=[1, 1, 96, 160], dtype=np.float32)
    obs = np.zeros(shape=[1, 9, lattice_space.vertical_num, lattice_space.horizon_num], dtype=np.float32)
    depth_in = torch.from_numpy(depth).cuda()
    obs_in = torch.from_numpy(obs).cuda()
    model_trt = torch2trt(model.policy, [depth_in, obs_in], fp16_mode=args.fp16_mode)
    torch.save(model_trt.state_dict(), args.filename)
    print("TensorRT Transfer Finish!")

    # from torch2trt import TRTModule
    # model_trt = TRTModule()
    # model_trt.load_state_dict(torch.load('yopo_trt.pth'))

    print("Evaluation...")
    # Warm Up...
    y_trt = model_trt(depth_in, obs_in)
    y = model.policy(depth_in, obs_in)
    torch.cuda.synchronize()

    # PyTorch Latency
    torch_start = time.time()
    y = model.policy(depth_in, obs_in)
    torch.cuda.synchronize()
    torch_end = time.time()

    # TensorRT Latency
    trt_start = time.time()
    y_trt = model_trt(depth_in, obs_in)
    torch.cuda.synchronize()
    trt_end = time.time()

    # Transfer Error
    error = torch.mean(torch.abs(y - y_trt))

    print(f"Torch Latency: {1000 * (torch_end - torch_start):.3f} ms, "
          f"TensorRT Latency: {1000 * (trt_end - trt_start):.3f} ms, "
          f"Transfer Error: {error.item():.8f}")

