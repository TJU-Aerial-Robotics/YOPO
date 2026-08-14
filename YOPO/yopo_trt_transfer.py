"""
    将yopo模型转换为Tensorrt
    prepare:
        1 pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
        2 git clone https://github.com/NVIDIA-AI-IOT/torch2trt
          cd torch2trt
          python setup.py install
    run (from the YOPO directory):
        python yopo_trt_transfer.py --trial=1 --epoch=50
"""

import os
import argparse
import time
import numpy as np
import torch
from torch2trt import torch2trt
from config.config import cfg
from policy.yopo_network import YopoNetwork


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--dir", type=str, default='yopo_trt.pth', help="output file name")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)

    print("Loading Network...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(weight, weights_only=True)
    policy = YopoNetwork()
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    # The inputs should be consistent with training
    depth = np.zeros(shape=[1, 1, cfg["image_height"], cfg["image_width"]], dtype=np.float32)
    obs = np.zeros(shape=[1, 9, cfg["vertical_num"], cfg["horizon_num"]], dtype=np.float32)
    depth_in = torch.from_numpy(depth).to(device)
    obs_in = torch.from_numpy(obs).to(device)

    print("TensorRT Transfer...")
    model_trt = torch2trt(policy, [depth_in, obs_in], fp16_mode=True)
    torch.save(model_trt.state_dict(), args.dir)


    print("Evaluation...")
    # Warm Up...
    traj_trt, score_trt, radius_trt = model_trt(depth_in, obs_in)
    traj, score, radius = policy(depth_in, obs_in)
    torch.cuda.synchronize()

    # PyTorch Latency
    torch_start = time.time()
    traj, score, radius = policy(depth_in, obs_in)
    torch.cuda.synchronize()
    torch_end = time.time()

    # TensorRT Latency
    trt_start = time.time()
    traj_trt, score_trt, radius_trt = model_trt(depth_in, obs_in)
    torch.cuda.synchronize()
    trt_end = time.time()

    # Transfer Error
    traj_error = torch.mean(torch.abs(traj - traj_trt))
    score_error = torch.mean(torch.abs(score - score_trt))
    radius_error = torch.mean(torch.abs(radius - radius_trt))

    print(f"Torch Latency: {1000 * (torch_end - torch_start):.3f} ms, "
          f"TensorRT Latency: {1000 * (trt_end - trt_start):.3f} ms, "
          f"Transfer Trajectory Error: {traj_error.item():.6f}, "
          f"Transfer Score Error: {score_error.item():.6f}, "
          f"Transfer Radius Error: {radius_error.item():.6f}")

