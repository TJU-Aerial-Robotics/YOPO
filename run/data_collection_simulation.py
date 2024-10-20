#!/usr/bin/env python3

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from flightgym import QuadrotorEnv_v1
from flightpolicy.envs import vec_env_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_each_env", type=int, default=10000, help="num of images to save in each env")
    parser.add_argument("--num_env", type=int, default=10, help="num of env to change")
    return parser


def main():
    args = parser().parse_args()

    configure_random_seed(args.seed)

    # load configurations
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml", 'r'))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = True
    cfg["env"]["supervised"] = False
    cfg["env"]["imitation"] = False

    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare/flightmare.x86_64 &")
    env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    env = wrapper.FlightEnvVec(env)
    env.connectUnity()

    iteration = args.num_each_env
    epoch = args.num_env

    home_dir = os.environ["FLIGHTMARE_PATH"] + cfg["env"]["dataset_path"]
    if not os.path.exists(home_dir):
        os.mkdir(home_dir)

    for epoch_i in range(epoch):
        spacing = cfg["unity"]["avg_tree_spacing"]
        env.spawnTreesAndSavePointcloud(epoch_i, spacing)
        env.setMapID(np.array([-1]))
        env.reset(random=True)

        positions = np.zeros([iteration, 3], dtype=np.float32)
        quaternions = np.zeros([iteration, 4], dtype=np.float32)

        save_dir = os.environ["FLIGHTMARE_PATH"] + cfg["env"]["dataset_path"] + str(epoch_i) + "/"
        label_path = save_dir + "/label.npz"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for frame_id in tqdm(range(iteration)):
            image_path = save_dir + "/img_" + str(frame_id) + ".tif"
            observation = env.reset()
            positions[frame_id, :] = observation[0, 0:3]
            quaternions[frame_id, :] = observation[0, 9:]
            depth = env.getDepthImage(resize=False)
            cv2.imwrite(image_path, depth[0][0])

        np.savez(
            label_path,
            positions=positions,
            quaternions=quaternions,
        )

    env.disconnectUnity()


if __name__ == "__main__":
    main()
