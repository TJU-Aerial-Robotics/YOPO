import argparse
import os
import random
import numpy as np
import torch
from flightgym import QuadrotorEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from flightpolicy.envs import vec_env_wrapper as wrapper
from flightpolicy.yopo.yopo_algorithm import YopoAlgorithm


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--train", type=int, default=1, help="train or evaluate the policy?")
    parser.add_argument("--render", type=int, default=0, help="render with Unity?")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=0, help="epoch number")
    parser.add_argument("--iter", type=int, default=0, help="iter number")
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
    parser.add_argument("--supervised", type=int, default=1, help="supervised learning?")
    parser.add_argument("--imitation", type=int, default=0, help="imitation learning?")
    return parser


def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml", 'r'))
    cfg["env"]["supervised"] = bool(args.supervised)
    cfg["env"]["imitation"] = bool(args.imitation)
    if not args.train:
        cfg["env"]["num_envs"] = 1
    cfg["env"]["render"] = bool(args.render)
    if args.render:
        cfg["env"]["ply_path"] = "/flightrender/RPG_Flightmare/pointcloud_data/"  # change the paths during test or imitation
        if not os.path.exists(os.environ["FLIGHTMARE_PATH"] + cfg["env"]["ply_path"]):
            os.mkdir(os.environ["FLIGHTMARE_PATH"] + cfg["env"]["ply_path"])
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare/flightmare.x86_64 &")

    # create training environment
    train_env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)

    model = YopoAlgorithm(
        tensorboard_log=log_dir,
        env=train_env,
        is_imitation=args.imitation,
        learning_starts=10000,              # How many samples are collected before starting imitation learning
        train_freq=200,                     # How many steps of data to collect from each environment per round
        gradient_steps=200,                 # How many steps to train per round
        change_env_freq=20,                 # How many rounds of "collect-train" to reset the tree (-1: not reset)
        learning_rate=1.5e-4,               # Learning rate
        batch_size=cfg["env"]["num_envs"],  # Equal to the number of environment, as gradients are from environments
        buffer_size=100000,                 # Buffer size
        loss_weight=[1.0, 10.0],            # Weights for the costs of endstate and score
        unselect=0,                         # Proportion of trajectories not optimized in each sample
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[256, 256],
            hidden_state=64
        ),
        verbose=1,
    )

    if args.render:
        train_env.connectUnity()
        spacing = cfg["unity"]["avg_tree_spacing"]
        train_env.spawnTreesAndSavePointcloud(0, spacing)
        train_env.setMapID(-np.ones((train_env.num_envs, 1)))
        train_env.reset(random=True)

    if args.train:
        if args.pretrained:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weight = rsg_root + "/saved/YOPO_{}/Policy/epoch{}_iter{}.pth".format(args.trial, args.epoch, args.iter)
            saved_variables = torch.load(weight, map_location=device)
            model.policy.load_state_dict(saved_variables["state_dict"], strict=False)
            print("use pretrained model ", weight)

        if args.supervised:
            model.supervised_learning(epoch=int(50), log_interval=(100, 50000))  # How many batches to print and save

        elif args.imitation:
            model.imitation_learning(total_timesteps=int(1 * 1e6), log_interval=(1, 40))

    else:
        weight = rsg_root + "/saved/YOPO_{}/Policy/epoch{}_iter{}.pth".format(args.trial, args.epoch, args.iter)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_variables = torch.load(weight, map_location=device)
        model.policy.load_state_dict(saved_variables["state_dict"], strict=False)
        model.test_policy(num_rollouts=20)

    print("Run YOPO Finish!")

if __name__ == "__main__":
    main()
