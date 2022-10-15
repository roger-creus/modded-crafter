import gym
import argparse
import crafter
import numpy as np
import os
from crafter.env import Env
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--traj-path', type=str, default='src/representations/trajectories/positions/',
                    help='path to the positions trajectories')
parser.add_argument('--seed', type=int, default=1,
                    help='seed of the environment')
args = parser.parse_args()

env = Env(seed = args.seed, view = (64,64))
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=False,
  save_video=False,
  save_episode=False,
)

positions_path = args.traj_path

for i in range(len(os.listdir(positions_path))):
    position = np.load(positions_path + "trajectory_positions_" + str(i) + ".npy")
    step = 0
    plt.plot(*zip(*position * (64,64)), color="black", linewidth=1)

obs = env.reset()
world_img = env.render_world()
plt.imshow(world_img.transpose(1,0,2))

plt.savefig("world_seed_" + str(args.seed) + "_with_trajectories.png")