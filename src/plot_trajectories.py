import gym
import argparse
import crafter
import numpy as np
import os

from crafter.env import Env
import matplotlib.pyplot as plt

from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1,
                    help='seed of the environment')
parser.add_argument('--envs-num', type=int, default=1,
                    help='num of different environments')
args = parser.parse_args()

positions_path = 'src/representations/trajectories/positions/'
envs_num = args.envs_num

for env_ in range(envs_num):
  env = Env(seed = args.seed + env_, view = (64,64))

  env_path = "env_" + str(args.seed + env_)
  env_pos_path = positions_path + env_path
  
  plt.figure(env_)

  for i in range(len(os.listdir(env_pos_path))):
      position = np.load(env_pos_path + "/trajectory_positions_" + str(i) + ".npy")
      step = 0
      plt.plot(*zip(*position * (64,64)), color="black", linewidth=1)

  obs = env.reset()
  world_img = env.render_world()
  plt.imshow(world_img.transpose(1,0,2))

  plt.savefig("world_seed_" + str(env_path) + "_with_trajectories.png")