import gym
import argparse
import crafter
import numpy as np
import os

from crafter.env import Env
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1,
                    help='seed of the environment')
args = parser.parse_args()

env = Env(seed = args.seed)  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=False,
  save_video=False,
  save_episode=False,
)

#obs = env.reset()
#world_img = env.render_world()
#print(world_img)
#plt.imsave("lol.png", world_img / 255.)

observations_path = 'src/trajectories/observations/'
for i in range(len(os.listdir(observations_path))):
    obs = np.load(observations_path + "trajectory_observations_" + str(i) + ".npy")
    step = 0
    for o in obs:
        plt.imshow(o.squeeze())
        step += 1
    break
plt.savefig('lel.png')


"""
positions_path = 'src/trajectories/positions/'
for i in range(len(os.listdir(positions_path))):
    position = np.load(positions_path + "trajectory_positions_" + str(i) + ".npy")
    step = 0
    for pos in position:
        plt.plot(step, pos)
        step += 1
    break

plt.savefig('lel.png')
"""