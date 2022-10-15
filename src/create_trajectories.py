import gym
import argparse
import crafter
import numpy as np
import os
import random

from utils.wrappers import *
from crafter.env import Env

import matplotlib.pyplot as plt
from IPython import embed

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--traj-length', type=int, default = 500,
                    help='trajectory length')
parser.add_argument('--traj-num', type=int, default=500,
                    help='num of trajectories')
parser.add_argument('--save-path', type=str, default="src/representations/trajectories/",
                    help='save path')
parser.add_argument('--num-envs', type=int, default=10,
                    help='num of different environment seeds to collect trajectories on')
parser.add_argument('--seed', type=int, default=1,
                    help='seed of the environment')
args = parser.parse_args()

traj_len = args.traj_length
traj_num = args.traj_num

env = Env(seed = args.seed)  # Or CrafterNoReward-v1

possible_actions = [1,2,3,4]

for i in range(traj_num):
    steps = 0
    done = False
    obs = env.reset("random")
    
    trajectory = []
    trajectory_positions = []

    while steps < traj_len and not done:
        
        obs, reward, done, info = env.step(random.choice(possible_actions))
        
        player_pos = np.array(info["player_pos"])

        # img to gray scale, eventually the model uses gray images
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        obs = obs[None,:, :]
        
        # scale imgs to [0,1], the model uses this format
        obs = np.array(obs).astype(np.float32) / 255.0
        
        trajectory.append(obs)
        trajectory_positions.append(player_pos)

    trajectory = np.array(trajectory)
    trajectory_positions = np.array(trajectory_positions)
    
    trajectory_observations_path = args.save_path + "observations/" + "trajectory_observations_"  + str(i) + ".npy"
    trajectory_positions_path = args.save_path + "positions/" + "trajectory_positions_" + str(i) + ".npy"

    with open(trajectory_observations_path, 'wb') as tf:
        np.save(tf, trajectory)
    
    with open(trajectory_positions_path, 'wb') as to:
        np.save(to, trajectory_positions)




        
        


