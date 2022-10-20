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

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.wrappers import *
from utils.utils import *
from utils.networks import *

parser = argparse.ArgumentParser()
parser.add_argument('--traj-length', type=int, default = 500,
                    help='trajectory length')
parser.add_argument('--traj-num', type=int, default=500,
                    help='num of trajectories')
parser.add_argument('--envs-num', type=int, default=1,
                    help='num of different environments')            
parser.add_argument('--save-path', type=str, default="src/representations/trajectories/",
                    help='save path')
parser.add_argument('--seed', type=int, default=1,
                    help='seed of the environment')

args = parser.parse_args()

traj_len = args.traj_length
traj_num = args.traj_num
envs_num = args.envs_num



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent().to(device)
agent.load_state_dict(torch.load("/home/mila/r/roger.creus-castanyer/modded-crafter/src/checkpoints/ppo-14254080.pt"))


for env_ in range(envs_num):
    
    env = Env(seed = args.seed + env_)  # Or CrafterNoReward-v1

    envs_path = "env_" + str(args.seed + env_) + "/"

    os.makedirs(args.save_path + "observations/")
    os.makedirs(args.save_path + "positions/")

    envs_obs_path = args.save_path + "observations/" + envs_path
    envs_pos_path = args.save_path + "positions/" + envs_path
    
    if not os.path.exists(envs_obs_path):
        os.makedirs(envs_obs_path)
    
    if not os.path.exists(envs_pos_path):
        os.makedirs(envs_pos_path)

    print("creating trajectories with seed: " + str(args.seed + env_))

    for i in range(traj_num):
        steps = 0
        done = False
        obs = env.reset("random")
        
        trajectory = []
        trajectory_positions = []

        while steps < traj_len and not done:

            # img to gray scale, eventually the model uses gray images
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
            obs = obs[None,:, :]
            
            # scale imgs to [0,1], the model uses this format
            obs = np.array(obs).astype(np.float32) / 255.0
            
            action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).to(device))
            
            obs, reward, done, info = env.step(action.cpu().numpy())
            
            player_pos = np.array(info["player_pos"])

            trajectory.append(obs)
            trajectory_positions.append(player_pos)

        trajectory = np.array(trajectory)
        trajectory_positions = np.array(trajectory_positions)
        
        trajectory_observations_path = envs_obs_path + "trajectory_observations_"  + str(i) + ".npy"
        trajectory_positions_path = envs_pos_path +  "trajectory_positions_" + str(i) + ".npy"

        with open(trajectory_observations_path, 'wb') as tf:
            np.save(tf, trajectory)
        
        with open(trajectory_positions_path, 'wb') as to:
            np.save(to, trajectory_positions)