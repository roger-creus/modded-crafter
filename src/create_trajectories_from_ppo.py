import gym
import argparse
import crafter
import numpy as np
import os
import random

from utils.wrappers import *
from utils.utils import plot_local_mask

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
#agent.load_state_dict(torch.load("/home/roger/Desktop/modded-crafter/src/checkpoints/ppo-14254080.pt"))

os.makedirs(args.save_path + "/tmp/" + "observations/")
os.makedirs(args.save_path + "/tmp/" + "semantics/")
os.makedirs(args.save_path + "/tmp/" + "positions/")

for env_ in range(envs_num):
    
    env = Env(seed = args.seed + env_)  # Or CrafterNoReward-v1

    envs_path = "env_" + str(args.seed + env_) + "/"

    envs_obs_path = args.save_path +  "/tmp/" + "observations/" + envs_path
    envs_semantics_path = args.save_path +  "/tmp/" + "semantics/" + envs_path
    envs_pos_path = args.save_path +  "/tmp/" + "positions/" + envs_path
    
    if not os.path.exists(envs_obs_path):
        os.makedirs(envs_obs_path)
    
    if not os.path.exists(envs_semantics_path):
        os.makedirs(envs_semantics_path)
    
    if not os.path.exists(envs_pos_path):
        os.makedirs(envs_pos_path)

    print("creating trajectories with seed: " + str(args.seed + env_))

    for i in range(traj_num):
        steps = 0
        done = False
        obs = env.reset("random")
        #obs, reward, done, info = env.step(15)
        
        trajectory = []
        trajectory_semantics = []
        trajectory_positions = []

        while steps < traj_len and not done:

            # img to gray scale, eventually the model uses gray images
            gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            gray_obs = cv2.resize(gray_obs, (64, 64), interpolation=cv2.INTER_AREA)
            gray_obs = gray_obs[None,:, :]
            
            # scale imgs to [0,1], the model uses this format
            gray_obs = np.array(gray_obs).astype(np.float32) / 255.0
            
            action, logprob, _, value = agent.get_action_and_value(torch.Tensor(gray_obs).unsqueeze(0).to(device))
            
            next_obs, reward, done, info = env.step(action.item())
            
            # get required info to generate semantic map and record positions
            semantic = info["semantic"]
            player_pos = np.array(info["player_pos"])
            inventory = info["inventory"]

            # get the 7x9 image
            local_semantic = get_local_semantic(semantic, player_pos)

            # get the 2x9 inventory
            inventory = np.pad(np.array(list(info['inventory'].values())).flatten(), [0,2], mode="constant", constant_values = -1).reshape(9,2)

            # append both to have a 9x9 image
            semantic = np.append(local_semantic, inventory, axis = 1)

            trajectory.append(obs)
            trajectory_semantics.append(semantic)
            trajectory_positions.append(player_pos)

            obs = next_obs


        trajectory = np.array(trajectory)
        trajectory_semantics = np.array(trajectory_semantics)
        trajectory_positions = np.array(trajectory_positions)
        
        trajectory_observations_path = envs_obs_path + "trajectory_observations_"  + str(i) + ".npy"
        trajectory_semantics_path = envs_semantics_path + "trajectory_semantics_"  + str(i) + ".npy"
        trajectory_positions_path = envs_pos_path +  "trajectory_positions_" + str(i) + ".npy"

        with open(trajectory_observations_path, 'wb') as tf:
            np.save(tf, trajectory)

        with open(trajectory_semantics_path, 'wb') as ts:
            np.save(ts, trajectory_semantics)
        
        with open(trajectory_positions_path, 'wb') as to:
            np.save(to, trajectory_positions)