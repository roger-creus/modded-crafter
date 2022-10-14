import gym
import argparse
import crafter
import numpy as np
import os
from IPython import embed
import torch
import torch.nn as nn

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.wrappers import *
from utils.utils import *
from utils.networks import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.vector.SyncVectorEnv(
        [make_env("CrafterReward-v1", np.random.randint(100), 999, True, "eval_crafter") for i in range(1)]
    )

agent = Agent(env).to(device)
agent.load_state_dict(torch.load("/home/roger/Desktop/solving-crafter/src/checkpoints/ppo-14254080.pt"))

obs = env.reset()
done = False
steps = 0
reward = 0
while not done and steps < 10000:       
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs).to(device))
        obs, r, done, info = env.step(action.cpu().numpy())
        reward += r
        steps += 1

print("Achieved reward: " + str(reward) + " and survived for: " + str(steps) + " steps")