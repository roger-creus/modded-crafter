import os
import sys
import torch
from os.path import join
import numpy as np
from pathlib import Path

from src.config import setSeed, getConfig
from src.representations.main.curl import CURL
from src.utils.networks import Agent

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])
    
path_weights = Path('/home/roger/Desktop/modded-crafter/src/checkpoints')

agent = Agent(conf = conf).cuda()
checkpoint = torch.load(join(path_weights, conf['test']['path_weights']))
agent.load_state_dict(checkpoint)

# Do it in two steps since anyways we need to store goal states in numpy arrays.
# First compute and store goal states (centroides with kmeans)
# Then only compute index_maps or reward_maps

#agent.store_clusters()

agent._construct_map()
