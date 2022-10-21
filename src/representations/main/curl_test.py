import os
import sys
import torch
from os.path import join
import numpy as np
from pathlib import Path

from src.config import setSeed, getConfig
from src.representations.main.curl import CURL

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])
    
path_weights = Path('/home/mila/r/roger.creus-castanyer/modded-crafter/crafter/2vl3hd9z/checkpoints')

conf['curl']['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curl = CURL(conf).cuda()
checkpoint = torch.load(join(path_weights, conf['test']['path_weights']))
curl.load_state_dict(checkpoint['state_dict'])

# Do it in two steps since anyways we need to store goal states in numpy arrays.
# First compute and store goal states (centroides with kmeans)
# Then only compute index_maps or reward_maps

#curl.store_clusters()

curl._construct_map()
