import yaml
import torch
import random
import numpy as np
from pathlib import Path

def setSeed(manualSeed):
	np.random.seed(manualSeed)
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)
	torch.cuda.manual_seed(manualSeed)
	torch.cuda.manual_seed_all(manualSeed)

	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def getConfig(file):
	with open("src/config/" + file + ".yml", 'r') as ymlfile:
		return yaml.load(ymlfile, Loader=yaml.Loader)