from src.representations.main.custom_loader import *
from src.representations.main.representation_utils import get_train_val_split

trajectories = ["observations/env_1", "observations/env_2", "observations/env_3"]
path = "/home/roger/Desktop/modded-crafter/src/representations/trajectories/tmp"

train_traj_list, val = get_train_val_split(trajectories, 0.9, path=path)

loader = CustomCrafterData_SEMANTIC(traj_list = train_traj_list, path = path )

loader[5]
