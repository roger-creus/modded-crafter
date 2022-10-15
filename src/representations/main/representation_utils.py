import os
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import shuffle
from IPython import embed
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd
from crafter.env import Env

import warnings
warnings.filterwarnings('ignore', '.*future.*', )

from src.representations.main.custom_loader import *
from src.representations.models.CURL import CURL_PL

def trainValSplit(traj_list, split):
    num_traj = len(traj_list)

    if split == 1:
        return traj_list, []
    else:
        shuffle(traj_list)
        return traj_list[:int(split*num_traj)], traj_list[int(split*num_traj):]


def get_train_val_split(t, split):
    path = Path('src/representations/trajectories/')
    items = sorted(os.listdir(path / t[0]), key=lambda x: int(x.split('.')[0].split('_')[2]))
    return trainValSplit(items, split)

def get_loader(trajectories, conf, shuffle=False):
    train, _ = get_train_val_split(trajectories, 1)
    train_dataset = CustomCrafterData(train, delay=False, **conf)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    return train_dataloader

def load_trajectories_positions():
    print("loading trajectories positions...")

    all_trajectories = []
    files = sorted([x for x in os.listdir(f"src/representations/trajectories/positions/")],  key=lambda x: int(x.split('.')[0].split('_')[2]))

    for file in files:
        all_trajectories.append(np.load("src/representations/trajectories/positions/"+file))

    f = np.concatenate(all_trajectories)
    return f

def compute_embeddings_curl(loader, encode):
    print("computing embeddings...")
    return np.array([encode(data[:,0].cuda()).detach().cpu().numpy() for data in loader]).squeeze()

def compute_kmeans(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

def construct_map(enc):
    loader = get_loader(enc.trajectories,enc.conf)

    embeddings = compute_embeddings_curl(loader, enc.encode)

    trajectories_positions = load_trajectories_positions()

    index_map(trajectories_positions, embeddings, enc)

def index_map(trajectories_positions, embeddings, enc):
    print("Get index from all data points...")
    values = pd.DataFrame(columns=['x', 'y', 'k'])

    colors = {0:'blue', 1:'green', 2:'red', 3:'purple', 4:'brown'}

    #embeddings = embeddings[0:5000, :]
    #trajectories_positions = trajectories_positions[0:2500, :]

    cluster_mappings = []
    x_list = []
    y_list = []
    for i, (e, p) in enumerate(zip(embeddings, trajectories_positions)):
        x = int(p[0])
        y = int(p[1])
        e = torch.from_numpy(e).cuda()
        k = enc.compute_argmax(e.unsqueeze(dim=0))
        x_list.append(x)
        y_list.append(y)
        cluster_mappings.append(int(k))

    values['x'] = x_list
    values['y'] = y_list
    values['k'] = cluster_mappings
    values['k'] = values['k'].astype('int32')

    embed()
    env = Env(seed = 999, view = (64,64))
    obs = env.reset()
    world_img = env.render_world()
    plt.scatter(values['x'], values['y'], c=values['k'].map(colors))
    plt.savefig("index_map.png")

    
