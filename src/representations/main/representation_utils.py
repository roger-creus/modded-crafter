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


def get_train_val_split(t, split, path="/home/roger/Desktop/modded-crafter/src/representations/trajectories/tmp"):
    path = Path(path)
    all_trajectories = []
    for l in t:
        items = sorted(os.listdir(path / l), key=lambda x: int(x.split('.')[0].split('_')[2]))
        items = [l + "/" + s for s in items]
        all_trajectories.append(items)

    all_trajectories = np.concatenate(all_trajectories)
    return trainValSplit(all_trajectories, split)

def get_loader(trajectories, conf, shuffle=False):
    train, _ = get_train_val_split(trajectories, 1)
    train_dataset = CustomCrafterData(train, delay=False, **conf)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    return train_dataloader

def get_loader_CURL(trajectories, conf, shuffle=False):
    train, _ = get_train_val_split(trajectories, 1)
    train_dataset = CustomCrafterData_CURL(train, delay=False, **conf)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    return train_dataloader

def load_trajectories_positions(trajectories):
    print("loading trajectories positions...")
    env_path = trajectories[0].split("/")[1]
    path_positions = os.environ["SCRATCH"] + "/tmp/positions/" + env_path + "/"
    
    files = sorted([x for x in os.listdir(path_positions)],  key=lambda x: int(x.split('.')[0].split('_')[2]))
        
    trajectories = []
    for file in files:
        trajectories.append(np.load(path_positions + file))
        
    f = np.concatenate(trajectories)
    return f

def compute_embeddings_curl(loader, encode):
    print("computing embeddings curl...")
    return np.array([encode(data[:,0].cuda()).detach().cpu().numpy() for data in loader]).squeeze()


def compute_kmeans(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

def construct_map(enc):
    loader = get_loader(enc.trajectories,enc.conf)

    embeddings = compute_embeddings_curl(loader, enc.encode)

    trajectories_positions = load_trajectories_positions(enc.trajectories)

    index_map(trajectories_positions, embeddings, enc)

def index_map(trajectories_positions, embeddings, enc):
    print("Get index from all data points...")
    values = pd.DataFrame(columns=['x', 'y', 'k'])

    #colors = {0:(210./255. , 238./255., 130./255., 0.15), 1:(40./255., 67./255., 135./255., 0.15),  2:(251./255., 192./255., 147./255., 0.15)}
    colors = {0: (1,0,0,0.05), 1: (0,1,0,0.05), 2: (0,0,1,0.05), 3: (1,1,0,0.05), 4: (1,0,1,0.05)}

    seed = int(enc.trajectories[0].split("/")[1].split("_")[1])

    cluster_mappings = []
    x_list = []
    y_list = []
    for i, (e, p) in enumerate(zip(embeddings, trajectories_positions)):
        x = int(p[0]) * 64
        y = int(p[1]) * 64
        e = torch.from_numpy(e).cuda()
        k = enc.compute_argmax(e.unsqueeze(dim=0))
        x_list.append(x)
        y_list.append(y)
        cluster_mappings.append(int(k))

    values['x'] = x_list
    values['y'] = y_list
    values['k'] = cluster_mappings
    values['k'] = values['k'].astype('int32')

    env = Env(seed = seed, view = (64,64))
    obs = env.reset()
    world_img = env.render_world()

    plt.scatter(values['x'], values['y'], c=values['k'].map(colors))
    plt.imshow(world_img.transpose(1,0,2))
    plt.savefig("/home/mila/r/roger.creus-castanyer/modded-crafter/imgs/index_map_" + enc.trajectories[0].split("/")[1] + ".png")



assets_path = "/home/mila/r/roger.creus-castanyer/modded-crafter/crafter/assets/"
map_semantic = {
    0 : "unknown.png",
    1 : "water.png",
    2 : "grass.png",
    3 : "stone.png",
    4 : "path.png",
    5 : "sand.png",
    6 : "tree.png",
    7 : "lava.png",
    8 : "coal.png",
    9 : "iron.png",
    10 : "diamond.png",
    11 : "table.png",
    12 : "furnace.png",
    13 : "player-down.png",
    14 : "cow.png",
    15 : "zombie.png",
    16 : "skeleton.png",
    17 : "arrow-right.png",
    18 : "plant.png"
}

map_inventory = {
    -1 : "unknown.png",
    0: "unknown.png",
    1 : "1.png",
    2 : "2.png",
    3 : "3.png",
    4 : "4.png",
    5 : "5.png",
    6 : "6.png",
    7 : "7.png",
    8 : "8.png",
    9 : "9.png"
}

def plot_local_mask(semantic):
    n_rows = 9
    n_cols = 7
    fig, ax = plt.subplots(9,9)

    semantic = semantic.reshape(9,9)

    semantic_img = semantic[0:7, :]
    inventory = np.transpose(semantic[7:9, :]).flatten()
    
    for i in range(n_rows):
        for j in range(n_cols):
            ax[j,i].imshow(plt.imread(assets_path + map_semantic[semantic_img[j,i]]))
            ax[j,i].axis('off')

    for i in range(n_rows):
        inv = inventory[i]
        
        ax[7, i].imshow(plt.imread(assets_path + map_inventory[inv]))
        ax[7, i].axis('off')

    for i in range(n_rows):
        if i + n_rows < len(inventory):
            inv = inventory[i + n_rows]
            ax[8, i].imshow(plt.imread(assets_path + map_inventory[inv]))
            ax[8, i].axis('off')

    ax[8, 7].axis('off')
    ax[8, 8].axis('off')

    plt.close()

    return fig