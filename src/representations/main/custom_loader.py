import os
import cv2
import torch
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import Dataset
from random import shuffle
from IPython import embed

from pathlib import Path



class CustomCrafterData(Dataset):
    def __init__(self, traj_list, path="src/representations/trajectories/observations/", delay=False, **kwargs) -> None:
        self.path = Path(path)
        self.traj_list = traj_list
        self.delay = delay
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.k_std = kwargs['k_std']
        self.k_mean = kwargs['k_mean']
        self.customLoad()

    def getTrajLastIdx(self, idx):
        list_idxs = self.list_idxs
        idx_acc = 0
        for i in list_idxs:
            if idx >= idx_acc and idx < (idx_acc + i):
                return idx_acc + i - 1
            idx_acc += i
        return None

    def getImages(self, idx, key_idx):
        query = torch.FloatTensor(self.data[idx])
        key = torch.FloatTensor(self.data[key_idx])
        return key, query


    def customLoad(self):
        print("Loading data...")
        data, list_idxs = [], []

        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(str(self.path) + "/" + traj, allow_pickle=True)
            data.append(obs)
            list_idxs.append(obs.shape[0])

        data = np.concatenate(np.array(data, dtype='object')).reshape(-1, 64, 64, 1)
        self.data = data
        self.list_idxs = list_idxs

        print("Loaded data of shape: " + str(data.shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Get query obs
        if self.delay:
            # Make sure that we pick a frame from the same trajectory
            fin_idx = self.getTrajLastIdx(index)
            key_idx = index + int(np.random.rand()*self.k_std + self.k_mean)

            # Get key obs
            key_idx = min(key_idx, fin_idx)
        else:
            key_idx = index

        key, query = self.getImages(index, key_idx)
        x = torch.stack((key, query))
        x = x.permute(0,3,1,2)
        return x


def trainValSplit(traj_list, split):
    num_traj = len(traj_list)

    if split == 1:
        return traj_list, []
    else:
        shuffle(traj_list)
        return traj_list[:int(split*num_traj)], traj_list[int(split*num_traj):]


def get_train_val_split(t, split):
    path = Path('src/representations/trajectories/')
    #total_t = []
    items = sorted(os.listdir(path / t[0]), key=lambda x: int(x.split('.')[0].split('_')[2]))
    #items = [path / t[0] / x for x in items]
    #total_t.extend(items)
    return trainValSplit(items, split)