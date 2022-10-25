import os
import cv2
import torch
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import shuffle
from IPython import embed

from pathlib import Path


class CustomCrafterData(Dataset):
    def __init__(self, traj_list, path="src/representations/trajectories/", delay=False):
        
        #self.path = Path(path) 
        self.path = Path("/network/scratch/r/roger.creus-castanyer/tmp/") 
        self.traj_list = traj_list
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.customLoad()

    def getImage(self, idx):
        query = torch.FloatTensor(self.data[idx])
        return query

    def customLoad(self):
        print("Loading data...")
        data = []

        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(str(self.path) + "/" + traj, allow_pickle=True)
            data.append(obs.flatten())
    
        data = np.concatenate(np.array(data, dtype='object'))
        print("current shape:", data.shape)
        data = data.reshape(-1, 64, 64, 3)
        print("reshaped shape:", data.shape)
        self.data = data

        print("Loaded data of shape: " + str(data.shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Get query obs
        x = self.getImage(index)

        # for RGB images only
        x = torch.div(x.unsqueeze(0).permute(0,3,1,2), 255)
        return x




class CustomCrafterData_CURL(Dataset):
    def __init__(self, traj_list, path="src/representations/trajectories/observations/", delay=False, **kwargs) -> None:
        
        try:
            #self.path = Path(path + kwargs['trajectories'][0]) 
            self.path = Path("/network/scratch/r/roger.creus-castanyer/tmp/") 
            self.k_std = kwargs['k_std']
            self.k_mean = kwargs['k_mean']
        except:
            self.path = Path(path + kwargs["cnn"]['trajectories'][0]) 

        self.traj_list = traj_list
        self.delay = delay
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
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
        # for RGB images only
        query = torch.div(query, 255)
        
        key = torch.FloatTensor(self.data[key_idx])
        key = torch.div(key, 255)
        return key, query


    def customLoad(self):
        print("Loading data...")
        data, list_idxs = [], []

        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(str(self.path) + "/" + traj, allow_pickle=True)
            data.append(obs)
            list_idxs.append(obs.shape[0])

        data = np.concatenate(np.array(data, dtype='object')).reshape(-1, 64, 64, 3)
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