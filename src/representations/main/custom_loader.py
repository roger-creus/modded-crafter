import os
import cv2
import torch
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import shuffle
from IPython import embed
import torch.nn.functional as F

from pathlib import Path

class CustomCrafterData(Dataset):
    def __init__(self, traj_list, delay=False):
        
        self.path = Path("/home/roger/Desktop/modded-crafter/src/representations/trajectories/tmp") 
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
        x = x.unsqueeze(0).permute(0,3,1,2)
        return x

class CustomCrafterSeqData(Dataset):
    def __init__(self, traj_list, delay=False):
        
        self.path = Path("/network/scratch/r/roger.creus-castanyer/tmp") 
        self.traj_list = traj_list
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.customLoad()

    def getSequence(self, idx):
        obs = torch.FloatTensor(self.data[idx].astype(float))
        acts = torch.FloatTensor(self.action_data[idx].astype(float))
        return obs, acts

    def customLoad(self):
        print("Loading data...")
        data = []
        action_data = []

        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')

            env_num = traj.split("/")[1]
            traj_num = traj.split("/")[2].split(".")[0].split("_")[2]

            act_traj = "actions/" + env_num + "/" + "trajectory_actions_" + traj_num + ".npy"

            obs = np.load(str(self.path) + "/" + traj, allow_pickle=True)
            acts = np.load(str(self.path) + "/" + act_traj, allow_pickle=True)[:,None]
            
            seq_len = 8
            
            ### load seqs of observations ###
            num_seqs = int(np.floor(obs.shape[0] / seq_len))

            if num_seqs == 0:
                continue

            splits = np.array_split(obs, num_seqs)
            
            for i in range(num_seqs):
                while len(splits[i]) > seq_len:
                    splits[i] = np.array(splits[i][1:])
            
            splits = np.stack(splits)

            ### load seqs of actions ###
            num_seqs = int(np.floor(acts.shape[0] / seq_len))
            splits_act = np.array_split(acts, num_seqs)
            
            for i in range(num_seqs):
                while len(splits_act[i]) > seq_len:
                    splits_act[i] = np.array(splits_act[i][1:])
            
            splits_act = np.stack(splits_act)
            
            data.append(splits)
            action_data.append(splits_act)

    
        data = np.concatenate(np.array(data, dtype='object'))
        action_data = np.concatenate(np.array(action_data, dtype='object'))
        #print("current shape:", data.shape)
        #data = data.reshape(-1, 64, 64, 3)
        #print("reshaped shape:", data.shape)
        
        self.data = data
        self.action_data = action_data

        print("Loaded data of shape: " + str(data.shape))
        print("Loaded action data of shape: " + str(action_data.shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Get query obs
        obs, acts = self.getSequence(index)

        obs = obs.permute(0,3,1,2)
        return obs, acts


class CustomCrafterData_SEMANTIC(Dataset):
    def __init__(self, traj_list, path = "/network/scratch/r/roger.creus-castanyer/tmp/", delay=False, **kwargs) -> None:
        
        self.path = Path(path) 
        self.traj_list = traj_list
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.customLoad()

    def getImage(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.semantics[idx])

        y = torch.rot90(y.reshape(9,9))
        y = torch.flip(y, dims=(0,))
        y = y.flatten()

        return x, y


    def customLoad(self):
        print("Loading data...")
        data, semantics = [], []

        for i, traj in enumerate(self.traj_list):
            env_name = traj.split("/")[1]
            traj_num = traj.split("/")[2].split("_")[2].split(".")[0]

            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(str(self.path) + "/" + traj, allow_pickle=True)
            semantic = np.load(str(self.path) + "/semantics/" + str(env_name) + "/trajectory_semantics_" + str(traj_num) + ".npy", allow_pickle=True)

            data.append(obs)
            semantics.append(semantic.flatten())

        data = np.concatenate(np.array(data, dtype='object')).reshape(-1, 64, 64, 3)
        semantics = np.concatenate(np.array(semantics, dtype='object')).reshape(-1, 81)
        
        self.data = data
        self.semantics = semantics

        print("Loaded data of shape: " + str(data.shape))
        print("Loaded semantics of shape: " + str(semantics.shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Get query obs
        x, y = self.getImage(index)

        # for RGB images only
        x = torch.div(x.unsqueeze(0).permute(0,3,1,2), 255)
        y = y.unsqueeze(0)

        img_label = y[:, 0:63]
        img_label_one_hots =  F.one_hot(img_label.to(torch.int64), num_classes = 19)
        
        inventory_label = torch.clip(y[:, 63:], 0)
        inventory_label_one_hots = F.one_hot(inventory_label.to(torch.int64), num_classes = 10)

        return x, (img_label_one_hots, inventory_label_one_hots)


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