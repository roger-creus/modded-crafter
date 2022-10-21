import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.representations.main.custom_loader import *
from src.representations.main.representation_utils import *
from src.representations.models.CURL import CURL_PL

from IPython import embed


class CURL(CURL_PL):
    def __init__(self, conf):
        
        img_size = conf['img_size']
        obs_shape = (3, img_size, img_size)
        conf['curl']['obs_shape'] = obs_shape
        self._load_clusters = conf["curl"]["load_clusters"]


        super(CURL, self).__init__(**conf['curl'])

        self.path_clusters = Path("/home/mila/r/roger.creus-castanyer/modded-crafter/src/representations/clusters/curl")
        self.experiment = conf['experiment']
        self.batch_size = conf['batch_size']
        self.lr = conf['lr']
        self.split = conf['split']
        self.delay = conf['delay']
        self.trajectories = conf['trajectories']
        self.trajectories_train, self.trajectories_val = get_train_val_split(self.trajectories, self.split)

        self.tau = conf['tau']
        self.soft_update = conf['soft_update']

        self.conf = conf['curl']
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test = conf['test']

        self.type = self.test['type']
        self.shuffle = self.test['shuffle']
        self.limit = self.test['limit']


    def forward(self, data):
        key, query = data[:,0], data[:,1]

        # Forward tensors through encoder
        z_a = self.encode(key)
        z_pos = self.encode(query, ema=True)

        # Compute distance
        logits = self.compute_train(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)

        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)

        self.log('loss/train_epoch', loss, on_step=False, on_epoch=True)

        if batch_idx % self.soft_update == 0:
            self.soft_update_params()

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.criterion(logits, labels)

        self.log('loss/val_epoch', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, amsgrad=False)

    def train_dataloader(self):
        train_dataset = CustomCrafterData_CURL(self.trajectories_train, delay=self.delay, **self.conf)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterData_CURL(self.trajectories_val, delay=self.delay, **self.conf)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)
        return val_dataloader

    def soft_update_params(self):
        net = self.encoder
        target_net = self.encoder_target
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    
    def store_clusters(self):
        num_clusters = 5
        
        loader = get_loader_CURL(
            self.trajectories,
            self.conf)

        print("computing embeddings")
        embeddings = compute_embeddings_curl(loader, self.encode)

        print("computing clusters")
        kmeans = compute_kmeans(embeddings, num_clusters)

        for i, k in enumerate(kmeans.cluster_centers_):
            with open(f'{str(self.path_clusters)}/{i}.npy', 'wb') as f:
                np.save(f, k)

    def _construct_map(self):
        construct_map(self)

