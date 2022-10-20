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
from src.representations.models.VAE import VanillaVAE_PL

from IPython import embed


class VAE(VanillaVAE_PL):
    def __init__(self, conf):
        
        img_size = conf['img_size']
        obs_shape = (1, img_size, img_size)
        conf['vae']['obs_shape'] = obs_shape
        self._load_clusters = conf["vae"]["load_clusters"]

        super(VAE, self).__init__(obs_shape[0], conf['vae']['z_dim'], hidden_dims = None, **conf['vae'])

        self.path_clusters = Path("src/representations/trajectories/clusters/vae")
        self.experiment = conf['experiment']
        self.batch_size = conf['batch_size']
        self.lr = conf['lr']
        self.split = conf['split']
        self.trajectories = conf['trajectories']
        self.trajectories_train, self.trajectories_val = get_train_val_split(self.trajectories, self.split)

        self.conf = conf['vae']
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test = conf['test']

        self.type = self.test['type']
        self.shuffle = self.test['shuffle']
        self.limit = self.test['limit']

    def training_step(self, batch, batch_idx):
        batch = batch.squeeze(1)

        recons, x, mu, log_var = self.forward(batch)

        losses = self.loss_function(recons, x, mu, log_var)

        loss = losses["loss"]
        recon_loss = losses["Reconstruction_Loss"]
        kl_loss = losses["KLD"]

        # loggers
        if batch_idx % 500 == 0:
            self.sample(num_samples = 8, current_device = 0, logger = self.logger)
            batch_plot = batch[0:8, :, :, :]
            self.generate(batch_plot, logger = self.logger)

        self.log('total_loss/train_epoch', loss, on_step=False, on_epoch=True)
        self.log('recon_loss/train_epoch', recon_loss, on_step=False, on_epoch=True)
        self.log('kl_loss/train_epoch', kl_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.squeeze(1)
        
        recons, x, mu, log_var = self(batch)
        
        losses = self.loss_function(recons, x, mu, log_var)
        
        loss = losses["loss"]
        recon_loss = losses["Reconstruction_Loss"]
        kl_loss = losses["KLD"]

        self.log('total_loss/val_epoch', loss, on_step=False, on_epoch=True)
        self.log('recon_loss/val_epoch', recon_loss, on_step=False, on_epoch=True)
        self.log('kl_loss/val_epoch', kl_loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=1e-5)

    def train_dataloader(self):
        train_dataset = CustomCrafterData(self.trajectories_train, **self.conf)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterData(self.trajectories_val, **self.conf)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)
        return val_dataloader

    
    def on_train_epoch_end(self):
        self.sample(num_samples = 8, current_device = 0, logger = self.logger)

    def store_clusters(self):
        num_clusters = 3
        
        loader = get_loader(
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

