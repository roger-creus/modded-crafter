import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.representations.main.custom_loader import *
from src.representations.main.representation_utils import *
import wandb

from collections import OrderedDict

class SEQ_VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(SEQ_VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = "cuda:0"

        self.batch_size = self.params['batch_size']
        self.split = self.params['split']
        self.trajectories = self.params['trajectories']
        self.trajectories_train, self.trajectories_val = get_train_val_split(self.trajectories, self.split)

        self.hold_graph = False
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def train_dataloader(self):
        train_dataset = CustomCrafterSeqData(self.trajectories_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterSeqData(self.trajectories_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return val_dataloader

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        state_, action_ = batch

        loss_kl, loss_img = self.model.calculate_loss(state_, action_)

        # evalutation (reconstructions)
        if batch_idx % 10 == 0:
            state_plot_, action_plot_ = batch[0].unsqueeze(0)
            self.model.evaluate_reconstruction(state_plot_, action_plot_, logger = self.logger)

        self.log('kl_loss/train_epoch', loss_kl, on_step=False, on_epoch=True, sync_dist = True)
        self.log('recon_loss/train_epoch', loss_img, on_step=False, on_epoch=True, sync_dist = True)

        return loss_img + loss_kl

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        state_, action_ = batch
        loss_kl, loss_img = self.model.calculate_loss(state_, action_)

        # evalutation (reconstructions)
        """
        if batch_idx % 2000 == 0:
            self.model.sample(num_samples = 8, current_device = 0, logger = self.logger)
            batch_plot = batch[0:8, :, :, :]
            self.model.generate(batch_plot, logger = self.logger)
        """
        self.log('kl_loss/val_epoch', loss_kl, on_step=False, on_epoch=True, sync_dist = True)
        self.log('recon_loss/val_epoch', loss_img, on_step=False, on_epoch=True, sync_dist = True)

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        return optims