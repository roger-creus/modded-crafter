import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from src.representations.main.custom_loader import *
from src.representations.main.representation_utils import *


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

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
        train_dataset = CustomCrafterData(self.trajectories_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterData(self.trajectories_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return val_dataloader


    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        batch = batch.squeeze(1)
        real_img = batch
        
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        if batch_idx % 2000 == 0:
            self.model.sample(num_samples = 8, current_device = 0, logger = self.logger)
            batch_plot = batch[0:8, :, :, :]
            self.model.generate(batch_plot, logger = self.logger)


        self.log('total_loss/train_epoch', train_loss["loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('recon_loss/train_epoch', train_loss["Reconstruction_Loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('kl_loss/train_epoch', train_loss["KLD"], on_step=False, on_epoch=True, sync_dist = True)                                      

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        batch = batch.squeeze(1)
        
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log('total_loss/val_epoch', val_loss["loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('recon_loss/val_epoch', val_loss["Reconstruction_Loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('kl_loss/val__epoch', val_loss["KLD"], on_step=False, on_epoch=True, sync_dist = True) 

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['lr'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims