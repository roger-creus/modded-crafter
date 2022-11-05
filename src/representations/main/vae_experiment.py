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

class VAEXperiment_SEMANTIC(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment_SEMANTIC, self).__init__()

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
        train_dataset = CustomCrafterData_SEMANTIC(self.trajectories_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterData_SEMANTIC(self.trajectories_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return val_dataloader

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        real_img = real_img.squeeze(1)
        labels = labels.squeeze(1)

        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              use_semantic = True,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        if batch_idx % 2000 == 0:
            self.model.sample(num_samples = 1, current_device = 0, logger = self.logger, use_semantic = True)
            x = real_img[0, :, :, :].unsqueeze(0)
            y = labels[0, :].unsqueeze(0)
            self.model.generate([x, y], logger = self.logger, use_semantic = True)

        self.log('total_loss/train_epoch', train_loss["loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('recon_loss/train_epoch', train_loss["Reconstruction_Loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('kl_loss/train_epoch', train_loss["KLD"], on_step=False, on_epoch=True, sync_dist = True)                                      

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        real_img, labels = batch
        real_img = real_img.squeeze(1)
        labels = labels.squeeze(1)

        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            use_semantic = True,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log('total_loss/val_epoch', val_loss["loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('recon_loss/val_epoch', val_loss["Reconstruction_Loss"], on_step=False, on_epoch=True, sync_dist = True)
        self.log('kl_loss/val_epoch', val_loss["KLD"], on_step=False, on_epoch=True, sync_dist = True)  

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
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
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


class SemanticPredictorExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(SemanticPredictorExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        
        weights = torch.load(self.params["weights"])['state_dict']
        weights = OrderedDict({k.replace('model.','') : v for k,v in dict(weights).items()})
        
        self.model.load_state_dict(weights)    
        for param in self.model.parameters():
            param.requires_grad = False

        self.curr_device = None
        self.batch_size = self.params['batch_size']
        self.split = self.params['split']
        self.trajectories = self.params['trajectories']
        self.trajectories_train, self.trajectories_val = get_train_val_split(self.trajectories, self.split)
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 81)
        )

        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def train_dataloader(self):
        train_dataset = CustomCrafterData_SEMANTIC(self.trajectories_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterData_SEMANTIC(self.trajectories_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return val_dataloader


    def forward(self, input, **kwargs):
        mu, log_var = self.model.encode(input)
        z = self.model.reparameterize(mu, log_var)
        p = self.predictor(z)
        return p

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        real_img = real_img.squeeze(1)
        labels = labels.squeeze(1)

        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = F.mse_loss(results, labels)

        if batch_idx % 2000 == 0:
            self.sample(num_samples = 1, current_device = 0)
            x = real_img[0, :, :, :].unsqueeze(0)
            y = labels[0, :].unsqueeze(0)
            self.generate(x, y)

        self.log('train_loss/train_epoch', train_loss, on_step=False, on_epoch=True, sync_dist = True)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        real_img = real_img.squeeze(1)
        labels = labels.squeeze(1)

        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        
        val_loss = F.mse_loss(results, labels)

        self.log('val_loss/val_epoch', val_loss, on_step=False, on_epoch=True, sync_dist = True)

    def sample(self, num_samples, current_device = 0):
        z = torch.randn(num_samples, self.model.latent_dim)
        z = z.to(current_device)

        samples = self.model.decode(z)
        predictions = self.predictor(z)
        
        p = plot_local_mask(predictions[0,:].squeeze(0).cpu().detach().numpy()) 
        
        self.logger.experiment.log({'VAE samples': wandb.Image(samples.squeeze(0).permute(1,2,0).cpu().detach().numpy())})
        self.logger.experiment.log({'VAE samples semantic predictions': wandb.Image(p)})


    def generate(self, inputs, labels):
        recons = self.model.forward(inputs)[0]
        p = self.forward(inputs)

        y = plot_local_mask(labels[0,:].squeeze(0).cpu().detach().numpy())
        predictor = plot_local_mask(p[0,:].squeeze(0).cpu().detach().numpy())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
        ax1.imshow(inputs[0,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
        ax2.imshow(recons[0,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
        ax1.axis('off')
        ax2.axis('off')

        self.logger.experiment.log({'VAE reconstructions': wandb.Image(fig)})
        self.logger.experiment.log({'True semantic': wandb.Image(y)})
        self.logger.experiment.log({'Semantic predictor': wandb.Image(predictor)})

        plt.close(fig)

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
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomCrafterData(self.trajectories_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return val_dataloader


    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        batch = batch.squeeze(1)
        real_img = batch
        
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              use_semantic = False,
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
                                            use_semantic = False,
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