import pytorch_lightning as pl
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from IPython import embed
import wandb

def build_mlp(input_dim, output_dim, hidden_units=[64, 64], hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    
    if output_activation is not None:
        layers.append(output_activation)
    
    return nn.Sequential(*layers)

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

class FixedGaussian(pl.LightningModule):
    """
    Fixed diagonal gaussian distribution.
    """
    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(pl.LightningModule):
    """
    Diagonal gaussian distribution with state dependent variances.
    """
    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std

class Decoder(pl.LightningModule):

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()

        
        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        """

        modules = []
        hidden_dims = [512, 256, 128, 64, 32]
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dims[0] * 4)
        ).apply(initialize_weight)

        self.decoder = nn.Sequential(*modules).apply(initialize_weight)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh()).apply(initialize_weight)
        """

        self.std = std

    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)

class Encoder(pl.LightningModule):

    def __init__(self, input_dim=3, output_dim=256):
        super(Encoder, self).__init__()

        
        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        """
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
            ))
            input_dim = h_dim

        self.encoder = nn.Sequential(*modules).apply(initialize_weight)
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        ).apply(initialize_weight)
        """

    def forward(self, x):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        x = self.net(x)
        x = x.view(B, S, -1)
        return x

class SEQ_VAE(pl.LightningModule):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    """
    def __init__(
        self,
        in_channels,
        num_actions,
        latent_dim,
        z1_dim,
        z2_dim,
        hidden_units,
        **kwargs
    ):
        super(SEQ_VAE, self).__init__()
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, (hidden_units, hidden_units))
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            z2_dim + num_actions,
            z1_dim,
            (hidden_units, hidden_units),
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + num_actions,
            z2_dim,
            (hidden_units, hidden_units),
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(latent_dim, z1_dim, (hidden_units, hidden_units))
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            latent_dim + z2_dim + num_actions,
            z1_dim,
            (hidden_units, hidden_units),
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # feat(t) = Encoder(x(t))
        self.encoder = Encoder(in_channels, latent_dim)
        
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            z1_dim + z2_dim,
            in_channels,
            std=np.sqrt(0.1),
        )
        
        self.apply(initialize_weight)

   
    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    
    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):

            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2,  actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    def calculate_loss(self, state_, action_):
        # change actions to one hot
        action_ = F.one_hot(action_[:, :-1, :].long(), num_classes = 17).squeeze(2)

        # Calculate the sequence of features.
        feature_ = self.encoder(state_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_image = -log_likelihood_.mean(dim=0).sum()

        return loss_kld, loss_image

    def evaluate_reconstruction(self, state_, action_, logger):
        action_ = F.one_hot(action_[:, :-1, :].long(), num_classes = 17).squeeze(2)

        with torch.no_grad():
            feature_ = self.encoder(state_)

            z1_mean_, z1_std_, z1_, z2_ = self.sample_posterior(feature_, action_)

            z_ = torch.cat([z1_, z2_], dim=-1)

            mean, cov = self.decoder(z_)

            reconstruction = mean.squeeze(0)

        logger.experiment.log({
            "true observation" : [wandb.Image(s) for s in state_.squeeze(0)],
            "posterior sample" : [wandb.Image(r) for r in reconstruction],
        })