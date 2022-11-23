import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from IPython import embed
import matplotlib.pyplot as plt

import wandb
import numpy as np

class BetaVAE_PL(pl.LightningModule):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels = None,
                 latent_dim = None,
                 hidden_dims = None,
                 beta: int = 4,
                 gamma:float = 10.0,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e4,
                 loss_type:str = 'B',
                 use_semantic = False,
                 **kwargs) -> None:
        super(BetaVAE_PL, self).__init__()


        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.use_semantic = use_semantic

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            #hidden_dims = [64, 128, 256, 512, 1024]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()),
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], out_channels=hidden_dims[i],
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm2d(hidden_dims[i]),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

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
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, labels=None):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        
        if labels is None:
            return  [self.decode(z), input, mu, log_var]
        else:
            return  [self.decode(z), labels, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        if not kwargs["use_semantic"]:
            recons_loss = F.mse_loss(recons, input)
        else:
            recons_loss = F.mse_loss(recons, input)
            print("Reconstruction loss:", recons_loss)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, logger = None, use_semantic = False):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        if logger is not None:
            if not use_semantic:
                fig, axs = plt.subplots(1, num_samples, sharey=True, figsize=(18, 2))

                for s in range(num_samples):
                    axs[s].imshow(samples[s,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
                    axs[s].axis('off')
                
                logger.experiment.log({'VAE samples': fig})
                plt.close(fig)
            else:
                sample = plot_local_mask(samples[0,:].squeeze(0).cpu().detach().numpy(), mode = "sample:") 
                logger.experiment.log({'VAE samples': wandb.Image(sample)})

        return samples

    def generate(self, data, logger = None, use_semantic = False):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        
        if logger is not None:

            if not use_semantic:
                recons = self.forward(data)[0]
                num_examples = data.size(0)

                fig, axs = plt.subplots(2, num_examples, figsize=(18, 4))

                for s in range(num_examples):
                    axs[0,s].imshow(data[s,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
                    axs[1,s].imshow(recons[s,:,:,:].permute(1,2,0).cpu().detach().numpy(), interpolation='nearest')
                    
                    axs[0,s].axis('off')
                    axs[1,s].axis('off')

                logger.experiment.log({'VAE reconstructions': wandb.Image(fig)})
            
            else:
                x, y = data[0], data[1]
                recons = self.forward(x)[0]
                num_examples = 1    

                y = plot_local_mask(y[0,:].squeeze(0).cpu().detach().numpy(), mode = "ground truth:")
                recons = plot_local_mask(recons[0,:].squeeze(0).cpu().detach().numpy(), mode = "reconstruction:")
                
                logger.experiment.log({'True semantic': wandb.Image(y)})
                logger.experiment.log({'VAE reconstruction': wandb.Image(recons)})

        return recons


