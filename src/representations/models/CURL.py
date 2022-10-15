import os
import numpy as np
import torch
import torch.nn as nn
from random import randint
import pytorch_lightning as pl
from IPython import embed
from pathlib import Path

from src.utils.networks import Encoder


class CURL_PL(pl.LightningModule):
    """
    CURL
    """

    def __init__(self,
            obs_shape=(1,64,64),
            z_dim=50,
            output_type="continuous",
            path_clusters = '/home/roger/Desktop/modded-crafter/src/representations/trajectories/clusters',
            device=None,
            **kwargs
            ):
        super(CURL_PL, self).__init__()

        self.encoder = Encoder(obs_shape[0], z_dim)
        self.encoder_target = Encoder(obs_shape[0], z_dim)

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type
        
        self.dev = "cuda"
        if path_clusters is not None:
            self.path_clusters = Path(path_clusters)
            self.clusters = self.load_clusters()

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos=None):
        if z_pos == None:
            z_pos = self.clusters
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        return logits

    def compute_train(self, z_a, z_pos):

        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        logits = self.compute_logits(z_a, z_pos)
        return logits - torch.max(logits, 1)[0][:, None]

    def compute_argmax(self, z_a, z_pos=None):
        logits = self.compute_logits(z_a, z_pos)
        return torch.argmax(logits).cpu().item()

    def load_clusters(self):
        clusters = []

        for gs in sorted(os.listdir(self.path_clusters)):
            if 'npy' in gs:
                clusters.append(np.load(os.path.join(self.path_clusters, gs)))
        
        clusters = np.array(clusters)
        clusters = torch.from_numpy(clusters).squeeze().float().to(self.dev)
        
        return clusters
