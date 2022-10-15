import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.utils.utils import * 


class Encoder(nn.Module):
    def __init__(self, in_channels = 1, z_dim = 512, hidden_channels = 32, stride = 4):
        super().__init__()
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, hidden_channels, 4, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(hidden_channels * 2 * 5 * 5, z_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = Encoder()

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        #x = x.squeeze(2)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        #x = x.squeeze(2)
        hidden = self.network(x)
        logits = self.actor(hidden)
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)