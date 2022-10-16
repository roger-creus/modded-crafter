from re import I
import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical
from numpy.linalg import norm

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
    def __init__(self, num_actions = 17, in_channels = 1, z_dim = 512, hidden_channels = 32,conf = None):
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

        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.dev = "cuda"

        if conf is not None:
            self.trajectories = conf["cnn"]["trajectories"]
            self.conf = conf
            self.path_clusters = Path("src/representations/trajectories/clusters/cnn")
            self._load_clusters = conf["cnn"]["load_clusters"]
            
            if self._load_clusters:
                self.clusters = self.load_clusters()

    def encode(self, x):
        return self.network(x)

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

    def compute_argmax(self, z_a):
        max_sim = -9999999
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        label = 0
        i = 0
        for c in self.clusters:
            cosine = cos(z_a, c)
            if cosine.item() > max_sim:
                label = i
                max_sim = cosine.item()
            i+=1
        return label

    def load_clusters(self):
        clusters = []

        for gs in sorted(os.listdir(self.path_clusters)):
            if 'npy' in gs:
                clusters.append(np.load(os.path.join(self.path_clusters, gs)))
        
        clusters = np.array(clusters)
        clusters = torch.from_numpy(clusters).squeeze().float().to(self.dev)
        
        return clusters

    def store_clusters(self):
        store_clusters_cnn(self.trajectories, self.conf, self.encode, self.path_clusters)

    def _construct_map(self):
        construct_map(self)
