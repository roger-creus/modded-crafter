from re import I
import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical
from numpy.linalg import norm

from src.utils.utils import * 
from src.representations.models.VAE import * 
from src.representations.models.CURL import * 

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
    def __init__(
            self,
            pretrained_curl = False,
            pretrained_vae = False,
            fine_tune = False,
            num_actions = 17,
            in_channels = 1,
            z_dim = 512,
            hidden_channels = 32,
            conf = None
        ):
        super().__init__()

        self.dev = "cuda"

        if pretrained_curl == True:
            self.network = CURL_PL(z_dim=256)
            self.network.load_state_dict(torch.load("/home/mila/r/roger.creus-castanyer/modded-crafter/crafter/2vl3hd9z/checkpoints/epoch=29-step=253410.ckpt")["state_dict"])
            self.network = self.network.encoder


            if fine_tune:
                for param in self.network.parameters():
                    param.requires_grad = True

            self.encoder_used = "curl"
        
            print("TRAINING WITH CURL")

        elif pretrained_vae == True:
            self.network = VanillaVAE_PL(latent_dim = z_dim)
            self.network.load_state_dict(torch.load("/home/mila/r/roger.creus-castanyer/modded-crafter/crafter/37mv5msc/checkpoints/epoch=99-step=1696500.ckpt")["state_dict"])

            for name, param in self.network.named_parameters():
                if "decoder" in name or "final" in name:
                    param.requires_grad = False
                else:
                    if fine_tune:
                        param.requires_grad = True
                        print("param that will be finetuned:", name)

            self.encoder_used = "vae"

            print("TRAINING WITH VAE")
                    
        else:
            modules = []
            hidden_dims = [32, 64, 128, 256, 512]
            in_channels = 3
            
            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.network = nn.Sequential(*modules)
            
            self.fc = nn.Linear(hidden_dims[-1] * 2 * 2, 512)

            self.encoder_used = "vanilla"

        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        if conf is not None:
            self.trajectories = conf["cnn"]["trajectories"]
            self.conf = conf
            self.path_clusters = Path("src/representations/trajectories/clusters/cnn")
            self._load_clusters = conf["cnn"]["load_clusters"]
            
            if self._load_clusters:
                self.clusters = self.load_clusters()

    def encode(self, x):
        if self.encoder_used == "vae":
            mu, log_var = self.network.encode(x)
            return self.network.reparameterize(mu, log_var)
            
        x = self.network(x)
        x = x.flatten(start_dim=1, end_dim= -1)
        x = self.fc(x)
        
        return x

    def get_value(self, x):
        #x = x.squeeze(2)
        return self.critic(self.encode(x))

    def get_action_and_value(self, x, action=None):
        #x = x.squeeze(2)
        hidden = self.encode(x)
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


class Agent_MLP(nn.Module):
    def __init__(self, hidden_channels = 32):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = layer_init(nn.Conv2d(35, hidden_channels, 3, stride=1))
        self.conv2 = layer_init(nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, stride=1))
        self.fc = layer_init(nn.Linear(hidden_channels * 2 * 1 * 3, 128))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(128 , 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 17), std=0.01)
        )

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return self.relu(x)

    def get_value(self, x):
        x = x.squeeze(1)
        x = self.encode(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.squeeze(1)
        x = self.encode(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
