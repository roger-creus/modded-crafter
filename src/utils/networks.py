from re import I
import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical
from numpy.linalg import norm

from src.utils.utils import * 
from src.representations.models.VAE import * 
from src.representations.models.CURL import * 

class Encoder(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        
        modules = []
        self.hidden_dims = [32, 64, 128, 256, 512]
        
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

class Agent(nn.Module):
    def __init__(
            self,
            pretrained_curl = False,
            pretrained_vae = False,
            fine_tune = False,
            num_actions = 17,
            in_channels = 3,
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
        
            print("----------Training with pretrained CURL----------")

        elif pretrained_vae == True:
            self.network = VanillaVAE_PL(latent_dim = z_dim)
            self.network.load_state_dict(torch.load("/home/mila/r/roger.creus-castanyer/modded-crafter/crafter/9wbbe8c7/checkpoints/epoch=29-step=253410.ckpt")["state_dict"])

            for name, param in self.network.named_parameters():
                if "decoder" in name or "final" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    print("param that will be finetuned:", name)

            self.encoder_used = "vae"

            print("----------Training with pretrained VAE----------")
                    
        else:
            self.network = Encoder(in_channels)
            self.fc = nn.Linear(self.network.hidden_dims[-1] * 2 * 2, z_dim)
            self.encoder_used = "vanilla"

            print("----------Training end-to-end from scratch----------")

        self.actor = layer_init(nn.Linear(z_dim, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(z_dim, 1), std=1)

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
                max_sim = cosine.item()64
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


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, in_channels=3, h_dim=1024, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        self.conv = Encoder(in_channels)
        self.feature_size = self.conv.hidden_dims[-1] * 2 * 2
        
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, num_actions)
        )

        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + num_actions, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, self.feature_size)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        return self.inverse_net(torch.cat((state_ft, next_state_ft), 1)), self.forward_net(
            torch.cat((state_ft, action), 1)), next_state_ft