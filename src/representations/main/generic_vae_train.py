import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from src.representations.models.VAE import VanillaVAE_PL
from src.representations.models.BETA_VAE import BetaVAE_PL
from src.representations.models.INFO_VAE import InfoVAE_PL
from src.representations.models.MIWAE import MIWAE_PL
from  src.representations.main.vae_experiment import VAEXperiment, VAEXperiment_SEMANTIC, SemanticPredictorExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import wandb


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default="/home/mila/r/roger.creus-castanyer/modded-crafter/src/config/vae.yml")

args = parser.parse_args()

with open("/home/mila/r/roger.creus-castanyer/modded-crafter/src/config/" + args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

os.environ["WANDB_API_KEY"] = "e352fb7178eccaebef862095e4789238001ffbaf"

#wandb.init(settings=wandb.Settings(start_method="fork"))

wandb_logger = WandbLogger(
    project='crafter',
    name=config['model_params']['name'],
    tags=['vae']
)

wandb_logger.log_hyperparams(config)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model_name = config['model_params']['name']
if model_name == "BetaVAE":
    model = BetaVAE_PL(**config['model_params'])
elif model_name == "VanillaVAE":
    model = VanillaVAE_PL(**config['model_params'])
elif model_name == "InfoVAE":
    model = InfoVAE_PL(**config['model_params'])
elif model_name == "MIWAE":
    model = MIWAE_PL(**config['model_params'])
elif model_name == "SemanticPredictorExperiment":
    model = VanillaVAE_PL(**config['model_params'])

if not config['model_params']["use_semantic"]:
    if model_name != "SemanticPredictorExperiment":
        experiment = VAEXperiment(model, config['exp_params'])
    else:
        experiment = SemanticPredictorExperiment(model, config['exp_params'])
else:
    experiment = VAEXperiment_SEMANTIC(model, config['exp_params'])

runner = Trainer(logger=wandb_logger,
                 accelerator='gpu',
                 enable_checkpointing = True,
                 devices=1,
                 #default_root_dir = os.environ["SLURM_TMPDIR"] + "/tmp/checkpoints/",
                 strategy=DDPPlugin(find_unused_parameters=True),
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)