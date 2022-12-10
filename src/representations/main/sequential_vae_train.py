import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from src.representations.models.SEQ_VAE import SEQ_VAE
from src.representations.main.seq_vae_experiment import SEQ_VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
    tags=['seq_vae']
)

wandb_logger.log_hyperparams(config)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model_name = config['model_params']['name']

model = SEQ_VAE(**config['model_params'])
experiment = SEQ_VAEXperiment(model, config['exp_params'])

runner = Trainer(logger=wandb_logger,
                 accelerator='gpu',
                 enable_checkpointing = True,
                 devices=1,
                 #default_root_dir = os.environ["SLURM_TMPDIR"] + "/tmp/checkpoints/",
                 #strategy=DDPStrategy(find_unused_parameters=True),
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)