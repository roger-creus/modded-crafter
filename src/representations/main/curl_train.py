import os
import sys
import wandb
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from IPython import embed


from src.representations.main.curl import CURL
from src.config import setSeed, getConfig

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"


os.environ["WANDB_API_KEY"] = "e352fb7178eccaebef862095e4789238001ffbaf"

conf = getConfig(sys.argv[1])

wandb_logger = WandbLogger(
    project='crafter',
    name=conf['experiment'],
    tags=['curl']
)

wandb_logger.log_hyperparams(conf)

curl = CURL(conf)

# make dir for cluster
if not os.path.exists(os.environ["SLURM_TMPDIR"] + "/tmp/checkpoints/"):
    os.makedirs(os.environ["SLURM_TMPDIR"] + "/tmp/checkpoints/")

trainer = pl.Trainer(
    accelerator='gpu',
    enable_checkpointing = True,
    devices=1,
    max_epochs=conf['epochs'],
    logger=wandb_logger,
    #default_root_dir="/home/roger/Desktop/modded-crafter/src/representations/results/" + conf['experiment']
    default_root_dir = os.environ["SLURM_TMPDIR"] + "/tmp/checkpoints/"
)

trainer.fit(curl)