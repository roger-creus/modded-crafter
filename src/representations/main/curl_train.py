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

conf = getConfig(sys.argv[1])

wandb_logger = WandbLogger(
    project='crafter',
    name=conf['experiment'],
    tags=['curl']
)

wandb_logger.log_hyperparams(conf)

curl = CURL(conf)

trainer = pl.Trainer(
    accelerator='gpu',
    enable_checkpointing = True,
    devices=1,
    max_epochs=conf['epochs'],
    logger=wandb_logger,
    default_root_dir="/home/roger/Desktop/modded-crafter/src/representations/results/" + conf['experiment']
)

trainer.fit(curl)