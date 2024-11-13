from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from sample.ap.data.interpolant import *
from pytorch_lightning.loggers.wandb import WandbLogger


# from analysis import metrics
# from analysis import utils as au
# from models import utils as mu
# from data.interpolant import Interpolant
# from data import utils as du
# from data import all_atom
# from data import so3_utils
# from data import residue_constants
# from experiments import utils as eu
# from sample.ap.models.flow_model import FlowModel
from sample.ap.models.base_model import BaseModel
import torch.nn as nn
from sample.ap.loss.loss import loss_fn



class BaseModule(LightningModule):

    def __init__(self, cfg):
        super(BaseModule, self).__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self.train_cfg = cfg.experiment.training
        self.cfg = cfg


        self.model = BaseModel(cfg.model)
        # self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def training_step(self, batch: Any, stage: int):
        """add interpolant ,loss """

        batch_data = self.interpolant.corrupt_batch(batch)
        batch_losses = self.model_step(batch_data)
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }

        num_batch = batch['seq'].shape[0]

        for k, v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        return batch_losses['total_loss']

    def model_step(self, batch):
        """add loss"""
        model_output = self.model(batch)
        batch_losses = loss_fn(self.cfg, batch, model_output)

        return batch_losses



    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )


    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )







