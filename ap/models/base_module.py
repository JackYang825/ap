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

# from analysis import metrics
# from analysis import utils as au
# from models import utils as mu
# from data.interpolant import Interpolant
# from data import utils as du
# from data import all_atom
# from data import so3_utils
# from data import residue_constants
# from experiments import utils as eu
from sample.ap.models.flow_model import FlowModel
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


        # self.model = BaseModel(cfg.model)
        self.model = FlowModel(cfg.model)

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

        batch_losses = batch_losses.sum()
        print(batch_losses)
        return batch_losses

    def model_step(self, batch):
        """add loss"""
        model_output = self.model(batch)
        loss = loss_fn(self.cfg, batch, model_output)

        return loss



    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )








