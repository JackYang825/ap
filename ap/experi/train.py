import os, torch
from omegaconf import DictConfig, OmegaConf
import hydra


from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from sample.ap.data.pdb_dataloader import ProteinData
# from models.flow_module import FlowModule
from sample.ap.data.datasets import PdbDataset
from sample.ap.data import utils as du
from sample.ap.models.base_module import BaseModule
from torch.utils.data import DataLoader
from sample.ap.data.utils import length_batching
import wandb

log = du.get_pylogger(__name__)



torch.set_float32_matmul_precision('high')

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = self._cfg.experiment
        self._task = self._data_cfg.task
        self._setup_dataset()
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        self._train_device_ids = du.get_available_device(self._exp_cfg.num_devices)
        self._module = BaseModule(self._cfg)


    def _setup_dataset(self):
        if self._data_cfg.dataset == 'pdb':
            self._train_dataset, self._valid_dataset = du.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task)


    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            # self._train_device_ids = [self._train_device_ids[0]]
            # self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )

            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

        trainer = Trainer(
            **self._exp_cfg.trainer,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=True,
            use_distributed_sampler=False,
            devices=self._train_device_ids,
        )

        trainer.fit(
            self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == '__main__':
    main()

