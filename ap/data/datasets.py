import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
from torch.utils.data import Dataset
from sample.ap.aa_model import make_indep
# from sample.ap.data.data import process_pdb

PDB_DATA_BASE = '/nfs-userfs/yangyuxing/data/bd/rep/'



class PdbDataset(Dataset):

    def __init__(self,
                 *,
                 dataset_cfg,
                 is_training,
                 task,
                 ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self._dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        self.csv = self.raw_csv
        self.data_base = self._dataset_cfg.data_base

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def __len__(self):
        return len(self.raw_csv)

    def __getitem__(self, row_index):
        csv_row = self.csv.iloc[row_index]
        feats = self.process_csv_row(csv_row)
        return feats

    def process_csv_row(self, csv_row):
        item_list = csv_row.values.tolist()
        item = item_list[0]
        feats = make_indep(f'{self.data_base}{item}', ligand='any')
        return {
            'seq': feats.seq,
            'xyz': feats.xyz,
            'bond_feats': feats.bond_feats,
            'is_sm': feats.is_sm,
            'pad_mask': None
        }

"""
ex:
target_feats
    seq : 236
    xyz : 236, 14, 3
    mask: 236, 14

"""
