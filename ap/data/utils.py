import logging

import GPUtil
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Any
from openfold.utils import rigid_utils as ru
Rigid = ru.Rigid

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

LIGAND_PADDING_FEATS = ["ligand_atom", "ligand_pos", "ligand_mask"]
GUIDE_LIGAND_PADDING_ATOM_FEATS = ["guide_ligand_atom", "guide_ligand_atom_mask"]
GUIDE_LIGAND_PADDING_EDGE_FEATS = ["guide_ligand_edge", "guide_ligand_edge_index", "guide_ligand_edge_mask"]
MSA_PADDING_FEATS = ["msa_1", "msa_mask", "msa_onehot_1", "msa_vectorfield", "msa_onehot_0", "msa_onehot_t", "msa_t"]

# atom_frames 38, 3, 2
# bond_feats 274, 274
# chirals 3, 5
# idx 274
# is_sm 274
# same_chain 274, 274
# seq 274
# xyz 274, 36, 3


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    )
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    )
    return train_dataset, eval_dataset

def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]

def get_pylogger(name=__name__):
    """Initializes multi-gpu-friendly python command line logger."""

    logger = logging.getLogger(name)

    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# def pad_dim(x, pad_dim, pad_size):
#     """pad dim from 0 1 2 3"""
#     shape_len = len(x.shape)
#     iter_len = shape_len - pad_dim
#     pad_item = [0, 0] * (iter_len - 1)
#     pad_dim_size = x.shape[pad_dim]
#     pad_item = pad_item + [0, pad_size - pad_dim_size]
#     out = F.pad(x, pad_item, 'constant', -1)
#     return out
#
#
# def pad_feats(x, pad_size):
#     """pad feats with seq level"""
#     x['seq'] = pad_dim(x['seq'], 0, pad_size)
#     x['xyz'] = pad_dim(x['xyz'], 0, pad_size)
#     x['bond_feats'] = pad_dim(x['bond_feats'], 1, pad_size)
#     x['bond_feats'] = pad_dim(x['bond_feats'], 0, pad_size)
#     x['pad_mask'] = x['seq'] > 0
#     x['is_sm'] = pad_dim(x['is_sm'], 0, pad_size)
#     return x


def length_batching(np_dicts: List[Dict[str, np.ndarray]]):
    def get_len(x):
        return x['aatype'].shape

    np_dicts = [x for x in np_dicts if x is not None]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]

    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0][0]

    # padded_batch = [pad_feats(x, max_len)['xyz'] for (_, x) in dicts_by_length]
    # padded_batch = [pad_feats(x, max_len) for (_, x) in dicts_by_length]
    padded_batch = [pad_feats(x, max_len) for (_, x) in dicts_by_length]

    return torch.utils.data.default_collate(padded_batch)



def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)


def pad_feats(raw_feats, max_len, pad_msa=False, pad_guide_atom=False, pad_guide_edge=False, use_torch=False):
    if pad_msa:
        PADDING_FEATS = MSA_PADDING_FEATS

    elif pad_guide_atom:
        PADDING_FEATS = GUIDE_LIGAND_PADDING_ATOM_FEATS

    elif pad_guide_edge:
        PADDING_FEATS = GUIDE_LIGAND_PADDING_EDGE_FEATS

    else:
        PADDING_FEATS = LIGAND_PADDING_FEATS

    if pad_guide_edge:
        if "guide_ligand_edge_index" in raw_feats.keys():
            raw_feats["guide_ligand_edge_index"] = raw_feats["guide_ligand_edge_index"].transpose(0, 1)


    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        # if feat_name in PADDING_FEATS
    }


    # if pad_guide_edge:
    #     if "guide_ligand_edge_index" in padded_feats.keys():
    #         padded_feats["guide_ligand_edge_index"] = padded_feats["guide_ligand_edge_index"].transpose(1, 0)
    #
    # for feat_name in raw_feats:
    #     if feat_name not in PADDING_FEATS:
    #         padded_feats[feat_name] = raw_feats[feat_name]
    #     else:
    #         padded_feats[feat_name] = padded_feats[feat_name]

    return padded_feats



def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[
        ..., None
    ]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram



def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


if __name__ == '__main__':
    # t4d = torch.empty(3, 3, 4, 2)
    # pad_dim(t4d, 10, 0)

    pass


