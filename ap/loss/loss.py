import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import numpy as np



def loss_fn(args, batch, model_out):


    training_cfg = args.experiment.training

    pad_mask = batch['pad_mask']
    batch_size, num_res = batch['pad_mask'].shape

    # amino acid loss
    gt_aa = batch['seq']
    pred_aa = model_out['pred_aa']
    aa_loss = F.cross_entropy(
        input=pred_aa.reshape(-1, args.model.vocab_size),
        target=gt_aa.flatten().long(),
        reduction='none'
    ).reshape(batch_size, num_res)
    aa_loss = sum((aa_loss * pad_mask).sum(dim=-1) / (pad_mask.sum(dim=-1) + 1e-10))


    # amino acid accuracy
    arg_pred_aa = torch.argmax(pred_aa, dim=-1)
    aa_acc = (torch.eq(arg_pred_aa, gt_aa).float() * batch['pad_mask']).sum() / batch['pad_mask'].sum()


    pred_xyz = (batch['noise_xyz'] - model_out['pred_noise']) * batch['pad_mask'].unsqueeze(-1).unsqueeze(-1)
    gt_xyz = torch.nan_to_num(batch['xyz'])


    # guassian noise loss
    guassian_loss = F.mse_loss(model_out['pred_noise'], batch['noise_xyz'] - torch.nan_to_num(batch['xyz']), reduction='sum')

    # backbone noise loss
    pred_aa_bb = pred_xyz[:, :, :4]
    gt_aa_bb = gt_xyz[:, :, :4]
    bb_atom_loss = F.mse_loss(pred_aa_bb, gt_aa_bb, reduction='sum')

    # pairwise distance loss
    gt_flat_atoms = gt_aa_bb.reshape([batch_size, num_res * 4, 3])
    gt_pair_dists = torch.linalg.norm(
        gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
    )
    pred_flat_atoms = pred_aa_bb.reshape([batch_size, num_res * 4, 3])
    pred_pair_dists = torch.linalg.norm(
        pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
    )
    # No loss on anything >6A
    proximity_mask = gt_pair_dists < args.experiment.dist_loss_filter

    dist_mat_loss = torch.sum(
        (gt_pair_dists - pred_pair_dists) ** 2 * proximity_mask, dim=(1, 2)
    )
    dist_mat_loss = torch.nan_to_num(dist_mat_loss).mean()

    total_loss = (guassian_loss * training_cfg.guassian_loss_weight +
                  aa_loss * training_cfg.aa_loss_weight +
                  bb_atom_loss * training_cfg.aux_loss_bb_loss_weight +
                  dist_mat_loss * training_cfg.dist_mat_loss_weight)


    # print(f'total: {total_loss.item()}  guassian_loss: {guassian_loss.item()}  aa_loss: {aa_loss.item()}  bb_atom_loss: {bb_atom_loss.item()} dist_mat_loss: {dist_mat_loss.item()}')

    return {
        'guassian_loss': guassian_loss,
        'aa_loss': aa_loss,
        'bb_atom_loss': bb_atom_loss,
        'dist_mat_loss': dist_mat_loss,
        'total_loss': total_loss,
        'aa_acc': aa_acc,
    }








