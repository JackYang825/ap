import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sample.ap.openfold.utils import rigid_utils as ru
from sample.ap.data import all_atom
Rigid = ru.Rigid


def multiclass_posterior(aa_t, aa_vectorfield, t):
    if aa_t.dim() == 2:
        theta = aa_t + aa_vectorfield * (1 - t[..., None])

    elif aa_t.dim() == 3:
        theta = aa_t + aa_vectorfield * (1 - t[..., None, None])  # (N, L, K)

    elif aa_t.dim() == 4:
        theta = aa_t + aa_vectorfield * (1 - t[..., None, None, None])

    theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-10)
    return theta


def intersection_loss(protein_atom, protein_pos, ligand_pos, ligand_mask, rho=2., gamma=6.):
    n_batch, n_res, n_atom = protein_atom.shape
    aa_pos = protein_pos.reshape(n_batch, n_res * n_atom, 3).unsqueeze(2)
    ligand_pos = ligand_pos.unsqueeze(1)
    dist_mask = protein_atom.reshape(
        n_batch, n_res * n_atom, 1
    ) * ligand_mask.reshape(n_batch, 1, -1)
    dist2 = (torch.square(aa_pos - ligand_pos).sum(dim=-1) + 1e-10)
    exp_dist2 = torch.divide(-dist2, rho).exp() * dist_mask
    loss_per_aa = -rho * exp_dist2.sum(dim=-1).clamp(min=1e-20).log()
    loss_per_aa = gamma - loss_per_aa
    interior_loss = torch.clamp(loss_per_aa, min=1e-10)
    return interior_loss.reshape(n_batch, n_res, n_atom)


def compute_dist(protein_atom, protein_pos, ligand_pos, ligand_mask):
    n_batch, n_res, n_atom = protein_atom.shape
    aa_pos = protein_pos.reshape(n_batch, n_res * n_atom, 3).unsqueeze(2)
    ligand_pos = ligand_pos.unsqueeze(1)
    dist_mask = protein_atom.reshape(
        n_batch, n_res * n_atom, 1
    ) * ligand_mask.reshape(n_batch, 1, -1)
    dist = torch.linalg.norm(
        aa_pos - ligand_pos, dim=-1
    ) * dist_mask
    return dist, dist_mask


def loss_fn(args, batch, model_out):
    """Computes loss and auxiliary data.

    Args:
        batch: Batched data.
        model_out: Output of model ran on batch.

    Returns:
        loss: Final training loss scalar.
        aux_data: Additional logging data.
    """

    device = model_out["pred_aa_types"].device
    bb_mask = batch["res_mask"].to(device)
    loss_mask = bb_mask
    batch_size, num_res = bb_mask.shape

    batch_loss_mask = torch.any(bb_mask, dim=-1)

    # Backbone atom loss
    pred_atom_pos = all_atom.to_atom37(model_out['pred_trans'], model_out['pred_rotmats'])
    gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_1"].type(torch.float32))
    _, atom_mask, _, gt_atom_pos = all_atom.to_atom37_with_rigids(gt_rigids)
    gt_atom_pos = gt_atom_pos[:, :, :4]
    atom_mask = atom_mask[:, :, :4]
    pred_atom_pos = pred_atom_pos[:, :, :4]

    gt_atom_pos = gt_atom_pos.to(device)
    atom_mask = atom_mask.to(device)

    bb_atom_loss_mask = atom_mask * loss_mask[..., None]
    bb_atom_loss = torch.sum(
        (gt_atom_pos - pred_atom_pos) ** 2 * bb_atom_loss_mask[..., None],
        dim=(-1, -2, -3),
    ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
    # bb_atom_loss *= args.exp.bb_atom_loss_weight
    # bb_atom_loss *= args.exp.aux_loss_weight

    # Pairwise distance loss
    gt_flat_atoms = gt_atom_pos.reshape([batch_size, num_res * 4, 3])
    gt_pair_dists = torch.linalg.norm(
        gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
    )
    pred_flat_atoms = pred_atom_pos.reshape([batch_size, num_res * 4, 3])
    pred_pair_dists = torch.linalg.norm(
        pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
    )

    flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
    flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 4])
    flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 4))
    flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 4])

    gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
    pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
    pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

    # No loss on anything >6A
    # proximity_mask = gt_pair_dists < args.exp.dist_loss_filter
    # pair_dist_mask = pair_dist_mask * proximity_mask

    dist_mat_loss = torch.sum(
        (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
    ) / ((torch.sum(pair_dist_mask, dim=(1, 2)) + 1e-10) - num_res)
    # dist_mat_loss *= args.exp.dist_mat_loss_weight
    # dist_mat_loss *= batch["t"] > args.exp.aux_loss_t_filter
    # dist_mat_loss *= args.exp.aux_loss_weight

    # Amino acid loss
    gt_aa = batch["aatype"]
    pred_aa = model_out["pred_aa_types"]
    aa_loss = F.cross_entropy(
        input=pred_aa.reshape(-1, args.model.num_aa_type),
        target=gt_aa.flatten().long(),
        reduction="none"
    ).reshape(batch_size, num_res)

    aa_loss = (aa_loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-10)
    # aa_loss *= args.exp.aa_loss_weight


    final_loss = bb_atom_loss + dist_mat_loss + aa_loss


    return final_loss

    #
    # if args.pretrain_kd_pred:
    #     final_loss += kd_loss
    #
    # def normalize_loss(x):
    #     return x.sum() / (batch_loss_mask.sum() + 1e-10)
    #
    # aux_data = {
    #     "batch_time": batch["t"],
    #     "batch_train_loss": final_loss,
    #     "batch_aa_loss": aa_loss,
    #     "batch_kd_loss": kd_loss,
    #     "batch_rot_loss": rot_loss,
    #     "batch_trans_loss": trans_loss,
    #     "batch_bb_atom_loss": bb_atom_loss,
    #     "batch_dist_mat_loss": dist_mat_loss,
    #     "total_loss": normalize_loss(final_loss).item(),
    #     "aa_loss": normalize_loss(aa_loss).item(),
    #     "kd_loss": normalize_loss(kd_loss).item(),
    #     "rot_loss": normalize_loss(rot_loss).item(),
    #     "trans_loss": normalize_loss(trans_loss).item(),
    #     "bb_atom_loss": normalize_loss(bb_atom_loss).item(),
    #     "dist_mat_loss": normalize_loss(dist_mat_loss).item(),
    #     "bb_lig_dist_loss": normalize_loss(bb_lig_dist_loss).item(),
    #     "examples_per_step": torch.tensor(batch_size).item(),
    #     "res_length": torch.mean(torch.sum(bb_mask, dim=-1)).item(),
    # }
    #
    # assert final_loss.shape == (batch_size,)
    # assert batch_loss_mask.shape == (batch_size,)
    # return normalize_loss(final_loss), aux_data
