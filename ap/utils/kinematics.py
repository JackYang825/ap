import torch

from sample.ap.utils.chemical import INIT_CRDS

PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}

def get_init_xyz(xyz_t, is_sm):
    # input: xyz_t (B, T, L, 14, 3)
    # is_sm: [L]
    # ouput: xyz (B, T, L, 14, 3)
    B, T, L = xyz_t.shape[:3]
    #init = INIT_CRDS.to(xyz_t.device).reshape(1,1,1,36,3).repeat(B,T,L,1,1)
    init = INIT_CRDS.to(xyz_t.device).reshape(1,1,1,36,3)
    init = init.repeat(B,T,L,1,1)
    # replace small mol N and C coords with nans
    init[:,:,is_sm, 0] = torch.nan
    init[:,:,is_sm, 2] = torch.nan
    if torch.isnan(xyz_t).all():
        return init

    missing_prot_coord = torch.isnan(xyz_t[:,:,:,:3]).any(dim=-1).any(dim=-1) # (B, T, L)
    missing_sm_coord = torch.isnan(xyz_t[:,:,:,1:2]).any(dim=-1).any(dim=-1) # (B, T, L)
    mask = torch.zeros(B, T, L).bool()
    mask[..., is_sm] = missing_sm_coord[...,is_sm]
    mask[..., ~is_sm] = missing_prot_coord[...,~is_sm]

    #
    center_CA = ((~mask[:,:,:,None]) * torch.nan_to_num(xyz_t[:,:,:,1,:])).sum(dim=2) / ((~mask[:,:,:,None]).sum(dim=2)+1e-4) # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B,T,1,1,3)
    #
    idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = torch.where(~mask[i_b, i_T])[0] # (L_sub)
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap = torch.argmin(seqmap, dim=-1) # (L)
            idx = torch.gather(exist_in_templ, -1, seqmap) # (L)
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L,1).expand(-1,3))
            init[i_b,i_T] += offset_CA.reshape(L,1,3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz
