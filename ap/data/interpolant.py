import numpy as np
import torch


## TODO mask with padding index should remove
def generate_consecutive_mask(atm_batch, ratio=0.32):
    b_size, max_res = atm_batch.size(0), atm_batch.size(1)
    enc_mask = torch.zeros(b_size, max_res)
    res_mask = atm_batch.clamp(max=1).float()

    for i in range(b_size):
        non_zero = res_mask[i].sum().long().item()
        block_size = int(ratio * non_zero)
        mask = torch.ones(non_zero)
        start = torch.randint(0, non_zero - block_size + 1, (1,)).item()
        mask[start:start + block_size] = 0
        enc_mask[i, :non_zero] = mask

    return enc_mask.to(atm_batch.device)

def add_noise_to_seqs(seqs):
    return seqs
def add_guassian_noise_to_cords(attrs):
    hat_t = np.random.uniform(-1.0, 1.0, size=attrs.shape)
    hat_t = torch.tensor(hat_t).float().cuda()
    noisy_attrs = attrs + hat_t
    return noisy_attrs



class Interpolant:

    def __init__(self, cfg):
        super(Interpolant, self).__init__()
        self.cfg = cfg


    def corrupt_batch_so3(self, batch):
        enc_mask = generate_consecutive_mask(batch['aatype'])

        batch['noise_aatype'] = (batch['aatype'] * enc_mask).type(torch.int)
        batch['noise_trans_1'] = add_guassian_noise_to_cords(batch['trans_1']) * enc_mask.unsqueeze(-1)
        batch['noise_rotmats_1'] = add_guassian_noise_to_cords(batch['rotmats_1']) * enc_mask.unsqueeze(-1).unsqueeze(-1)
        return batch


    def corrupt_batch(self, batch):
        enc_mask = generate_consecutive_mask(batch['seq'])
        batch['noise_seq'] = (batch['seq'] * enc_mask).type(torch.int)

        hat_t = np.random.uniform(-1.0, 1.0, size=batch['xyz'].shape)
        hat_t = torch.tensor(hat_t).float().cuda()
        batch['noise_xyz'] = (batch['xyz'] + hat_t * (batch['atom_mask'] * batch['pad_mask'].unsqueeze(-1).unsqueeze(-1))) * enc_mask.unsqueeze(-1).unsqueeze(-1)
        batch['noise_xyz'] = torch.nan_to_num(batch['noise_xyz'])
        return batch



