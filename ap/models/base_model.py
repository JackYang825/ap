import numpy as np
import torch
from torch import nn
from sru import SRUpp
import torch.nn.functional as F
"""
frame
"""
class FrameAvergaging(nn.Module):

    def __init__(self):
        super(FrameAvergaging, self).__init__()
        self.ops = torch.tensor([
                [i, j, k] for i in [-1,1] for j in [-1, 1] for k in [-1, 1]
        ]).cuda()
    def create_frame(self, X, mask):
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask
        C = torch.bmm(X.transpose(1, 2), X)
        _, V = torch.linalg.eigh(C.detach(), 'U')

        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]

        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)

class FAEncoder(FrameAvergaging):

    def __init__(self, args):
        super(FAEncoder, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)
        self.W = nn.Linear(args.embedding_size, args.hidden_size)
        self.encoder = SRUpp(
            args.hidden_size + 3,
            args.hidden_size // 2,
            args.hidden_size // 2,
            num_layers=args.depth,
            dropout=args.dropout,
            bidirectional=True,
        )



    def forward(self, x):

        seq, cords, mask = x['noisy_seqs'], x['noisy_cords'], x['mask']

        B, N = seq.size(0), seq.size(1)
        h_X, _, _ = self.create_frame(cords[:,:,1], mask)
        seq[seq < 0] = 82
        embed_seq = self.embedding(seq)
        h_S = embed_seq.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B*8, N, -1)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N)
        h = torch.cat([h_X, h_S], dim=-1)
        h, _, _ = self.encoder(
            h.transpose(0, 1),
            mask_pad=(~mask.transpose(0, 1).bool())
        )
        h = h.transpose(0, 1).view(B, 8, N, -1)

        return h.mean(dim=1)


"""
energy
"""
class FARigidModel(nn.Module):

    def __init__(self):
        super(FARigidModel, self).__init__()

    def mean(self, X, mask):
        return (X * mask[...,None]).sum(dim=1) / mask[...,None].sum(dim=1).clamp(min=1e-6)

    def inertia(self, X, mask):
        inner = (X ** 2).sum(dim=-1)
        inner = inner[...,None,None] * torch.eye(3).to(X)[None,None,...]  # [B,N,3,3]
        outer = X.unsqueeze(2) * X.unsqueeze(3)  # [B,N,3,3]
        inertia = (inner - outer) * mask[...,None,None]
        return 0.1 * inertia.sum(dim=1)  # [B,3,3]

    def rotate(self, X, w):
        B, N = X.size(0), X.size(1)
        X = X.reshape(B, N * 14, 3)
        w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
        c = w.norm(dim=-1, keepdim=True)  # [B,N,1]
        c1 = torch.sin(c) / c.clamp(min=1e-6)
        c2 = (1 - torch.cos(c)) / (c ** 2).clamp(min=1e-6)
        cross = lambda a,b : torch.cross(a, b, dim=-1)
        X = X + c1 * cross(w, X) + c2 * cross(w, cross(w, X))
        return X.view(B, N, 14, 3)




class AllAtomEnergyModel(FARigidModel):

    def __init__(self, args):
        super(AllAtomEnergyModel, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.encoder = FAEncoder(args)
        self.W_o = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.SiLU(),
            nn.Linear(args.hidden_size, 14 * 3),
        )

    def forward(self, X):

        noisy_seqs, noisy_cords, mask = X['noisy_seqs'], X['noisy_cords'], X['mask']
        h = self.encoder(X)
        h = self.W_o(h)

        h = h.reshape(noisy_seqs.size(0), -1, 14, 3)

        total_loss = 0
        # guassian loss
        guassian_loss = F.mse_loss(h, hat_t, reduction='none')
        guassian_loss = (guassian_loss * mask.unsqueeze(-1).unsqueeze(-1)).sum()
        total_loss += guassian_loss
        # print(f'loss: {loss}')

        return {}




class BaseModel(nn.Module):

    def __init__(self, model_conf):
        super().__init__()
        self.energy_model = AllAtomEnergyModel(model_conf)

    def forward(self, input_feats):
        output = self.energy_model(input_feats)
        return output

