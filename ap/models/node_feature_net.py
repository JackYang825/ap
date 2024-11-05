import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        self.linear = nn.Linear(embed_size, self.c_s)

        self.aatype_embedding = nn.Embedding(22, self.c_s) # Always 21 because of 20 amino acids + 1 for unk +1 for mas



    def forward(self, aa_types, res_mask, diffuse_mask, pos):
        # s: [b]

        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.type(torch.float32).unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.aatype_embedding(aa_types),
        ]
        return self.linear(torch.cat(input_feats, dim=-1))
