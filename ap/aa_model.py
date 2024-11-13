import torch
import pdbio
import networkx as nx
import numpy as np
from sample.ap.utils import utils
from dataclasses import dataclass
from omegaconf import OmegaConf
from sample.ap.utils.kinematics import get_init_xyz


import rf2aa.util
from sample.ap.rf2aa.kinematics import get_chirals
from sample.ap.rf2aa.data.parsers import parse_mol
from sample.ap.rf2aa.chemical import ChemicalData as ChemData
from sample.ap.rf2aa.chemical import initialize_chemdata
initialize_chemdata(OmegaConf.create({'use_phospate_frames_for_NA': True}))

NINDEL=1
NTERMINUS=2
NMSAFULL=ChemData().NAATOKENS+NINDEL+NTERMINUS
NMSAMASKED=ChemData().NAATOKENS+ChemData().NAATOKENS+NINDEL+NINDEL+NTERMINUS

MSAFULL_N_TERM = ChemData().NAATOKENS+NINDEL
MSAFULL_C_TERM = MSAFULL_N_TERM+1

MSAMASKED_N_TERM = 2*ChemData().NAATOKENS + 2*NINDEL
MSAMASKED_C_TERM = 2*ChemData().NAATOKENS + 2*NINDEL + 1

N_TERMINUS = 1
C_TERMINUS = 2


alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def chain_letters_from_same_chain(same_chain):
    L = same_chain.shape[0]
    G = nx.from_numpy_array(same_chain.numpy())
    cc = list(nx.connected_components(G))
    cc.sort(key=min)
    chain_letters = np.chararray((L,), unicode=True)

    for ch_i, ch_name in zip(cc, alphabet):
        chain_letters[list(ch_i)] = ch_name

    return chain_letters

@dataclass
class Indep:
    seq: torch.Tensor  # [L]
    xyz: torch.Tensor  # [L, 36?, 3]
    idx: torch.Tensor

    # SM specific
    bond_feats: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    same_chain: torch.Tensor
    is_sm: torch.Tensor
    terminus_type: torch.Tensor

    def write_pdb(self, path, **kwargs):
        with open(path, kwargs.pop('file_mode', 'w')) as fh:
            return self.write_pdb_file(fh, **kwargs)

    def write_pdb_file(self, fh, **kwargs):
        seq = self.seq
        seq = torch.where(seq == 20, 0, seq)
        seq = torch.where(seq == 21, 0, seq)
        chain_letters = self.chains()
        return pdbio.writepdb_file(fh,
                                   torch.nan_to_num(self.xyz[:, :14]), seq, idx_pdb=self.idx,
                                   chain_letters=chain_letters, bond_feats=self.bond_feats[None], **kwargs)

    def chains(self):
        return chain_letters_from_same_chain(self.same_chain)


def filter_het(pdb_lines, ligand):
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        # if l[17:17+4].strip() != ligand:
        #     continue
        lines.append(l)
        try:
            hetatm_ids.append(int(l[7:7+5].strip()))
        except Exception:
            print(l)

    violations = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        if all(i in hetatm_ids for i in ids):
            lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            ligand_atms_bonded_to_protein = [i for i in ids if i in hetatm_ids]
            violations.append(f'line {l} references atom ids in the target ligand {ligand}: {ligand_atms_bonded_to_protein} and another atom')
    if violations:
        raise Exception('\n'.join(violations))
    return lines


# TODO optimize make_indep seperat
def make_indep(pdb, ligand=None, center=True):
    chirals = torch.Tensor()
    atom_frames = torch.zeros((0, 3, 2))
    target_feats = utils.parse_pdb(pdb)
    xyz_prot, mask_prot, idx_prot, seq_prot = target_feats['xyz'], target_feats['mask'], target_feats['idx'], \
    target_feats['seq']
    xyz_prot[:, 14:] = 0  # remove hydrogens
    mask_prot[:, 14:] = False
    xyz_prot = torch.tensor(xyz_prot)
    mask_prot = torch.tensor(mask_prot)
    xyz_prot[~mask_prot] = np.nan
    protein_L, nprotatoms, _ = xyz_prot.shape
    msa_prot = torch.tensor(seq_prot)[None].long()
    if ligand:
        with open(pdb, 'r') as fh:
            stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
        stream = filter_het(stream, ligand)
    if len(stream):
        mol, msa_sm, ins_sm, xyz_sm, _ = parse_mol("".join(stream), filetype="pdb", string=True)
        G = rf2aa.util.get_nxgraph(mol)
        atom_frames = rf2aa.util.get_atom_frames(msa_sm, G)
        N_symmetry, sm_L, _ = xyz_sm.shape
        Ls = [protein_L, sm_L]
        msa = torch.cat([msa_prot[0], msa_sm])[None]
        chirals = get_chirals(mol, xyz_sm[0])
        if chirals.numel() != 0:
            chirals[:, :-1] += protein_L
    else:
        Ls = [msa_prot.shape[-1], 0]
        N_symmetry = 1
        msa = msa_prot

    xyz = torch.full((N_symmetry, sum(Ls), ChemData().NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
    if ligand and len(stream):
        xyz[:, Ls[0]:, 1, :] = xyz_sm
    xyz = xyz[0]
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, Ls[0], nprotatoms)
    if len(idx_prot) == 0:
        max_idx_prot = 0
    else:
        max_idx_prot = max(idx_prot)
    # idx_sm = torch.arange(max(idx_prot), max(idx_prot) + Ls[1]) + 200
    idx_sm = torch.arange(max_idx_prot, max_idx_prot + Ls[1]) + 200
    idx_pdb = torch.concat([torch.tensor(idx_prot), idx_sm])

    seq = msa[0]

    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()

    if Ls[0] > 0:
        bond_feats[:Ls[0], :Ls[0]] = rf2aa.util.get_protein_bond_feats(Ls[0])
    if ligand and len(stream):
        bond_feats[Ls[0]:, Ls[0]:] = rf2aa.util.get_bond_feats(mol)


    same_chain = torch.zeros((sum(Ls), sum(Ls))).long()
    same_chain[:Ls[0], :Ls[0]] = 1
    same_chain[Ls[0]:, Ls[0]:] = 1
    is_sm = torch.zeros(sum(Ls)).bool()
    is_sm[Ls[0]:] = True
    assert len(Ls) <= 2, 'multi chain inference not implemented yet'
    terminus_type = torch.zeros(sum(Ls))
    terminus_type[0] = N_TERMINUS
    terminus_type[Ls[0] - 1] = C_TERMINUS

    if center:
        xyz = get_init_xyz(xyz[None, None], is_sm).squeeze()
    xyz[is_sm, 0] = 0
    xyz[is_sm, 2] = 0

    xyz = xyz[:, :14, :]

    indep = Indep(
        seq,
        xyz,
        idx_pdb,
        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain,
        is_sm,
        terminus_type)
    return indep


if __name__ == '__main__':
    pdb_path = '/yangyuxing/exp/sample/rf_diffusion_all_atom/input/7v11.pdb'
    # pdb_path = '/nfs-userfs/yangyuxing/data/bd/rep/AF-A0A009E921-F1-model_v4.pdb'
    indep = make_indep(pdb_path, ligand='q')
    print(indep)


"""
AA
{'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'UNK': 20, 'MAS': 21, 'MEN': 20}

"""



