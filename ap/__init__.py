# from rdkit import Chem
#
# suppl = Chem.SDMolSupplier('/nfs-userfs/yangyuxing/data/test/pcqm4m/pcqm4m-v2-train.sdf')
# for idx, mol in enumerate(suppl):
#     print(f'{idx}-th rdkit mol obj: {mol}')
#     ret = Chem.MolToPDBFile(mol, '/nfs-userfs/yangyuxing/data/test/pcqm4m/pdbs/' + f'{idx}.pdb')
#     print(ret)

# from ogb.graphproppred import PygGraphPropPredDataset
# from torch_geometric.data import DataLoader
#
# # Download and process data at './dataset/ogbg_molhiv/'
# dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = '/nfs-userfs/yangyuxing/data/test/pcqm4m/dataset')

