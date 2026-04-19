import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from CMPNE import CMPNEncoder, prompt_generator_output, MolGraph, AllMolGraph
class DTIDataset(data.Dataset):

    def __init__(self, list_IDs,  df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        # self.CMPN = CMPNEncoder(atom_fdim=133, bond_fdim=147)
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len
    def Mol2Feature(self, smiles):
        all_ = self.Mol2graph(smiles)
        
        return all_

    def Mol2graph(self, smiles):
        # print(smiles)
        mol_graphs = []
     
        mol_graph = MolGraph(smiles)
        mol_graphs.append(mol_graph)
        return AllMolGraph(mol_graphs)
    def __getitem__(self, index):
        index = self.list_IDs[index]
        # print(f"Index: {index}, Length of dataset: {len(self.df)}") 
        # v_d = self.Mol2Feature(smiles=self.df.iloc[index]['SMILES'])
        # Obtain SMILES and convert to molecular graphs
        v_s =  self.df.iloc[index]['SMILES']
        # print(v_s)
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        
        mol_graph = self.Mol2Feature(smiles=v_s)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num, fg_num, f_fgs, fg_scope = mol_graph.get_components()

        # Process node features of molecular graphs
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        num_virtual_nodes = max(0, num_virtual_nodes)
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                       torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})
        v_d = v_d.add_self_loop()

        # Select all columns except 'SMILES' and 'class' as the protein feature vectors
        v_p = self.df.iloc[index].drop(['SMILES', 'class']).values
        v_p = torch.tensor(v_p.astype(np.float32), dtype=torch.float32)  # Convert to PyTorch tensors
     
        # Obtain labels
        y = self.df.iloc[index]['class']
        # print(v_p)

        return v_d, v_p,f_fgs,atom_num, y
    
class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
