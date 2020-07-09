import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd

class Interactomes(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Computers"`,
            :obj:`"Photo"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['human']
        super(Interactomes, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = ['{}_proteome_embeddings.npz'.format(self.name),
                 '{}_interactome.edgelist'.format(self.name)]

        return files

    @property
    def processed_file_names(self):
        return '{}_interactome.pt'.format(self.name)

    def process(self):

        embeddings = np.load(self.raw_paths[0])
        proteins = embeddings.files

        ppis = pd.read_csv(self.raw_paths[1], header=None)
        ppis = ppis[ppis.isin(proteins).all(axis=1)]
        intact_prots = list(set(ppis.values.flatten()))

        embeddings_matrix = np.array([embeddings[x] for x in intact_prots])

        idx_mapping = {x:i for i,x in enumerate(intact_prots)}

        ppis_idx = ppis.applymap(lambda x: idx_mapping[x])
        reversed = ppis_idx.iloc[:, ::-1]
        reversed.columns = [0, 1]
        undirected_ppis_idx = pd.concat([ppis_idx,reversed])
        
        edges = undirected_ppis_idx.values

        edge_index = torch.tensor(edges, dtype=torch.long)
        x = torch.tensor(embeddings_matrix, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        data.protein_names = intact_prots
        data = data if self.pre_transform is None else self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())
