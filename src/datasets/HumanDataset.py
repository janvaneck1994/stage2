import torch
from torch_geometric.data import InMemoryDataset, download_url
import networkx as nx
import numpy as np
from torch_geometric.data import Data
import random

class Interactome(InMemoryDataset):
    r"""Protein-protein interaction data of different interactomes

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Human"`,
            :obj:`"Yeast"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    # url = 'github'

    def __init__(self, root, transform=None, pre_transform=None):
        super(Interactome, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = ['interactome_human.csv', 'embeddings.npz', 'interactome_human_proteins.csv']
        return files

    @property
    def processed_file_names(self):
        return 'interactome_human.pt'

    def process(self):

        G = nx.read_edgelist(self.raw_paths[0])
        embeddings = np.load(self.raw_paths[1])
        nodes = pd.read_csv(self.raw_paths[2], header=None).values.flatten()

        nodelist = [x for x in nodes if x in embeddings.keys()]
        G = nx.Graph(G.subgraph(nodelist))
        G.remove_edges_from(nx.selfloop_edges(G))
        nodemapping = {x:int(i) for i,x in enumerate(nodelist)}
        embeddings_matrix = np.array([embeddings[x] for x in nodelist])
        H = nx.relabel_nodes(G, nodemapping)
        edges = [list(x) for x in H.edges(nbunch=nodemapping.values())]
        edges = edges + [t[::-1] for t in edges]

        edge_index = torch.tensor(edges, dtype=torch.long)
        x = torch.tensor(embeddings_matrix, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous())

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data = data if self.pre_transform is None else self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
