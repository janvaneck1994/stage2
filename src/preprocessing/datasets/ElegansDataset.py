import torch
from torch_geometric.data import InMemoryDataset, download_url
import networkx as nx
import numpy as np
from torch_geometric.data import Data
import random

class ElegansInteractome(InMemoryDataset):
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
        super(ElegansInteractome, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = ['interactome_elegans.csv', 'embeddings_elegans.npz']
        return files

    @property
    def processed_file_names(self):
        return 'interactome_elegans.pt'

    # def download(self):
    #     download_url(self.url + self.raw_file_names, self.raw_dir)

    def create_mask(self, nodelist_length):
        total_nodes = nodelist_length
        train_nodes = int(0.25*total_nodes)
        val_nodes = int(0.10*total_nodes)
        test_nodes = total_nodes-val_nodes-train_nodes

        idx = list(range(total_nodes))

        train_idx = [idx.pop(random.randint(0,len(idx)-1)) for i in range(train_nodes)]
        val_idx = [idx.pop(random.randint(0,len(idx)-1)) for i in range(val_nodes)]
        test_idx = idx

        mask_train = torch.zeros(total_nodes)
        mask_val = torch.zeros(total_nodes)
        mask_test = torch.zeros(total_nodes)

        mask_train[train_idx] = 1
        mask_val[val_idx] = 1
        mask_test[test_idx] = 1

        return mask_train, mask_val, mask_test

    def process(self):

        G = nx.read_edgelist(self.raw_paths[0])
        embeddings = np.load(self.raw_paths[1])

        nodelist = [x for x in G.nodes() if x in embeddings.keys()]
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

        train_mask, val_mask, test_mask = self.create_mask(len(nodelist))

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data = data if self.pre_transform is None else self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
