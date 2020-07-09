import torch
from datasets.Interactomes import Interactomes

from utils.train_test_split_edges_fair import train_test_split_edges_fair

torch.manual_seed(12345)

dataset = Interactomes('../../Data/interactomes/human/', 'human')
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges_fair(data, val_ratio=0.1, test_ratio=0.2)
