from os import read
import torch
import json
from .dataset_utils import load_squad

import sys

from torch.utils.data.dataset import Dataset
# Data handling goes here
class SquadDataset(Dataset):
    def __init__(self, root_dir, debug=False):
        self.encodings = load_squad(root_dir, debug)

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

class ConvAIDataset(Dataset):
    def __init__(self, root_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

s = SquadDataset('data/squad/sq_dev.json')