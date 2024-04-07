from torch.utils.data import Dataset
import torch
import math
from typing import List, Dict
import ipdb
class LSDataset(Dataset):

    def __init__(self, sequence: List, seqName: List,seqID:List, label_dict: Dict,local_rank: int):
        super(LSDataset).__init__()

        self.sequence = sequence
        self.label_dict = label_dict
        self.seqName = seqName
        self.seqID = seqID
        self.local_rank = local_rank
        # ipdb.set_trace()
    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return (self.sequence[idx], {self.seqName[idx]:self.seqID[idx]}, {k:v[idx] for k,v in self.label_dict.items()})
    
    



