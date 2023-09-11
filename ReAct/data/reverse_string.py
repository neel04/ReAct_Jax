from random import randint
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RevDataset(Dataset):
    complexity: int
    
    def __init__(self, seqlen: int, complexity: int, dataset_size: int = 50_000):
        '''
        Sequence length is fixed to seqlen
        Complexity is the length of the non-zero (non-padded) part of the sequence
        '''
        self.seqlen = seqlen
        self.complexity = complexity
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.complexity <= self.seqlen, 'Complexity cannot be greater than seqlen'
        
        current_complexity = randint(1, self.complexity)
        
        src = torch.randint(
            0, 2,
            (current_complexity,)
            ).long()
        
        padding = self.seqlen - current_complexity
        pad_src = torch.cat([src, torch.zeros(padding).long()])

        return pad_src, torch.flip(pad_src, (0,))

if __name__ == '__main__':
    seqlen = 16
    length = 3
    bsz = 1

    dataset = RevDataset(seqlen, length)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)

    # Example usage
    for i, batch in tqdm(enumerate(dataloader)):
        print(batch)
        print(
            batch[0].shape,
            batch[1].shape
        )
        
        break