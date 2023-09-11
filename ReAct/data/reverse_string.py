import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple

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
        src = torch.randint(0, 2, (self.complexity,)).long()
        padding = self.seqlen - self.complexity
        pad_src = torch.cat([src, torch.zeros(padding).long()])

        return pad_src, torch.flip(pad_src, (0,))

if __name__ == '__main__':
    seqlen = 8
    length = 4
    bsz = 1

    dataset = RevDataset(seqlen, length)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)

    # Example usage
    for batch in tqdm(dataloader):
        print(batch)
        break