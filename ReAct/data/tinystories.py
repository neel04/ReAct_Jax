import os
from typing import List, Tuple

import torch
from datasets import load_dataset
from .tokenizer import Tok
from tqdm import tqdm


class TinyStoriesDataset:
    def __init__(self, split: str = 'train', max_length=32, bsz: int = 256, vocab_dir='./ReAct/data'):
        self.bsz = bsz
        self.max_length = max_length + 1
        
        self.dataset = load_dataset('roneneldan/TinyStories', split=split, ignore_verifications=True, 
                                    keep_in_memory=False).to_iterable_dataset()
        
        self.tok = Tok(vocab_dir=vocab_dir, max_length=self.max_length)

    def tokenize_and_pad(self, text: str):
        encoded = self.tok.encode(text['text'])
        
        return {'text': [i.ids for i in encoded]}
    
    def _shift_seq(self, seq: List[int]) -> Tuple[List]:
        input_seq = torch.tensor(seq) # [1, 2, 3, 4, 5]
        targets = torch.roll(input_seq, shifts=-1) # [2, 3, 4, 5, 1]
        # remove last elements for both
        input_seq, targets = input_seq[:-1], targets[:-1] # [1, 2, 3, 4], [2, 3, 4, 5]
        # pad mask for normal tokens (1) and 0 for padding token (0)
        pad_mask = torch.Tensor([1 if i != 0 else 0 for i in input_seq])
        
        return input_seq.tolist(), targets.tolist(), pad_mask.tolist()

    def shift_tokens(self, text: List[List]) -> List[Tuple]:
        return [self._shift_seq(x.copy()) for x in text]
        
    @staticmethod
    def collate_fn(batch: List[dict]) -> List:
        # batch is a list of dicts, each with a key 'text'
        # collate them into a nested list
        nested_list = list(map(lambda x: x['text'], batch))
        
        return nested_list        

    def create_dataloader(self):
        self.dataset = self.dataset.map(self.tokenize_and_pad, batched=True, batch_size=self.bsz)
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.bsz,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            prefetch_factor=16)
        
        return self.dataloader

if __name__ == '__main__':
    dataset = TinyStoriesDataset(bsz=256)
    dataloader = dataset.create_dataloader()
    
    for idx, batch in tqdm(enumerate(dataloader)):
        out = dataset.mask_tokens(batch)
        #print(f'Input: {out[0][0]} | Label: {out[0][1]}')
        #print(f'\nDecoded input: {dataset.tok.decode(out[0][0])} | Decoded label: {dataset.tok.decode(out[0][1])}')
        #print(f'Input: {out[1][0]} | Label: {out[1][1]}')
        #print(f'\nDecoded input: {dataset.tok.decode(out[1][0])} | Decoded label: {dataset.tok.decode(out[1][1])}')