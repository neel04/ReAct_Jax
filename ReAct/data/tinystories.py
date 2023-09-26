import os
from random import randint
from typing import List, Tuple

import torch
from datasets import load_dataset
from .tokenizer import Tok
from tqdm import tqdm


class TinyStoriesDataset:
    def __init__(self, split: str = 'train', max_length=32, bsz: int = 256, vocab_dir='./ReAct/data'):
        self.bsz = bsz
        
        self.dataset = load_dataset('roneneldan/TinyStories', split=split, streaming=True,
                                    ignore_verifications=True, keep_in_memory=False)
        
        self.tok = Tok(vocab_dir=vocab_dir, max_length=max_length)

    def tokenize_and_pad(self, text: str):
        encoded = self.tok.encode(text['text'])
        
        return {'text': [i.ids for i in encoded]}
    
    def _mask_seq(self, seq: List[int]) -> Tuple[List[int], int]:
        pad_idx = seq.index(0) if 0 in seq else None
        mask_token_id = 3
        
        if pad_idx is not None:
            random_idx = randint(0, pad_idx)
        else:
            random_idx = randint(0, len(seq) - 1)
        
        label = seq[random_idx]
        
        seq[random_idx] = mask_token_id # [MASK] token id
        
        # synthesize binary attention mask: 1 for real tokens, 0 for [PAD], [MASK].
        # Specials tokens are: [PAD] = 0
        attn_mask = [1 if i == mask_token_id else 0 for i in seq]
        return seq, [label], attn_mask
    
    def mask_tokens(self, text: List[List]) -> List[Tuple]:
        return [self._mask_seq(x.copy()) for x in text]
        
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